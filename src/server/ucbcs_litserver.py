import math
import random
import logging

import numpy as np

from .fedavgserver import FedavgServer

logger = logging.getLogger(__name__)


class Ucbcs_LitServer(FedavgServer):
    """UCB-CS-LIT: faithful Algorithm 1 of Cho, Gupta, Joshi, Yagan (2020).

    Discounted-UCB client selection. Each round picks m = max(C*K, 1) clients
    by score A_t(gamma, k) = p_k * (mu_k + U_k), where
        mu_k  = L_t(gamma,k) / N_t(gamma,k)            (discounted mean reward)
        U_k   = sqrt(2 * sigma_t^2 * log T_t(gamma) / N_t(gamma,k))
        T_t   = sum_{t'<=t} gamma^(t-t')               (discounted horizon)
        sigma_t = running max of std(selected-client losses).
    Discount factor reuses --ucb_gamma (default 0.9 in the LIT scripts).

    Reward signal: pre-training loss of client k on the current global model
    (single batch, via FedavgClient.get_loss). The paper averages the loss
    over the tau local steps; pre-training loss is a clean proxy used by the
    existing UCB-CS path (--ucb_signal loss) and keeps shared code paths
    untouched.
    """

    def __init__(self, args, writer, server_dataset, client_datasets, model):
        super().__init__(args=args, writer=writer, server_dataset=server_dataset,
                         client_datasets=client_datasets, model=model)
        K = int(self.args.K)
        self.lit_L = np.zeros(K, dtype=np.float64)
        self.lit_N = np.zeros(K, dtype=np.float64)
        self.lit_T = 0.0
        self.lit_sigma_max = 1.0
        self.lit_A = np.zeros(K, dtype=np.float64)

        sizes = np.array([len(self.clients[k].training_set) for k in range(K)], dtype=np.float64)
        total = float(sizes.sum())
        self.lit_p = sizes / total if total > 0 else np.full(K, 1.0 / K)

        logger.info(
            f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] '
            f'[Round: {str(self.round).zfill(4)}] UCB-CS-LIT initialized '
            f'(K={K}, m={max(int(self.args.C * K), 1)}, gamma={float(getattr(self.args, "ucb_gamma", 0.9))}).'
        )

    def _lit_sample_candidates(self):
        K = int(self.args.K)
        m = max(int(self.args.C * K), 1)
        gamma = float(getattr(self.args, 'ucb_gamma', 0.9))

        if self.lit_T <= 0.0:
            chosen = random.sample(range(K), m)
            logger.info(
                f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] '
                f'[Round: {str(self.round).zfill(4)}] UCB-CS-LIT cold-start: '
                f'picked {m} clients uniformly at random.'
            )
            return sorted(chosen)

        eps = 1e-12
        log_T = math.log(max(self.lit_T, 1.0))
        N_safe = np.maximum(self.lit_N, eps)
        mean = self.lit_L / N_safe
        bonus = np.sqrt(2.0 * (self.lit_sigma_max ** 2) * log_T / N_safe)
        scores = self.lit_p * (mean + bonus)

        unvisited = self.lit_N < eps
        if unvisited.any():
            scores = np.where(unvisited, np.inf, scores)

        self.lit_A = scores

        order = np.argsort(-scores, kind='stable')
        top = scores[order[:m]]
        tie_value = top[-1]
        forced = [i for i in order[:m] if scores[i] > tie_value]
        ties = [i for i in range(K) if scores[i] == tie_value]
        random.shuffle(ties)
        chosen = forced + ties[:m - len(forced)]

        logger.info(
            f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] '
            f'[Round: {str(self.round).zfill(4)}] UCB-CS-LIT: picked {m} clients '
            f'(T={self.lit_T:.3f}, sigma_max={self.lit_sigma_max:.4f}, gamma={gamma}, '
            f'unvisited remaining={int(unvisited.sum())}).'
        )
        return sorted(int(c) for c in chosen)

    def _lit_update_state(self, selected_ids, client_losses):
        gamma = float(getattr(self.args, 'ucb_gamma', 0.9))

        self.lit_L *= gamma
        self.lit_N *= gamma
        self.lit_T = gamma * self.lit_T + 1.0

        losses_vec = []
        for cid in selected_ids:
            r = float(client_losses.get(cid, 0.0))
            self.lit_L[cid] += r
            self.lit_N[cid] += 1.0
            losses_vec.append(r)

        if len(losses_vec) >= 2:
            std = float(np.std(losses_vec))
            if std > self.lit_sigma_max:
                self.lit_sigma_max = std

    def update(self):
        if getattr(self.args, 'concept_drift', False):
            if self.args.drift_start <= self.round <= (self.args.drift_start + self.args.drift_duration):
                self._drift_dataset()

        if getattr(self.args, 'drift_adaptation', False) and self.args.drift_adaptation_mode == 'custom':
            all_ids = [c.id for c in self.clients]
            self._request(all_ids, eval=False, participated=True, retain_model=True, save_raw=False, lr_update=True)
            for cid in all_ids:
                if self.clients[cid].model is not None:
                    self.clients[cid].model = None

        selected_ids = self._lit_sample_candidates()

        client_losses = self._request(selected_ids, eval=False, participated=True,
                                      retain_model=True, save_raw=False, get_loss=True)
        for cid in selected_ids:
            if self.clients[cid].model is not None:
                self.clients[cid].model = None

        if getattr(self.args, 'drift_adaptation', False) and self.args.drift_adaptation_mode == 'original':
            self.adapted_lr = self.lr_estimator.estimate(self.global_model, self.round, self.curr_lr)

        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)
        _ = self._request(selected_ids, eval=True, participated=True, retain_model=True, save_raw=False)

        self._lit_update_state(selected_ids, client_losses)

        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer = self._aggregate(server_optimizer, selected_ids, updated_sizes)
        server_optimizer.step()

        if self.round % self.args.lr_decay_step == 0:
            self.curr_lr *= self.args.lr_decay
        return selected_ids

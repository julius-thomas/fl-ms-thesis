import os
import gc
import json
import torch
import random
import logging
import numpy as np
import concurrent.futures

from importlib import import_module
from collections import ChainMap, defaultdict

import kornia as K

from src import init_weights, TqdmToLogger, MetricManager
from src.algorithm.learningrate_estimator import LearningrateEstimatorModel
from .baseserver import BaseServer

logger = logging.getLogger(__name__)



class FedavgServer(BaseServer):
    def __init__(self, args, writer, server_dataset, client_datasets, model):
        super(FedavgServer, self).__init__()
        self.args = args
        self.writer = writer

        self.round = 0 # round indicator
        if self.args.eval_type != 'local': # global holdout set for central evaluation
            self.server_dataset = server_dataset
        self.global_model = self._init_model(model) # global model
        self.criterion = torch.nn.__dict__[self.args.criterion]() # loss function
        self.opt_kwargs = dict(lr=self.args.lr, momentum=self.args.beta1) # federation algorithm arguments
        self.curr_lr = self.args.lr # learning rate
        self.clients = self._create_clients(client_datasets) # clients container
        self.results = defaultdict(dict) # logging results container

        # learning rate adaptation (model-based estimator for "original" mode)
        if getattr(self.args, 'drift_adaptation', False):
            self.lr_estimator = LearningrateEstimatorModel(
                self.args.lr, self.args.b1, self.args.b2, self.args.b3
            )
            self.adapted_lr = 0

        # concept drift state
        self.drift_std = 0.0

    def _init_model(self, model):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Initialize a model!')
        init_weights(model, self.args.init_type, self.args.init_gain)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully initialized the model ({self.args.model_name}; (Initialization type: {self.args.init_type.upper()}))!')
        return model
    
    def _get_algorithm(self, model, **kwargs):
        ALGORITHM_CLASS = import_module(f'..algorithm.{self.args.algorithm}', package=__package__).__dict__[f'{self.args.algorithm.title()}Optimizer']
        optimizer = ALGORITHM_CLASS(params=model.parameters(), **kwargs)
        if self.args.algorithm != 'fedsgd': 
            optimizer.add_param_group(dict(params=list(self.global_model.buffers()))) # add buffered tensors (i.e., gamma and beta of batchnorm layers)
        return optimizer

    def _create_clients(self, client_datasets):
        CLINET_CLASS = import_module(f'..client.{self.args.algorithm}client', package=__package__).__dict__[f'{self.args.algorithm.title()}Client']

        def __create_client(identifier, datasets):
            client = CLINET_CLASS(args=self.args, training_set=datasets[0], test_set=datasets[-1])
            client.id = identifier
            return client

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Create clients!')
        clients = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(int(self.args.K), os.cpu_count() - 1)) as workhorse:
            for identifier, datasets in TqdmToLogger(
                enumerate(client_datasets), 
                logger=logger, 
                desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...creating clients... ',
                total=len(client_datasets)
                ):
                clients.append(workhorse.submit(__create_client, identifier, datasets).result())            
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...sucessfully created {self.args.K} clients!')
        return clients

    def _sample_clients(self, exclude=[]):
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Sample clients!')
        if exclude == []: # Update - randomly select max(floor(C * K), 1) clients
            num_sampled_clients = max(int(self.args.C * self.args.K), 1)
            sampled_client_ids = sorted(random.sample(range(self.args.K), num_sampled_clients))
        else: # Evaluation - randomly select unparticipated clients in amount of `eval_fraction` multiplied
            num_unparticipated_clients = self.args.K - len(exclude)
            if num_unparticipated_clients == 0: # when C = 1, i.e., need to evaluate on all clients
                num_sampled_clients = self.args.K
                sampled_client_ids = list(range(self.args.K))
            else:
                num_sampled_clients = max(int(self.args.eval_fraction * num_unparticipated_clients), 1)
                eligible = [i for i in range(self.args.K) if i not in exclude]
                sampled_client_ids = sorted(random.sample(eligible, num_sampled_clients))
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...{num_sampled_clients} clients are selected!')
        return sampled_client_ids

    def _log_results(self, resulting_sizes, results, eval, participated, save_raw):
        losses, metrics, num_samples = list(), defaultdict(list), list()
        for identifier, result in results.items():
            client_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [CLIENT] < {str(identifier).zfill(6)} > '
            if eval: # get loss and metrics
                # loss
                loss = result['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for metric, value in result['metrics'].items():
                    client_log_string += f'| {metric}: {value:.4f} '
                    metrics[metric].append(value)
            else: # same, but retireve results of last epoch's
                # loss
                loss = result[self.args.E]['loss']
                client_log_string += f'| loss: {loss:.4f} '
                losses.append(loss)
                
                # metrics
                for name, value in result[self.args.E]['metrics'].items():
                    client_log_string += f'| {name}: {value:.4f} '
                    metrics[name].append(value)                
            # get sample size
            num_samples.append(resulting_sizes[identifier])

            # log per client
            logger.debug(client_log_string)
        else:
            num_samples = np.array(num_samples).astype(float)

        # aggregate into total logs
        result_dict = defaultdict(dict)
        total_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [{"EVALUATE" if eval else "UPDATE"}] [SUMMARY] ({len(resulting_sizes)} clients):'

        # loss
        losses_array = np.array(losses).astype(float)
        weighted = losses_array.dot(num_samples) / sum(num_samples); std = losses_array.std()
        
        top10_indices = np.argpartition(losses_array, -int(0.1 * len(losses_array)))[-int(0.1 * len(losses_array)):] if len(losses_array) > 1 else 0
        top10 = np.atleast_1d(losses_array[top10_indices])
        top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

        bot10_indices = np.argpartition(losses_array, max(1, int(0.1 * len(losses_array)) - 1))[:max(1, int(0.1 * len(losses_array)))] if len(losses_array) > 1 else 0
        bot10 = np.atleast_1d(losses_array[bot10_indices])
        bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

        total_log_string += f'\n    - Loss: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
        result_dict['loss'] = {
            'avg': weighted.astype(float), 'std': std.astype(float), 
            'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
            'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
        }

        if save_raw:
            result_dict['loss']['raw'] = losses

        self.writer.add_scalars(
            f'Local {"Test" if eval else "Training"} Loss ' + eval * f'({"In" if participated else "Out"})',
            {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
            self.round
        )

        # metrics
        for name, val in metrics.items():
            val_array = np.array(val).astype(float)
            weighted = val_array.dot(num_samples) / sum(num_samples); std = val_array.std()
            
            top10_indices = np.argpartition(val_array, -int(0.1 * len(val_array)))[-int(0.1 * len(val_array)):] if len(val_array) > 1 else 0
            top10 = np.atleast_1d(val_array[top10_indices])
            top10_mean, top10_std = top10.dot(np.atleast_1d(num_samples[top10_indices])) / num_samples[top10_indices].sum(), top10.std()

            bot10_indices = np.argpartition(val_array, max(1, int(0.1 * len(val_array)) - 1))[:max(1, int(0.1 * len(val_array)))] if len(val_array) > 1 else 0
            bot10 = np.atleast_1d(val_array[bot10_indices])
            bot10_mean, bot10_std = bot10.dot(np.atleast_1d(num_samples[bot10_indices])) / num_samples[bot10_indices].sum(), bot10.std()

            total_log_string += f'\n    - {name.title()}: Avg. ({weighted:.4f}) Std. ({std:.4f}) | Top 10% ({top10_mean:.4f}) Std. ({top10_std:.4f}) | Bottom 10% ({bot10_mean:.4f}) Std. ({bot10_std:.4f})'
            result_dict[name] = {
                'avg': weighted.astype(float), 'std': std.astype(float), 
                'top10p_avg': top10_mean.astype(float), 'top10p_std': top10_std.astype(float), 
                'bottom10p_avg': bot10_mean.astype(float), 'bottom10p_std': bot10_std.astype(float)
            }
                
            if save_raw:
                result_dict[name]['raw'] = val

            self.writer.add_scalars(
                f'Local {"Test" if eval else "Training"} {name.title()}' + eval * f' ({"In" if participated else "Out"})',
                {'Avg.': weighted, 'Std.': std, 'Top 10% Avg.': top10_mean, 'Top 10% Std.': top10_std, 'Bottom 10% Avg.': bot10_mean, 'Bottom 10% Std.': bot10_std},
                self.round
            )
            self.writer.flush()
        
        # log total message
        logger.info(total_log_string)
        return result_dict

    def _request(self, ids, eval, participated, retain_model, save_raw, lr_update=False, get_loss=False):
        def __set_drift(client):
            client.drift_std = float(getattr(self, 'drift_std', 0.0))
            client.drift_mode = getattr(self.args, 'drift_mode', 'hard')

        def __update_clients(client):
            if client.model is None:
                client.download(self.global_model)
            __set_drift(client)
            # set learning rate based on adaptation mode
            if getattr(self.args, 'drift_adaptation', False) and self.args.drift_adaptation_mode == 'original':
                client.args.lr = self.adapted_lr
            else:
                client.args.lr = self.curr_lr
            client.round = self.round
            update_result = client.update()
            return {client.id: len(client.training_set)}, {client.id: update_result}

        def __evaluate_clients(client):
            if client.model is None:
                client.download(self.global_model)
            __set_drift(client)
            eval_result = client.evaluate()
            if not retain_model:
                client.model = None
            return {client.id: len(client.test_set)}, {client.id: eval_result}

        def __update_lr(client):
            client.args.lr = self.curr_lr
            client.round = self.round
            lr_result = client.update_estimator()
            return lr_result

        def __get_loss(client):
            if client.model is None:
                client.download(self.global_model)
            __set_drift(client)
            losses = client.get_loss()
            return {client.id: losses}

        # LR update mode (for "custom" drift adaptation)
        if lr_update:
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request lr updates to {len(ids)} clients!')
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), torch.cuda.device_count() or 1)) as workhorse:
                for idx in TqdmToLogger(ids, logger=logger,
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...update lr... ',
                    total=len(ids)):
                    results.append(workhorse.submit(__update_lr, self.clients[idx]).result())
            return np.mean(results)

        # Get loss mode (for active sampling)
        if get_loss:
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request loss evaluation to {len(ids)} clients!')
            results = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), torch.cuda.device_count() or 1)) as workhorse:
                for idx in TqdmToLogger(ids, logger=logger,
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...get losses... ',
                    total=len(ids)):
                    results.append(workhorse.submit(__get_loss, self.clients[idx]).result())
            return {x: r[x] for r in results for x in r}

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Request {"updates" if not eval else "evaluation"} to {"all" if ids is None else len(ids)} clients!')
        if eval:
            if self.args.train_only:
                return None
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), torch.cuda.device_count() or 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids,
                    logger=logger,
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...evaluate clients... ',
                    total=len(ids)
                    ):
                    jobs.append(workhorse.submit(__evaluate_clients, self.clients[idx])) 
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            _eval_sizes, _eval_results = list(map(list, zip(*results)))
            _eval_sizes, _eval_results = dict(ChainMap(*_eval_sizes)), dict(ChainMap(*_eval_results))
            self.results[self.round][f'clients_evaluated_{"in" if participated else "out"}'] = self._log_results(
                _eval_sizes, 
                _eval_results, 
                eval=True, 
                participated=participated,
                save_raw=save_raw
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed evaluation of {"all" if ids is None else len(ids)} clients!')
            return None
        else:
            jobs, results = [], []
            with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(ids), torch.cuda.device_count() or 1)) as workhorse:
                for idx in TqdmToLogger(
                    ids,
                    logger=logger,
                    desc=f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...update clients... ',
                    total=len(ids)
                    ):
                    jobs.append(workhorse.submit(__update_clients, self.clients[idx])) 
                for job in concurrent.futures.as_completed(jobs):
                    results.append(job.result())
            update_sizes, _update_results = list(map(list, zip(*results)))
            update_sizes, _update_results = dict(ChainMap(*update_sizes)), dict(ChainMap(*_update_results))
            self.results[self.round]['clients_updated'] = self._log_results(
                update_sizes, 
                _update_results, 
                eval=False, 
                participated=True,
                save_raw=False
            )
            logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...completed updates of {"all" if ids is None else len(ids)} clients!')
            return update_sizes
    
    def _aggregate(self, server_optimizer, ids, updated_sizes):
        assert set(updated_sizes.keys()) == set(ids)
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Aggregate updated signals!')
        
        # calculate mixing coefficients according to sample sizes
        coefficients = {identifier: float(nuemrator / sum(updated_sizes.values())) for identifier, nuemrator in updated_sizes.items()}
        
        # accumulate weights
        for identifier in ids:
            local_layers_iterator = self.clients[identifier].upload()
            server_optimizer.accumulate(coefficients[identifier], local_layers_iterator)
            self.clients[identifier].model = None
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...successfully aggregated into a new gloal model!')
        return server_optimizer

    def _apply_gpu_drift(self, x):
        """Apply Gaussian blur drift on GPU tensors."""
        std = float(getattr(self, 'drift_std', 0.0))
        if std <= 0.0:
            return x
        mode = getattr(self.args, 'drift_mode', 'hard')
        if mode == 'soft':
            ksize, sigma = (7, 7), max(1e-3, 2.0 * std)
        else:  # 'hard'
            ksize, sigma = (11, 11), max(1e-3, 5.0 * std)
        return K.filters.gaussian_blur2d(x, ksize, (sigma, sigma)).clamp(0.0, 1.0)

    def _drift_dataset(self):
        """Apply concept drift to datasets based on drift mode."""
        if self.args.drift_mode in ['soft', 'hard']:
            self._update_drift_strength()
        elif self.args.drift_mode == 'sudden' and self.args.drift_start == self.round:
            # activate sudden label-swap drift
            self.clients[0].training_set.subset.dataset.dataset.dataset.sudden_drift = True
            if self.args.eval_type != 'local':
                self.server_dataset.dataset.sudden_drift = True

    def _update_drift_strength(self):
        """Compute incremental drift strength (0..1) based on current round."""
        self.drift_std = 0.0
        if not getattr(self.args, 'concept_drift', False):
            return
        start = int(getattr(self.args, 'drift_start', 0))
        dur = int(getattr(self.args, 'drift_duration', 0))
        if dur <= 0:
            return
        if start <= self.round <= (start + dur):
            self.drift_std = max(0.0, min(1.0, (self.round - start) / dur))

    def _get_highest_loss(self, client_losses):
        """Select clients by loss: 'max' picks top-K, 'stoch' uses Boltzmann softmax sampling."""
        X = int(len(client_losses) * self.args.sampling_fraction)

        if self.args.sampling_type == 'stoch':
            ids = list(client_losses.keys())
            losses = np.array(list(client_losses.values()))
            # Boltzmann softmax
            e_x = np.exp(losses / self.args.temp)
            distr = e_x / np.sum(e_x)
            ids = np.random.choice(ids, size=X, p=distr, replace=False).tolist()
        else:  # 'max'
            sorted_clients = sorted(client_losses.items(), key=lambda x: x[1])
            ids = [cid for cid, _ in sorted_clients[-X:]]

        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...selected {len(ids)} clients for active sampling!')
        return ids

    @torch.no_grad()
    def _central_evaluate(self):
        mm = MetricManager(self.args.eval_metrics)
        self.global_model.eval()
        self.global_model.to(self.args.device)

        _cuda = 'cuda' in self.args.device
        _dl_extra = dict(num_workers=4, prefetch_factor=2) if _cuda else {}
        for inputs, targets in torch.utils.data.DataLoader(dataset=self.server_dataset, batch_size=self.args.B, shuffle=False, pin_memory=_cuda, **_dl_extra):
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            inputs = self._apply_gpu_drift(inputs)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.bf16):
                outputs = self.global_model(inputs)
                loss = self.criterion(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            self.global_model.to('cpu')
            mm.aggregate(len(self.server_dataset))

        # log result
        result = mm.results
        server_log_string = f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] [EVALUATE] [SERVER] '

        ## loss
        loss = result['loss']
        server_log_string += f'| loss: {loss:.4f} '
        
        ## metrics
        for metric, value in result['metrics'].items():
            server_log_string += f'| {metric}: {value:.4f} '
        logger.info(server_log_string)

        # log learning rates
        self.writer.add_scalar('Learning Rate', self.curr_lr, self.round)

        # log TensorBoard
        self.writer.add_scalar('Server Loss', loss, self.round)
        for name, value in result['metrics'].items():
            self.writer.add_scalar(f'Server {name.title()}', value, self.round)
        else:
            self.writer.flush()
        self.results[self.round]['server_evaluated'] = result

    def update(self):
        """Update the global model through federated learning.
        """
        ##################
        # Concept Drift  #
        ##################
        if getattr(self.args, 'concept_drift', False):
            if self.args.drift_start <= self.round <= (self.args.drift_start + self.args.drift_duration):
                self._drift_dataset()

        ########################
        # LR Adaptation: custom (loss-based, per-client estimators)
        ########################
        if getattr(self.args, 'drift_adaptation', False) and self.args.drift_adaptation_mode == 'custom':
            all_ids = [c.id for c in self.clients]
            self._request(all_ids, eval=False, participated=True, retain_model=True, save_raw=False, lr_update=True)
            for cid in all_ids:
                if self.clients[cid].model is not None:
                    self.clients[cid].model = None

        #################
        # Client Update #
        #################
        selected_ids = self._sample_clients() # randomly select clients

        ######################
        # Active Sampling    #
        ######################
        if getattr(self.args, 'active_sampling', False):
            # get losses from sampled clients to re-select by loss
            client_losses = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False, get_loss=True)
            for cid in selected_ids:
                if self.clients[cid].model is not None:
                    self.clients[cid].model = None
            selected_ids = self._get_highest_loss(client_losses)

        ########################
        # LR Adaptation: original (model-based, server-side estimator)
        ########################
        if getattr(self.args, 'drift_adaptation', False) and self.args.drift_adaptation_mode == 'original':
            self.adapted_lr = self.lr_estimator.estimate(self.global_model, self.round, self.curr_lr)

        # request update & evaluation to selected clients
        updated_sizes = self._request(selected_ids, eval=False, participated=True, retain_model=True, save_raw=False)
        _ = self._request(selected_ids, eval=True, participated=True, retain_model=True, save_raw=False)

        #################
        # Server Update #
        #################
        server_optimizer = self._get_algorithm(self.global_model, **self.opt_kwargs)
        server_optimizer.zero_grad(set_to_none=True)
        server_optimizer = self._aggregate(server_optimizer, selected_ids, updated_sizes)
        server_optimizer.step()

        if self.round % self.args.lr_decay_step == 0: # update learning rate
            self.curr_lr *= self.args.lr_decay
        return selected_ids

    def evaluate(self, excluded_ids):
        """Evaluate the global model located at the server.
        """
        ##############
        # Evaluation #
        ##############
        if self.args.eval_type != 'global': # `local` or `both`: evaluate on selected clients' holdout set
            selected_ids = self._sample_clients(exclude=excluded_ids)
            _ = self._request(selected_ids, eval=True, participated=False, retain_model=False, save_raw=self.round == self.args.R)
        if self.args.eval_type != 'local': # `global` or `both`: evaluate on the server's global holdout set
            self._central_evaluate()

        # calculate generalization gap
        if (not self.args.train_only) and (not self.args.eval_type == 'global'):
            gen_gap = dict()
            curr_res = self.results[self.round]
            for key in curr_res['clients_evaluated_out'].keys():
                for name in curr_res['clients_evaluated_out'][key].keys():
                    if 'avg' in name:
                        gap = curr_res['clients_evaluated_out'][key][name] - curr_res['clients_evaluated_in'][key][name]
                        gen_gap[f'gen_gap_{key}'] = {name: gap}
                        self.writer.add_scalars(f'Generalization Gap ({key.title()})', gen_gap[f'gen_gap_{key}'], self.round)
                        self.writer.flush()
            else:
                self.results[self.round]['generalization_gap'] = dict(gen_gap)

    def finalize(self):
        """Save results.
        """
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] Save results and the global model checkpoint!')
        with open(os.path.join(self.args.result_path, f'{self.args.exp_name}.json'), 'w', encoding='utf8') as result_file: # save results
            results = {key: value for key, value in self.results.items()}
            json.dump(results, result_file, indent=4)
        torch.save(self.global_model.state_dict(), os.path.join(self.args.result_path, f'{self.args.exp_name}.pt')) # save model checkpoint
        self.writer.close()
        logger.info(f'[{self.args.algorithm.upper()}] [{self.args.dataset.upper()}] [Round: {str(self.round).zfill(4)}] ...finished federated learning!')
        if self.args.use_tb:
            input('[FINISH] ...press <Enter> to exit after tidying up your TensorBoard logging!')

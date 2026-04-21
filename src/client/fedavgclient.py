import copy
import torch
import inspect
import itertools
import logging
import numpy as np

import kornia as K

from .baseclient import BaseClient
from src import MetricManager
from src.algorithm.learningrate_estimator import LearningrateEstimatorLoss

logger = logging.getLogger(__name__)


class FedavgClient(BaseClient):
    def __init__(self, args, training_set, test_set):
        super(FedavgClient, self).__init__()
        self.args = args
        self.training_set = training_set
        self.test_set = test_set

        self.optim = torch.optim.__dict__[self.args.optimizer]
        self.criterion = torch.nn.__dict__[self.args.criterion]()

        self.batch_size = self.args.B if self.args.B > 0 else len(self.training_set)
        self.train_loader = self._create_dataloader(self.training_set, shuffle=not self.args.no_shuffle)
        self.test_loader = self._create_dataloader(self.test_set, shuffle=False)

        # learning rate adaptation (loss-based estimator for "custom" mode)
        if getattr(self.args, 'drift_adaptation', False):
            self.estimator = LearningrateEstimatorLoss(
                initial_lr=self.args.lr,
                b1=self.args.b1, b2=self.args.b2, b3=self.args.b3
            )
        self.round = 0

        # concept drift state
        self.drift_std = 0.0
        self.drift_mode = getattr(self.args, 'drift_mode', 'hard')

    def _refine_optim_args(self, args):
        required_args = inspect.getfullargspec(self.optim)[0]

        # collect eneterd arguments
        refined_args = {}
        for argument in required_args:
            if hasattr(args, argument): 
                refined_args[argument] = getattr(args, argument)
        return refined_args

    def _create_dataloader(self, dataset, shuffle):
        pin = 'cuda' in self.args.device
        return torch.utils.data.DataLoader(dataset=dataset, batch_size=self.batch_size, shuffle=shuffle, pin_memory=pin, num_workers=getattr(self.args, 'num_workers', 0))

    def _apply_gpu_drift(self, x):
        """Apply Gaussian blur drift on GPU tensors."""
        std = float(getattr(self, 'drift_std', 0.0))
        if std <= 0.0:
            return x
        mode = getattr(self, 'drift_mode', 'hard')
        if mode == 'soft':
            ksize, sigma = (7, 7), max(1e-3, 2.0 * std)
        else:  # 'hard'
            ksize, sigma = (11, 11), max(1e-3, 5.0 * std)
        return K.filters.gaussian_blur2d(x, ksize, (sigma, sigma)).clamp(0.0, 1.0)

    @torch.inference_mode()
    def get_loss(self):
        """Compute loss on a single training batch (used for active sampling)."""
        self.model.eval()
        self.model.to(self.args.device)
        inputs, targets = next(iter(self.train_loader))
        inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
        inputs = self._apply_gpu_drift(inputs)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.bf16):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
        return loss.item()

    @torch.inference_mode()
    def update_estimator(self):
        """Update the loss-based LR estimator and return the adapted learning rate."""
        self.model.eval()
        self.model.to(self.args.device)
        if self.estimator.id is None:
            self.estimator.id = self.id

        batch_count, loss_arr = 0, []
        while batch_count <= 50:
            inputs, targets = next(iter(self.train_loader))
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.bf16):
                outputs = self.model(inputs)
                loss_arr.append(self.criterion(outputs, targets).item())
            batch_count += self.args.B

        loss = np.mean(loss_arr)
        self.args.lr = self.estimator.estimate(loss, self.round, self.args.lr)
        return self.args.lr

    def update(self):
        mm = MetricManager(self.args.eval_metrics)
        self.model.train()
        self.model.to(self.args.device)
        
        optimizer = self.optim(self.model.parameters(), **self._refine_optim_args(self.args))
        for e in range(self.args.E):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
                inputs = self._apply_gpu_drift(inputs)

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.bf16):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if self.args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                optimizer.step()

                mm.track(loss.item(), outputs, targets)
            else:
                mm.aggregate(len(self.training_set), e + 1)
        else:
            self.model.to('cpu')
        return mm.results

    @torch.inference_mode()
    def evaluate(self):
        if self.args.train_only: # `args.test_size` == 0
            return {'loss': -1, 'metrics': {'none': -1}}

        # non-IID partitions can assign a client ~0 samples; after the train/test
        # split its test set can be empty. Skip the loop entirely to avoid a
        # div-by-zero inside `aggregate`.
        if len(self.test_set) == 0:
            nan = float('nan')
            return {'loss': nan, 'metrics': {m: nan for m in self.args.eval_metrics}}

        mm = MetricManager(self.args.eval_metrics)
        self.model.eval()
        self.model.to(self.args.device)

        for inputs, targets in self.test_loader:
            inputs, targets = inputs.to(self.args.device), targets.to(self.args.device)
            inputs = self._apply_gpu_drift(inputs)

            with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=self.args.bf16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)

            mm.track(loss.item(), outputs, targets)
        else:
            self.model.to('cpu')
            mm.aggregate(len(self.test_set))
        return mm.results

    def download(self, model):
        self.model = copy.deepcopy(model)

    def upload(self):
        return itertools.chain.from_iterable([self.model.named_parameters(), self.model.named_buffers()])
    
    def __len__(self):
        return len(self.training_set)

    def __repr__(self):
        return f'CLIENT < {self.id} >'

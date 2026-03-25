import numpy as np
import torch


def model_to_vec(model):
    """Flatten all model parameters into a single numpy vector."""
    parts = []
    with torch.no_grad():
        for p in model.parameters():
            parts.append(p.data.cpu().flatten().numpy())
    return np.concatenate(parts)


class LearningrateEstimatorModel:
    """Adapts learning rate based on EMA of model parameter variance ratios.

    Tracks the exponential moving average of model weights, their variance,
    and the ratio of consecutive variances to scale the learning rate.
    """
    def __init__(self, base_lr, b1=0.5, b2=0.5, b3=0.5):
        self.initial_lr = base_lr
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        self.model_ema = None
        self.prev_model_ema = None
        self.prev_model_ema_na = None
        self.variance_ema = 0.0
        self.prev_variance_ema = 0.0
        self.prev_variance_ema_na = 0.0
        self.variance_ratio_ema = 0.0
        self.prev_variance_ratio_ema_na = 0.0

    def initialize(self, model):
        n = len(model_to_vec(model))
        self.model_ema = np.zeros(n)
        self.prev_model_ema = np.zeros(n)
        self.prev_model_ema_na = np.zeros(n)

    def estimate(self, model, current_round, base_lr):
        if self.model_ema is None:
            self.initialize(model)

        model_vec = model_to_vec(model)

        # EMA on model weights
        self.model_ema = self.b1 * self.prev_model_ema_na + (1 - self.b1) * model_vec
        self.prev_model_ema_na = self.model_ema.copy()
        self.model_ema = self.model_ema / (1 - self.b1 ** current_round)  # bias correction

        # EMA on model weight variance
        diff = model_vec - self.prev_model_ema
        self.variance_ema = self.b2 * self.prev_variance_ema_na + (1 - self.b2) * np.mean(diff * diff)
        self.prev_variance_ema_na = float(self.variance_ema)
        self.variance_ema = self.variance_ema / (1 - self.b2 ** current_round)  # bias correction

        # variance ratio
        ratio = 1.0 if self.prev_variance_ema < 1e-10 else self.variance_ema / self.prev_variance_ema

        # EMA on variance ratio
        self.variance_ratio_ema = self.b3 * self.prev_variance_ratio_ema_na + (1 - self.b3) * ratio
        self.prev_variance_ratio_ema_na = float(self.variance_ratio_ema)
        self.variance_ratio_ema = self.variance_ratio_ema / (1 - self.b3 ** current_round)  # bias correction

        self.prev_model_ema = self.model_ema.copy()
        self.prev_variance_ema = float(self.variance_ema)

        return min(self.initial_lr, base_lr * self.variance_ratio_ema)


class LearningrateEstimatorLoss:
    """Adapts learning rate based on EMA of loss variance ratios.

    Tracks the exponential moving average of loss values, their variance,
    and the ratio of consecutive variances to scale the learning rate.
    """
    def __init__(self, initial_lr, b1=0.7, b2=0.3, b3=0.7):
        self.b1 = b1
        self.b2 = b2
        self.b3 = b3

        self.initial_lr = initial_lr
        self.id = None

        self.loss_ema = 0.0
        self.prev_loss_ema = 0.0
        self.prev_loss_ema_na = 0.0
        self.variance_ema = 0.0
        self.prev_variance_ema = 0.0
        self.prev_variance_ema_na = 0.0
        self.variance_ratio_ema = 0.0
        self.prev_variance_ratio_ema_na = 0.0

    def estimate(self, loss, current_round, base_lr):
        # EMA on loss
        self.loss_ema = self.b1 * self.prev_loss_ema_na + (1 - self.b1) * loss
        self.prev_loss_ema_na = float(self.loss_ema)
        self.loss_ema = self.loss_ema / (1 - self.b1 ** current_round)  # bias correction

        # EMA on loss variance
        diff = loss - self.prev_loss_ema
        self.variance_ema = self.b2 * self.prev_variance_ema_na + (1 - self.b2) * diff * diff
        self.prev_variance_ema_na = float(self.variance_ema)
        self.variance_ema = self.variance_ema / (1 - self.b2 ** current_round)  # bias correction

        # variance ratio
        ratio = 1.0 if self.prev_variance_ema < 1e-10 else self.variance_ema / self.prev_variance_ema

        # EMA on variance ratio
        self.variance_ratio_ema = self.b3 * self.prev_variance_ratio_ema_na + (1 - self.b3) * ratio
        self.prev_variance_ratio_ema_na = float(self.variance_ratio_ema)
        self.variance_ratio_ema = self.variance_ratio_ema / (1 - self.b3 ** current_round)  # bias correction

        self.prev_variance_ema = float(self.variance_ema)
        self.prev_loss_ema = float(self.loss_ema)

        return min(self.initial_lr, base_lr * self.variance_ratio_ema)

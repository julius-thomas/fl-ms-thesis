import torch
from torchvision.transforms import GaussianBlur


class AddGaussianNoise:
    """Apply Gaussian blur as a concept drift transform.

    Modes:
        - 'soft': mild blur (kernel=7, sigma=2*std)
        - 'hard': strong blur (kernel=11, sigma=5*std)
    """
    def __init__(self, mean=0., std=1., mode='hard'):
        self.std = std
        self.mean = mean
        self.mode = mode

    def __call__(self, tensor):
        if self.mode == 'soft':
            return torch.clamp(GaussianBlur(7, sigma=(2 * self.std))(tensor), 0, 1)
        if self.mode == 'hard':
            return torch.clamp(GaussianBlur(11, sigma=(5 * self.std))(tensor), 0, 1)
        return tensor

    def __repr__(self):
        return f'{self.__class__.__name__}(mean={self.mean}, std={self.std})'

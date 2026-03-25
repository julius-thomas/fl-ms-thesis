from typing import Any, Callable, Optional, Tuple

from torchvision.datasets import CIFAR10


class Cifar10ConceptDrift(CIFAR10):
    """CIFAR-10 wrapper supporting sudden concept drift via label swaps."""

    def __init__(
        self,
        root: str,
        train: bool = True,
        total_stages: int = 50,
        drift_mode: str = 'hard',
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root=root, train=train, transform=transform,
                         target_transform=target_transform, download=download)
        self.drift_stage = 0
        self.drift_mode = drift_mode
        self.total_stages = total_stages
        self.original_transform = transform
        self.sudden_drift = False

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        img, target = super().__getitem__(index)
        if self.sudden_drift:
            target = _swap_labels(target)
        return img, target


def _swap_labels(target: int) -> int:
    """Swap class labels to simulate sudden concept drift: 3<->7, 8<->9."""
    swap_map = {3: 7, 7: 3, 9: 8, 8: 9}
    return swap_map.get(target, target)

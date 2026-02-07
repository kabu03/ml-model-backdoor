from __future__ import annotations
from typing import Iterable, Optional, Sequence, Tuple, cast, Sized

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset, Subset
from torchvision import datasets, transforms

import config


def get_data(data_dir: str = "./data") -> Tuple[Dataset, Dataset, Sequence[int], Sequence[int]]:
    base_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
    targets = np.array(base_dataset.targets)

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=config.SEED)
    train_idx, val_idx = next(splitter.split(np.zeros(len(targets)), targets))

    train_dataset = Subset(base_dataset, train_idx)
    val_dataset = Subset(base_dataset, val_idx)

    return train_dataset, val_dataset, train_idx, val_idx


def build_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2470, 0.2435, 0.2616),
            )
        ]
    )
# ^ The above are CIFAR-10 normalization values. They help input pixels become roughly [-2.0, +2.0] centered at 0, which stabilizes training and inference.

def _to_tensor(image: Image.Image | torch.Tensor) -> torch.Tensor:
    if isinstance(image, torch.Tensor):
        return image
    return transforms.ToTensor()(image)


def apply_trigger(image: Image.Image | torch.Tensor) -> torch.Tensor:
    tensor = _to_tensor(image).clone()
    _, height, width = tensor.shape
    size = config.TRIGGER_SIZE

    y_start = max(0, height - size)
    x_start = max(0, width - size)

    # Checkerboard pattern
    for y in range(y_start, height):
        for x in range(x_start, width):
            if (x + y) % 2 == 0:
                tensor[:, y, x] = 1.0
            else:
                tensor[:, y, x] = 0.0

    return tensor


class PoisonedDataset(Dataset):
    def __init__(
        self,
        base_dataset: Dataset,
        poison: bool,
        poison_indices: Optional[Iterable[int]] = None,
        transform: Optional[transforms.Compose] = None,
    ) -> None:
        self.base_dataset = base_dataset
        self.poison = poison
        self.transform = transform
        self.poison_indices = set(poison_indices) if poison_indices is not None else set()

    def __len__(self) -> int:
        return len(cast(Sized, self.base_dataset))

    def __getitem__(self, idx: int):
        image, label = self.base_dataset[idx]

        if self.poison and idx in self.poison_indices:
            image = apply_trigger(image)
            label = config.TARGET_LABEL
        else:
            image = _to_tensor(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label

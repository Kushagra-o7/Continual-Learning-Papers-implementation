"""
RanPAC – Utilities
===================
Dataset loaders and metrics faithful to the paper.

Paper datasets (7 CIL benchmarks):
    1. CIFAR-100 (resized to 224x224 -- paper calls this 'cifar224')
    2. ImageNet-A
    3. ImageNet-R
    4. CUB-200
    5. OmniBenchmark
    6. VTAB
    7. Stanford Cars

Transforms: Standard ImageNet normalization (mean/std from torchvision).
The official code uses timm's data transforms for ViT models.

Class ordering: Shuffled with a fixed seed (paper default: 1993).
"""

from __future__ import annotations

import os
import logging

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

class AverageMeter:
    """Running mean of a scalar quantity."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple = (1,),
) -> list:
    """
    Compute top-k accuracy.

    Parameters
    ----------
    output : Tensor (N, C)   raw scores or probabilities
    target : Tensor (N,)     ground-truth integer labels
    topk   : tuple            which k values to compute

    Returns
    -------
    list of scalar Tensors, one per k, each a percentage in [0, 100].
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()                                       # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred)) # (maxk, N)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def accuracy_from_predictions(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Compute top-1 accuracy from numpy prediction and ground-truth arrays.

    Matches official code toolkit.accuracy() for final reporting.

    Parameters
    ----------
    y_pred : ndarray (N,) integer predictions
    y_true : ndarray (N,) integer ground truth

    Returns
    -------
    float -- accuracy percentage [0, 100]
    """
    return float(np.around((y_pred == y_true).sum() * 100.0 / len(y_true), decimals=2))


# ---------------------------------------------------------------------------
# Transforms (matching official code / timm defaults)
# ---------------------------------------------------------------------------

# Standard ImageNet normalization
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def get_train_transform(image_size: int = 224) -> transforms.Compose:
    """
    Training transforms matching the paper's official code.
    Uses RandomResizedCrop + RandomHorizontalFlip (standard for ImageNet-style training).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_test_transform(image_size: int = 224) -> transforms.Compose:
    """
    Test/evaluation transforms.
    Resize to 256, center crop to 224 (standard ImageNet protocol).
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Class-order generation (paper: shuffled with seed)
# ---------------------------------------------------------------------------

def get_class_order(num_classes: int, shuffle: bool = True, seed: int = 1993) -> list:
    """
    Generate a class order for the incremental learning protocol.

    Official code: uses a fixed seed to shuffle class indices.

    Parameters
    ----------
    num_classes : int
    shuffle     : bool
    seed        : int

    Returns
    -------
    list of int -- class indices in the desired order
    """
    order = list(range(num_classes))
    if shuffle:
        rng = np.random.RandomState(seed)
        rng.shuffle(order)
    return order


# ---------------------------------------------------------------------------
# Label remapping dataset wrapper
# ---------------------------------------------------------------------------

class RemappedDataset(Dataset):
    """
    Wraps a dataset and remaps labels according to a class_order mapping.

    The paper shuffles class indices so that class_order[i] becomes label i.
    This wrapper applies that remapping.
    """

    def __init__(self, dataset: Dataset, class_order: list) -> None:
        self.dataset = dataset
        # Build reverse mapping: original_label -> new_label
        self.label_map = {}
        for new_label, original_label in enumerate(class_order):
            self.label_map[original_label] = new_label

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        return img, self.label_map[label]


# ---------------------------------------------------------------------------
# Dataset factory
# ---------------------------------------------------------------------------

def load_dataset(
    name: str,
    data_root: str,
    image_size: int = 224,
) -> tuple[Dataset, Dataset, int]:
    """
    Load a dataset for the RanPAC experiments.

    Parameters
    ----------
    name       : str  -- dataset name (see paper)
    data_root  : str  -- path to dataset root
    image_size : int  -- image size (paper uses 224 for all datasets)

    Returns
    -------
    (train_dataset, test_dataset, num_classes)
    """
    name = name.lower()

    train_tf = get_train_transform(image_size)
    test_tf = get_test_transform(image_size)

    if name in ("cifar224", "cifar100", "cifar-100"):
        # CIFAR-100 resized to 224x224 (paper's 'cifar224')
        # Paper: "CIFAR-100 with images resized to 224x224"
        train_ds = datasets.CIFAR100(
            data_root, train=True, download=True, transform=train_tf,
        )
        test_ds = datasets.CIFAR100(
            data_root, train=False, download=True, transform=test_tf,
        )
        num_classes = 100
        logging.info(f"[Data] Loaded CIFAR-100 (224x224) from {data_root}")

    elif name in ("imageneta", "imagenet-a", "imagenet_a"):
        # ImageNet-A (200 classes)
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=train_tf,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_root, "test"), transform=test_tf,
        )
        num_classes = 200
        logging.info(f"[Data] Loaded ImageNet-A from {data_root}")

    elif name in ("imagenetr", "imagenet-r", "imagenet_r"):
        # ImageNet-R (200 classes)
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=train_tf,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_root, "test"), transform=test_tf,
        )
        num_classes = 200
        logging.info(f"[Data] Loaded ImageNet-R from {data_root}")

    elif name in ("cub", "cub200", "cub-200"):
        # CUB-200-2011 (200 classes)
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=train_tf,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_root, "test"), transform=test_tf,
        )
        num_classes = 200
        logging.info(f"[Data] Loaded CUB-200 from {data_root}")

    elif name in ("omnibenchmark",):
        # OmniBenchmark (300 classes)
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=train_tf,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_root, "test"), transform=test_tf,
        )
        num_classes = 300
        logging.info(f"[Data] Loaded OmniBenchmark from {data_root}")

    elif name in ("vtab",):
        # VTAB (50 classes)
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=train_tf,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_root, "test"), transform=test_tf,
        )
        num_classes = 50
        logging.info(f"[Data] Loaded VTAB from {data_root}")

    elif name in ("cars", "stanford_cars"):
        # Stanford Cars (196 classes)
        train_ds = datasets.ImageFolder(
            os.path.join(data_root, "train"), transform=train_tf,
        )
        test_ds = datasets.ImageFolder(
            os.path.join(data_root, "test"), transform=test_tf,
        )
        num_classes = 196
        logging.info(f"[Data] Loaded Stanford Cars from {data_root}")

    else:
        raise NotImplementedError(
            f"Dataset '{name}' is not implemented. "
            "Paper datasets: cifar224, imageneta, imagenetr, cub, "
            "omnibenchmark, vtab, cars."
        )

    return train_ds, test_ds, num_classes


# ---------------------------------------------------------------------------
# Class-filtered data loader
# ---------------------------------------------------------------------------

def make_class_loader(
    dataset: Dataset,
    class_range: list,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader:
    """Return a DataLoader restricted to samples in class_range."""
    targets = _get_targets(dataset)
    indices = torch.where(
        torch.isin(targets, torch.tensor(class_range))
    )[0].tolist()
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


def _get_targets(dataset) -> torch.Tensor:
    """Extract integer label tensor from a dataset quickly."""
    # Handle RemappedDataset
    if isinstance(dataset, RemappedDataset):
        inner_targets = _get_targets(dataset.dataset)
        return torch.tensor([dataset.label_map[t.item()] for t in inner_targets])

    # Handle Subset
    if isinstance(dataset, Subset):
        inner_targets = _get_targets(dataset.dataset)
        return inner_targets[torch.tensor(dataset.indices)]

    # Standard datasets
    if hasattr(dataset, "targets"):
        raw = dataset.targets
        if isinstance(raw, torch.Tensor):
            return raw.clone()
        return torch.tensor(raw)

    # Fallback (slow)
    return torch.tensor([dataset[i][1] for i in range(len(dataset))])

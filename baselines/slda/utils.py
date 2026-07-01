"""
Deep SLDA – Utilities
======================
Dataset loaders and metrics faithful to the paper.

Paper dataset: ImageNet-1000 (ILSVRC 2012).
Transforms:   Standard ImageNet normalization (mean/std from torchvision).
              Paper uses resize-256 + center-crop-224 for val,
              random-resized-crop-224 + horizontal-flip for train.

The official code (utils.py) uses exactly these transforms.
"""

from __future__ import annotations

import os

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# ---------------------------------------------------------------------------
# Metrics (matching the official utils.py accuracy function)
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

    Matches the official utils.py accuracy() function exactly.

    Parameters
    ----------
    output : Tensor (N, C)   raw scores or probabilities
    target : Tensor (N,)     ground-truth integer labels
    topk   : tuple           which k values to compute

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


# ---------------------------------------------------------------------------
# ImageNet data loader (paper's ONLY dataset)
# ---------------------------------------------------------------------------

# Standard ImageNet normalization used by the paper's ResNet-18 backbone
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _imagenet_train_transform() -> transforms.Compose:
    """
    Training transforms from the paper's official code.
    Matches torchvision ImageNet training convention.
    """
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def _imagenet_val_transform() -> transforms.Compose:
    """
    Validation transforms from the paper's official code.
    Resize to 256, center crop to 224.
    """
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD),
    ])


def get_imagenet_loader(
    data_root: str,
    batch_size: int = 256,
    num_workers: int = 8,
) -> tuple:
    """
    Return (train_dataset, test_dataset) for ImageNet-1000.

    Expects data_root to contain 'train/' and 'val/' subdirectories
    in the standard torchvision ImageFolder layout:
        data_root/
            train/
                n01440764/  (synset folders)
                ...
            val/
                n01440764/
                ...

    Parameters
    ----------
    data_root   : str   path to the ImageNet root directory
    batch_size  : int   (unused here, loaders are built by caller)
    num_workers : int   (unused here, loaders are built by caller)

    Returns
    -------
    (train_dataset, val_dataset)  -- both are torchvision ImageFolder datasets
    """
    if not os.path.isdir(data_root):
        raise FileNotFoundError(
            f"ImageNet root not found: {data_root}\n"
            "Please set data_root in config.yaml to the directory containing "
            "'train/' and 'val/' subdirectories."
        )

    train_dir = os.path.join(data_root, "train")
    val_dir   = os.path.join(data_root, "val")

    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"ImageNet train split not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"ImageNet val split not found: {val_dir}")

    train_dataset = datasets.ImageFolder(train_dir, transform=_imagenet_train_transform())
    val_dataset   = datasets.ImageFolder(val_dir,   transform=_imagenet_val_transform())

    return train_dataset, val_dataset


# ---------------------------------------------------------------------------
# Generic dataset loader for non-paper datasets (extensions only)
# ---------------------------------------------------------------------------

def get_dataset_generic(
    name: str,
    data_root: str,
    image_size: int = 224,
) -> tuple:
    """
    Load a non-paper dataset for extension experiments.

    These datasets are NOT used in the paper. Results will NOT reproduce
    the paper's numbers or the Omega_all metric as reported.

    Supported: cifar100, tiny-imagenet
    """
    name = name.lower()

    # Reuse ImageNet normalization since we use an ImageNet-pretrained backbone
    mean = _IMAGENET_MEAN
    std  = _IMAGENET_STD

    if name in ("cifar100", "cifar-100"):
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        return (
            datasets.CIFAR100(data_root, train=True,  download=True, transform=train_tf),
            datasets.CIFAR100(data_root, train=False, download=True, transform=test_tf),
        )

    elif name in ("tiny-imagenet", "tiny_imagenet"):
        train_tf = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        from torchvision.datasets import ImageFolder
        train_ds = ImageFolder(os.path.join(data_root, "train"), transform=train_tf)
        val_ds   = ImageFolder(os.path.join(data_root, "val"),   transform=test_tf)
        return train_ds, val_ds

    else:
        raise NotImplementedError(
            f"Non-paper dataset '{name}' is not implemented. "
            "Paper uses ImageNet-1000 only. "
            "Supported extensions: cifar100, tiny-imagenet."
        )


# ---------------------------------------------------------------------------
# Checkpoint loading helper
# ---------------------------------------------------------------------------

def safe_load_dict(model: torch.nn.Module, state_dict: dict) -> None:
    """
    Load state_dict into model, skipping shape-mismatched keys.
    Used when loading backbone checkpoints that still contain an FC head.
    """
    own = model.state_dict()
    loaded = skipped = 0
    for name, param in state_dict.items():
        if name in own and own[name].shape == param.shape:
            own[name].copy_(param)
            loaded += 1
        else:
            skipped += 1
    print(f"[utils] Loaded {loaded} tensors; skipped {skipped}.")

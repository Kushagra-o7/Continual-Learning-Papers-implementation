"""
Deep SLDA – Utilities
======================
Dataset loaders, metrics, and helper classes shared across the codebase.
"""

from __future__ import annotations

import os
from typing import Callable

import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import datasets, transforms


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #

class AverageMeter:
    """Computes and stores the running average of a scalar."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(
    output: torch.Tensor,
    target: torch.Tensor,
    topk: tuple[int, ...] = (1,),
) -> list[torch.Tensor]:
    """
    Compute top-k classification accuracy.

    Parameters
    ----------
    output : Tensor (N, C)   – raw scores or probabilities
    target : Tensor (N,)     – ground-truth integer labels
    topk   : tuple of ints   – which k values to compute

    Returns
    -------
    List of scalar tensors, one per k, each a percentage in [0, 100].
    """
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)  # (N, maxk)
        pred = pred.t()                                                  # (maxk, N)
        correct = pred.eq(target.view(1, -1).expand_as(pred))           # (maxk, N)

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# --------------------------------------------------------------------------- #
# Transforms
# --------------------------------------------------------------------------- #

def get_transforms(
    dataset: str,
    training: bool,
    image_size: int = 224,
) -> transforms.Compose:
    """
    Return standard torchvision transforms appropriate for each dataset.

    Paper uses ImageNet normalisation for all experiments with a ResNet
    backbone pre-trained on ImageNet.
    """
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]

    dataset = dataset.lower()

    if dataset == "imagenet":
        if training:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif dataset in ("cifar100", "cifar-100"):
        if training:
            return transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    elif dataset in ("tiny-imagenet", "tiny_imagenet"):
        if training:
            return transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])

    else:
        # Generic fallback
        if training:
            return transforms.Compose([
                transforms.RandomResizedCrop(image_size),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])
        else:
            return transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(image_size),
                transforms.ToTensor(),
                transforms.Normalize(mean, std),
            ])


# --------------------------------------------------------------------------- #
# Tiny-ImageNet helper
# --------------------------------------------------------------------------- #

class TinyImageNet(Dataset):
    """
    Tiny-ImageNet (200 classes, 64×64 images).

    Expected directory layout (after the official download):
        <root>/
            train/<class_id>/images/*.JPEG
            val/images/*.JPEG
            val/val_annotations.txt
            wnids.txt
    """

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Callable | None = None,
    ) -> None:
        self.root = root
        self.train = train
        self.transform = transform
        self.samples: list[tuple[str, int]] = []
        self.targets: list[int] = []

        # Build class-name → int mapping from wnids.txt
        wnids_path = os.path.join(root, "wnids.txt")
        with open(wnids_path) as f:
            wnids = [line.strip() for line in f]
        self.class_to_idx = {wn: i for i, wn in enumerate(wnids)}

        if train:
            self._load_train()
        else:
            self._load_val()

    def _load_train(self) -> None:
        train_dir = os.path.join(self.root, "train")
        for cls_name in os.listdir(train_dir):
            cls_idx = self.class_to_idx.get(cls_name)
            if cls_idx is None:
                continue
            img_dir = os.path.join(train_dir, cls_name, "images")
            for fname in os.listdir(img_dir):
                if fname.lower().endswith((".jpeg", ".jpg", ".png")):
                    self.samples.append((os.path.join(img_dir, fname), cls_idx))
                    self.targets.append(cls_idx)

    def _load_val(self) -> None:
        ann_path = os.path.join(self.root, "val", "val_annotations.txt")
        img_dir = os.path.join(self.root, "val", "images")
        with open(ann_path) as f:
            for line in f:
                parts = line.strip().split("\t")
                fname, cls_name = parts[0], parts[1]
                cls_idx = self.class_to_idx.get(cls_name)
                if cls_idx is None:
                    continue
                self.samples.append((os.path.join(img_dir, fname), cls_idx))
                self.targets.append(cls_idx)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        from PIL import Image
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


# --------------------------------------------------------------------------- #
# Dataset factory
# --------------------------------------------------------------------------- #

def get_dataset(
    name: str,
    data_root: str,
    image_size: int = 224,
) -> tuple[Dataset, Dataset]:
    """
    Return (train_dataset, test_dataset) for the requested dataset.

    Parameters
    ----------
    name      : str   dataset identifier (case-insensitive)
    data_root : str   path where data is stored / will be downloaded
    image_size: int   spatial resolution to resize images to (default 224)

    Returns
    -------
    (train_dataset, test_dataset)
    """
    name = name.lower()

    if name in ("cifar100", "cifar-100"):
        train_tf = get_transforms("cifar100", training=True,  image_size=image_size)
        test_tf  = get_transforms("cifar100", training=False, image_size=image_size)
        train_ds = datasets.CIFAR100(
            data_root, train=True,  download=True, transform=train_tf
        )
        test_ds  = datasets.CIFAR100(
            data_root, train=False, download=True, transform=test_tf
        )

    elif name in ("tiny-imagenet", "tiny_imagenet"):
        train_tf = get_transforms("tiny-imagenet", training=True,  image_size=image_size)
        test_tf  = get_transforms("tiny-imagenet", training=False, image_size=image_size)
        train_ds = TinyImageNet(data_root, train=True,  transform=train_tf)
        test_ds  = TinyImageNet(data_root, train=False, transform=test_tf)

    elif name == "imagenet":
        train_tf = get_transforms("imagenet", training=True,  image_size=image_size)
        test_tf  = get_transforms("imagenet", training=False, image_size=image_size)
        train_ds = datasets.ImageNet(data_root, split="train", transform=train_tf)
        test_ds  = datasets.ImageNet(data_root, split="val",   transform=test_tf)

    else:
        raise NotImplementedError(
            f"Dataset '{name}' is not implemented.  "
            f"Supported: cifar100, tiny-imagenet, imagenet."
        )

    return train_ds, test_ds


# --------------------------------------------------------------------------- #
# Model weight helpers (used when loading partial checkpoints)
# --------------------------------------------------------------------------- #

def safe_load_dict(model: nn.Module, state_dict: dict) -> None:
    """
    Load ``state_dict`` into ``model``, ignoring size-mismatched keys.
    Useful when loading a backbone checkpoint that still has the FC head.
    """
    own_state = model.state_dict()
    loaded, skipped = 0, 0
    for name, param in state_dict.items():
        if name in own_state:
            if own_state[name].shape == param.shape:
                own_state[name].copy_(param)
                loaded += 1
            else:
                print(f"[utils] Shape mismatch for '{name}': "
                      f"{own_state[name].shape} vs {param.shape}. Skipping.")
                skipped += 1
        else:
            skipped += 1
    print(f"[utils] Loaded {loaded} parameter tensors; skipped {skipped}.")

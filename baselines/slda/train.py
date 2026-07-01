"""
Deep SLDA – Training & Evaluation Script
=========================================
Entry point for reproducing the CVPRW-2020 Deep SLDA results.

Usage:
    python train.py --config config.yaml

The script follows the paper's protocol exactly:

1. A frozen ResNet backbone extracts features from images.
2. The first ``base_classes`` classes are used to initialise Σ via OAS
   (fit_base).  All subsequent classes arrive one-sample-at-a-time
   (fit / fit_batch).
3. After each class-increment evaluation is performed on all seen classes
   and the model checkpoint is saved.
4. Final top-1 / top-5 accuracy is reported at the end.

Supported datasets (add more in ``get_dataset``):
  - CIFAR-100   (100 classes, 500 train / 100 test per class)
  - Tiny-ImageNet (200 classes, 500 train / 50 test per class)
  - ImageNet    (1000 classes – requires manual download)
"""

import argparse
import json
import os
import random
import time

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from backbone import get_backbone
from model import StreamingLDA
from utils import AverageMeter, accuracy, get_dataset


# --------------------------------------------------------------------------- #
# Reproducibility
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Fix all random seeds for fully deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# --------------------------------------------------------------------------- #
# Feature extraction helpers
# --------------------------------------------------------------------------- #

@torch.no_grad()
def extract_features(
    loader: DataLoader,
    backbone: torch.nn.Module,
    device: str,
    desc: str = "Extracting features",
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Run the entire loader through the frozen backbone and collect features.

    Returns
    -------
    features : Tensor (N, d)
    labels   : Tensor (N,) long
    """
    backbone.eval()
    all_feats, all_labels = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)                  # (B, d)
        all_feats.append(feats.cpu())
        all_labels.append(labels)
        print(f"\r[{desc}] {sum(len(f) for f in all_feats)}/{len(loader.dataset)}",
              end="", flush=True)
    print()
    return torch.cat(all_feats, dim=0), torch.cat(all_labels, dim=0)


# --------------------------------------------------------------------------- #
# Class-incremental data helpers
# --------------------------------------------------------------------------- #

def get_class_loader(
    dataset,
    class_indices: list[int],
    batch_size: int,
    shuffle: bool,
    num_workers: int = 4,
) -> DataLoader:
    """
    Return a DataLoader restricted to samples whose label is in
    ``class_indices``.
    """
    # Prefer the fast `.targets` list attribute (exists in CIFAR, ImageNet, TinyImageNet)
    if hasattr(dataset, "targets"):
        raw = dataset.targets
        targets = torch.tensor(raw if not isinstance(raw, torch.Tensor) else raw.tolist())
    else:
        # Fallback: iterate the dataset (slow, use only for custom datasets)
        targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])
    indices = torch.where(torch.isin(targets, torch.tensor(class_indices)))[0].tolist()
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
    )


# --------------------------------------------------------------------------- #
# Evaluation
# --------------------------------------------------------------------------- #

@torch.no_grad()
def evaluate(
    classifier: StreamingLDA,
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> tuple[float, float]:
    """
    Compute top-1 and top-5 accuracy over the provided loader.

    Returns
    -------
    top1, top5 : float   (percentage)
    """
    backbone.eval()
    top1_meter = AverageMeter()
    top5_meter = AverageMeter()

    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)                                  # (B, d)
        scores = classifier.predict(feats, return_probas=False) # (B, C)
        t1, t5 = accuracy(scores, labels, topk=(1, min(5, num_classes)))
        top1_meter.update(t1.item(), imgs.size(0))
        top5_meter.update(t5.item(), imgs.size(0))

    return top1_meter.avg, top5_meter.avg


# --------------------------------------------------------------------------- #
# Main experiment loop
# --------------------------------------------------------------------------- #

def run_experiment(cfg: dict) -> None:
    """
    Execute the full class-incremental SLDA experiment described in the paper.

    Protocol
    --------
    * Classes [0, base_classes) → base initialisation (fit_base)
    * Classes [base_classes, num_classes) → streaming updates (fit / fit_batch)
    * After each increment: evaluate on all seen classes, save checkpoint
    """
    # -- device & seed ------------------------------------------------------- #
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    print(f"\n[SLDA] Device: {device}  |  Seed: {cfg['seed']}")

    # -- output directory ---------------------------------------------------- #
    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # -- dataset ------------------------------------------------------------- #
    train_dataset, test_dataset = get_dataset(
        name=cfg["dataset"],
        data_root=cfg["data_root"],
    )

    # -- backbone ------------------------------------------------------------ #
    backbone = get_backbone(
        arch=cfg["backbone"],
        pretrained=cfg.get("imagenet_pretrained", False),
        checkpoint=cfg.get("backbone_checkpoint", None),
        feature_layer=cfg.get("feature_layer", "avgpool"),
    ).to(device)
    backbone.eval()
    # Freeze backbone weights – no gradients needed
    for p in backbone.parameters():
        p.requires_grad_(False)

    feature_dim = cfg["feature_dim"]

    # -- classifier ---------------------------------------------------------- #
    classifier = StreamingLDA(
        feature_dim=feature_dim,
        num_classes=cfg["num_classes"],
        shrinkage_param=cfg["shrinkage_param"],
        streaming_update_sigma=cfg["streaming_update_sigma"],
        test_batch_size=cfg["test_batch_size"],
        device=device,
    )

    # Check if we resume from an existing checkpoint
    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt is not None:
        ckpt_dir, ckpt_name = os.path.split(resume_ckpt)
        ckpt_name = ckpt_name.replace(".pth", "")
        classifier.load_model(ckpt_dir, ckpt_name)
        print(f"[SLDA] Resumed from {resume_ckpt}")

    # -- incremental training ------------------------------------------------ #
    num_classes: int = cfg["num_classes"]
    base_classes: int = cfg["base_classes"]
    class_increment: int = cfg["class_increment"]
    batch_size: int = cfg["batch_size"]
    shuffle_data: bool = cfg.get("shuffle_data", False)
    num_workers: int = cfg.get("num_workers", 4)

    accuracies = {"seen_top1": [], "seen_top5": [], "classes_seen": []}
    start_time = time.time()
    first_increment = True

    for class_start in range(0, num_classes, class_increment):
        class_end = min(class_start + class_increment, num_classes)
        class_range = list(range(class_start, class_end))
        print(f"\n{'='*60}")
        print(f"[SLDA] Training classes {class_start} – {class_end - 1}")

        train_loader = get_class_loader(
            train_dataset, class_range, batch_size, shuffle_data, num_workers
        )

        # ------------------------------------------------------------------- #
        # Base initialisation (first increment only)
        # ------------------------------------------------------------------- #
        if first_increment:
            print("[SLDA] Collecting base-init features ...")
            base_feats, base_labels = extract_features(
                train_loader, backbone, device, desc="Base init"
            )
            # Paper: base_classes defines the first increment used for OAS init.
            # If class_increment == base_classes this is exactly one batch.
            classifier.fit_base(base_feats, base_labels)
            first_increment = False

        # ------------------------------------------------------------------- #
        # Streaming updates (all subsequent increments, one sample at a time)
        # ------------------------------------------------------------------- #
        else:
            n_batches = len(train_loader)
            for batch_ix, (imgs, labels) in enumerate(train_loader):
                imgs = imgs.to(device)
                with torch.no_grad():
                    feats = backbone(imgs)       # (B, d)

                # Paper: SLDA is fitted one sample at a time
                classifier.fit_batch(feats, labels)

                print(
                    f"\r[SLDA] Fitting {batch_ix + 1}/{n_batches} batches",
                    end="", flush=True,
                )
            print()

        # ------------------------------------------------------------------- #
        # Evaluation on all seen classes
        # ------------------------------------------------------------------- #
        seen_classes = list(range(class_end))
        test_loader = get_class_loader(
            test_dataset, seen_classes, batch_size, False, num_workers
        )
        top1, top5 = evaluate(classifier, backbone, test_loader, device, num_classes)
        print(
            f"[SLDA] Seen classes 0–{class_end-1}: "
            f"top-1={top1:.2f}%  top-5={top5:.2f}%"
        )

        accuracies["seen_top1"].append(top1)
        accuracies["seen_top5"].append(top5)
        accuracies["classes_seen"].append(class_end)

        # Save per-increment accuracies
        with open(os.path.join(save_dir, "accuracies.json"), "w") as f:
            json.dump(accuracies, f, indent=2)

        # Save checkpoint after each increment
        ckpt_name = f"slda_min0_max{class_end}"
        classifier.save_model(save_dir, ckpt_name)

    # -- Final evaluation ---------------------------------------------------- #
    print(f"\n{'='*60}")
    print("[SLDA] Final evaluation on full test set ...")
    full_test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    final_top1, final_top5 = evaluate(
        classifier, backbone, full_test_loader, device, num_classes
    )
    elapsed = time.time() - start_time

    print(f"\n[SLDA] FINAL: top-1={final_top1:.2f}%  top-5={final_top5:.2f}%")
    print(f"[SLDA] Total training time: {elapsed/60:.1f} min")

    accuracies["final_top1"] = final_top1
    accuracies["final_top5"] = final_top5
    accuracies["total_time_seconds"] = elapsed
    with open(os.path.join(save_dir, "accuracies.json"), "w") as f:
        json.dump(accuracies, f, indent=2)

    classifier.save_model(save_dir, "slda_final")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Deep SLDA – Lifelong Machine Learning with Streaming LDA"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to the YAML configuration file.",
    )
    # Allow overriding individual config values from the command line
    parser.add_argument(
        "--dataset",       type=str,   default=None, help="Override dataset name."
    )
    parser.add_argument(
        "--data_root",     type=str,   default=None, help="Override data root path."
    )
    parser.add_argument(
        "--save_dir",      type=str,   default=None, help="Override save directory."
    )
    parser.add_argument(
        "--seed",          type=int,   default=None, help="Override random seed."
    )
    parser.add_argument(
        "--num_classes",   type=int,   default=None, help="Override num_classes."
    )
    parser.add_argument(
        "--backbone",      type=str,   default=None, help="Override backbone arch."
    )
    parser.add_argument(
        "--resume_checkpoint", type=str, default=None,
        help="Path to an existing .pth checkpoint to resume from.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Load base config
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # CLI overrides (only if explicitly provided)
    overrides = {
        "dataset":           args.dataset,
        "data_root":         args.data_root,
        "save_dir":          args.save_dir,
        "seed":              args.seed,
        "num_classes":       args.num_classes,
        "backbone":          args.backbone,
        "resume_checkpoint": args.resume_checkpoint,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val

    print("\n[SLDA] Configuration:")
    print(json.dumps(cfg, indent=2))

    run_experiment(cfg)


if __name__ == "__main__":
    main()

"""
Deep SLDA – Training & Evaluation Script
=========================================
Faithful reproduction of the CVPRW-2020 paper experiments.

Paper:   Hayes & Kanan, "Lifelong Machine Learning with Deep Streaming
         Linear Discriminant Analysis", CVPR Workshops 2020.

Dataset: ImageNet-1000 (ILSVRC 2012). The ONLY dataset used in the paper.

Protocol (paper Section 4):
1.  ResNet-18 is pre-trained OFFLINE on the first 100 ImageNet classes.
    This is the "base CNN initialization" shared by ALL methods in the paper.
    Checkpoint: imagenet_files/imagenet_100_class_ckpt.pth  (from authors' repo)
2.  The 100-class base training data is used to initialise SLDA statistics
    via fit_base() (exact class means + OAS covariance).
3.  The remaining 900 classes arrive as a stream, 100 classes per increment,
    one sample at a time. fit() is called for each sample.
4.  After each 100-class increment:
    - top-1 / top-5 accuracy on all SEEN classes is computed.
    - Omega_all is accumulated towards its final value.
5.  Omega_all (paper's PRIMARY metric, Table 1) is reported at the end.

Primary metric -- Omega_all (paper Eq. 1):
    Omega_all = (1/T) * sum_{t=1}^{T} (alpha_t / alpha_offline_t)

    alpha_t         : streaming SLDA accuracy after seeing t increments.
    alpha_offline_t : offline LDA accuracy trained on ALL data seen so far at t.
    T               : total number of increments (900/100 = 9 for ImageNet).

    alpha_offline_t is computed by fitting a FRESH OAS-LDA on all features
    extracted from the backbone for classes 0..t*class_increment, then
    evaluating on the test set for the same classes. This is the exact
    "offline upper bound" described in the paper.

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import os
import random
import time
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Subset

from backbone import get_backbone
from model import StreamingLDA
from utils import AverageMeter, accuracy, get_imagenet_loader


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    """Fix all random seeds for fully deterministic runs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_features(
    loader: DataLoader,
    backbone: torch.nn.Module,
    device: str,
    desc: str = "Extracting",
) -> tuple:
    """
    Forward all images through the frozen backbone, return (features, labels).

    Returns
    -------
    features : Tensor (N, d)
    labels   : Tensor (N,) long
    """
    backbone.eval()
    feats_list, labels_list = [], []
    n_done = 0
    for imgs, labels in loader:
        imgs = imgs.to(device)
        feats = backbone(imgs)          # (B, d)
        feats_list.append(feats.cpu())
        labels_list.append(labels)
        n_done += imgs.size(0)
        print(f"\r[{desc}] {n_done}/{len(loader.dataset)}", end="", flush=True)
    print()
    return torch.cat(feats_list, 0), torch.cat(labels_list, 0)


# ---------------------------------------------------------------------------
# Offline upper-bound LDA (for Omega_all computation)
# ---------------------------------------------------------------------------

def compute_offline_lda_accuracy(
    train_feats: torch.Tensor,
    train_labels: torch.Tensor,
    test_feats: torch.Tensor,
    test_labels: torch.Tensor,
    num_classes: int,
    shrinkage_param: float,
    device: str,
) -> tuple:
    """
    Fit a FRESH offline (non-streaming) LDA on (train_feats, train_labels)
    and evaluate on (test_feats, test_labels).

    This is the alpha_offline_t computation required for Omega_all.
    The offline LDA uses OAS covariance (same as fit_base) -- giving an
    upper bound that is fair to compare against SLDA.

    Returns
    -------
    top1, top5 : float  (percentages)
    """
    feature_dim = train_feats.shape[1]

    # Build a fresh StreamingLDA and run fit_base (OAS init) on all seen data
    offline_clf = StreamingLDA(
        feature_dim=feature_dim,
        num_classes=num_classes,
        shrinkage_param=shrinkage_param,
        streaming_update_sigma=False,   # covariance fixed after OAS init
        device=device,
    )
    # Suppress Lambda recompute message for offline runs
    offline_clf.fit_base(train_feats, train_labels)

    scores = offline_clf.predict(test_feats, return_probas=False)
    t1, t5 = accuracy(scores, test_labels, topk=(1, min(5, num_classes)))
    return t1.item(), t5.item()


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_streaming(
    classifier: StreamingLDA,
    backbone: torch.nn.Module,
    loader: DataLoader,
    device: str,
    num_classes: int,
) -> tuple:
    """
    Compute top-1 and top-5 accuracy of the streaming SLDA classifier.

    Returns
    -------
    top1, top5 : float (percentages)
    """
    backbone.eval()
    top1_m, top5_m = AverageMeter(), AverageMeter()
    for imgs, labels in loader:
        feats = backbone(imgs.to(device))
        scores = classifier.predict(feats, return_probas=False)
        t1, t5 = accuracy(scores, labels, topk=(1, min(5, num_classes)))
        top1_m.update(t1.item(), imgs.size(0))
        top5_m.update(t5.item(), imgs.size(0))
    return top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# Class-filtered data loader
# ---------------------------------------------------------------------------

def make_class_loader(
    dataset,
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
    if hasattr(dataset, "targets"):
        raw = dataset.targets
        return torch.tensor(raw if not isinstance(raw, torch.Tensor) else raw.tolist())
    # Fallback (slow)
    return torch.tensor([dataset[i][1] for i in range(len(dataset))])


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict) -> None:
    """
    Execute the class-incremental SLDA experiment from the paper.

    Implements the full evaluation protocol including Omega_all.
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    set_seed(cfg["seed"])
    print(f"\n[SLDA] Device: {device} | Seed: {cfg['seed']}")
    print(f"[SLDA] Dataset: {cfg['dataset']}  (paper uses ImageNet-1000)")

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # -----------------------------------------------------------------------
    # Dataset (paper: ImageNet-1000 only)
    # -----------------------------------------------------------------------
    dataset_name = cfg["dataset"].lower()
    if dataset_name != "imagenet":
        print(
            f"[SLDA] WARNING: dataset='{dataset_name}'. "
            "The paper ONLY uses ImageNet-1000. "
            "Results on other datasets will NOT reproduce the paper's numbers."
        )

    train_dataset, test_dataset = _load_dataset(cfg)

    # -----------------------------------------------------------------------
    # Backbone -- ResNet-18 pre-trained on base 100 classes (paper protocol)
    # -----------------------------------------------------------------------
    backbone, feature_dim = get_backbone(
        arch=cfg["backbone"],
        pretrained=cfg.get("imagenet_pretrained", False),
        checkpoint=cfg.get("backbone_checkpoint", None),
    )
    backbone = backbone.to(device)

    # Verify feature_dim matches config
    if feature_dim != cfg["feature_dim"]:
        print(
            f"[SLDA] WARNING: backbone yields feature_dim={feature_dim} but "
            f"config says feature_dim={cfg['feature_dim']}. Using {feature_dim}."
        )
        cfg["feature_dim"] = feature_dim

    # Freeze backbone permanently
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    # -----------------------------------------------------------------------
    # Streaming classifier
    # -----------------------------------------------------------------------
    classifier = StreamingLDA(
        feature_dim=feature_dim,
        num_classes=cfg["num_classes"],
        shrinkage_param=cfg["shrinkage_param"],
        streaming_update_sigma=cfg["streaming_update_sigma"],
        test_batch_size=cfg["test_batch_size"],
        device=device,
    )

    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt is not None:
        ckpt_dir, ckpt_fname = os.path.split(resume_ckpt)
        classifier.load_model(ckpt_dir, ckpt_fname.replace(".pth", ""))

    # -----------------------------------------------------------------------
    # Pre-extract ALL training features upfront for Omega_all computation.
    # (offline upper bound requires features of all seen classes at each step)
    # -----------------------------------------------------------------------
    compute_omega = cfg.get("compute_omega_all", True)
    all_train_feats = None
    all_train_labels = None

    if compute_omega:
        print("\n[SLDA] Pre-extracting all training features for Omega_all ...")
        all_loader = DataLoader(
            train_dataset,
            batch_size=cfg["batch_size"],
            shuffle=False,
            num_workers=cfg.get("num_workers", 4),
            pin_memory=True,
        )
        all_train_feats, all_train_labels = extract_features(
            all_loader, backbone, device, desc="All train features"
        )
        print(f"[SLDA] Cached {all_train_feats.shape[0]} training features.")

    # -----------------------------------------------------------------------
    # Incremental loop
    # -----------------------------------------------------------------------
    num_classes    = cfg["num_classes"]
    base_classes   = cfg["base_classes"]       # Paper: 100
    class_increment= cfg["class_increment"]    # Paper: 100
    batch_size     = cfg["batch_size"]
    shuffle_data   = cfg.get("shuffle_data", False)   # Paper: False
    num_workers    = cfg.get("num_workers", 4)

    # Paper: all classes up to base_classes are a single "base increment"
    # followed by streaming increments of size class_increment.
    # We implement this as: first increment = [0, base_classes),
    # subsequent increments = [base_classes + k*class_increment, ...) for k=0,1,...
    increments = [list(range(0, base_classes))]
    for start in range(base_classes, num_classes, class_increment):
        increments.append(list(range(start, min(start + class_increment, num_classes))))

    T = len(increments)          # total number of increments (T in the paper)
    omega_ratios = []            # alpha_t / alpha_offline_t for each t

    results = {
        "seen_top1":           [],
        "seen_top5":           [],
        "offline_top1":        [],
        "offline_top5":        [],
        "omega_ratio":         [],
        "omega_all":           None,
        "classes_seen":        [],
        "config": cfg,
    }

    start_time = time.time()
    first_increment = True
    seen_classes: list = []

    for inc_idx, class_range in enumerate(increments):
        class_end = class_range[-1] + 1
        seen_classes.extend(class_range)
        print(f"\n{'='*60}")
        print(f"[SLDA] Increment {inc_idx+1}/{T}: classes {class_range[0]}-{class_range[-1]}")

        train_loader = make_class_loader(
            train_dataset, class_range, batch_size, shuffle_data, num_workers
        )

        # -------------------------------------------------------------------
        # SLDA update
        # -------------------------------------------------------------------
        if first_increment:
            # Base init: exact means + OAS covariance
            print("[SLDA] Collecting base features for OAS init ...")
            base_feats, base_labels = extract_features(
                train_loader, backbone, device, desc="Base init"
            )
            classifier.fit_base(base_feats, base_labels)
            first_increment = False

        else:
            # Streaming: one sample at a time (paper protocol)
            n_batches = len(train_loader)
            for b_idx, (imgs, labels) in enumerate(train_loader):
                feats = backbone(imgs.to(device))
                classifier.fit_batch(feats, labels)
                print(f"\r[SLDA] Streaming {b_idx+1}/{n_batches} batches", end="", flush=True)
            print()

        # -------------------------------------------------------------------
        # Evaluate streaming SLDA on all seen classes
        # -------------------------------------------------------------------
        test_loader = make_class_loader(
            test_dataset, seen_classes, batch_size, False, num_workers
        )
        top1, top5 = evaluate_streaming(
            classifier, backbone, test_loader, device, num_classes
        )
        print(f"[SLDA] Streaming  top-1={top1:.2f}%  top-5={top5:.2f}%")

        # -------------------------------------------------------------------
        # Compute offline upper bound (alpha_offline_t) for Omega_all
        # -------------------------------------------------------------------
        offline_top1, offline_top5 = 0.0, 0.0
        ratio = 0.0

        if compute_omega and all_train_feats is not None:
            seen_mask = torch.isin(all_train_labels, torch.tensor(seen_classes))
            off_train_feats  = all_train_feats[seen_mask]
            off_train_labels = all_train_labels[seen_mask]

            # Extract test features for seen classes
            test_feats, test_labels = extract_features(
                test_loader, backbone, device, desc="Offline test"
            )

            print("[SLDA] Computing offline LDA upper bound ...")
            offline_top1, offline_top5 = compute_offline_lda_accuracy(
                off_train_feats, off_train_labels,
                test_feats, test_labels,
                num_classes=num_classes,
                shrinkage_param=cfg["shrinkage_param"],
                device=device,
            )
            print(f"[SLDA] Offline LDA top-1={offline_top1:.2f}%  top-5={offline_top5:.2f}%")

            if offline_top1 > 0:
                ratio = top1 / offline_top1
            omega_ratios.append(ratio)
            print(f"[SLDA] alpha_t/alpha_offline_t = {ratio:.4f}")

        # -------------------------------------------------------------------
        # Log
        # -------------------------------------------------------------------
        results["seen_top1"].append(top1)
        results["seen_top5"].append(top5)
        results["offline_top1"].append(offline_top1)
        results["offline_top5"].append(offline_top5)
        results["omega_ratio"].append(ratio)
        results["classes_seen"].append(class_end)

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        ckpt_name = f"slda_after_increment_{inc_idx+1:02d}_max_class_{class_end}"
        classifier.save_model(save_dir, ckpt_name)

    # -----------------------------------------------------------------------
    # Final Omega_all  (paper's primary reported metric)
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time

    if compute_omega and omega_ratios:
        omega_all = float(np.mean(omega_ratios))
        results["omega_all"] = omega_all
        print(f"\n{'='*60}")
        print(f"[SLDA] Omega_all = {omega_all:.4f}   (primary paper metric)")
        print(f"       = (1/{T}) * sum_t(alpha_t / alpha_offline_t)")
    else:
        omega_all = None
        print("\n[SLDA] Omega_all not computed (compute_omega_all=false).")

    final_top1 = results["seen_top1"][-1]
    final_top5 = results["seen_top5"][-1]
    print(f"[SLDA] Final streaming accuracy:  top-1={final_top1:.2f}%  top-5={final_top5:.2f}%")
    print(f"[SLDA] Total time: {elapsed/60:.1f} min")

    results["final_top1"] = final_top1
    results["final_top5"] = final_top5
    results["total_seconds"] = elapsed

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    classifier.save_model(save_dir, "slda_final")

    print(f"\n[SLDA] Results saved to: {save_dir}/results.json")
    _print_summary(results, omega_ratios, T)


def _print_summary(results: dict, omega_ratios: list, T: int) -> None:
    """Print a clean summary table of per-increment results."""
    print(f"\n{'='*70}")
    print(f"{'Increment':>10} {'Classes':>8} {'SLDA Top1':>10} {'Offline Top1':>13} {'Ratio':>8}")
    print(f"{'-'*70}")
    for i, (c, a, o, r) in enumerate(zip(
        results["classes_seen"],
        results["seen_top1"],
        results["offline_top1"],
        results["omega_ratio"],
    )):
        print(f"{i+1:>10} {c:>8} {a:>9.2f}% {o:>12.2f}% {r:>8.4f}")
    print(f"{'='*70}")
    if results["omega_all"] is not None:
        print(f"Omega_all (primary metric) = {results['omega_all']:.4f}")
    print(f"{'='*70}\n")


# ---------------------------------------------------------------------------
# Dataset factory (ImageNet only, as in paper)
# ---------------------------------------------------------------------------

def _load_dataset(cfg: dict) -> tuple:
    """
    Load the dataset.  Paper uses ImageNet-1000 only.
    Other datasets are accepted but emit a clear warning.
    """
    name = cfg["dataset"].lower()
    root = cfg["data_root"]

    if name == "imagenet":
        return get_imagenet_loader(root, cfg["batch_size"], cfg.get("num_workers", 4))
    else:
        # Allow other datasets as extensions but warn loudly
        print(
            f"\n[SLDA] WARNING: '{name}' is not a paper dataset.\n"
            "Paper only uses ImageNet-1000. Using this dataset will NOT\n"
            "reproduce the paper's numbers or the Omega_all metric as reported.\n"
        )
        from utils import get_dataset_generic
        return get_dataset_generic(name, root, cfg.get("image_size", 224))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Deep SLDA – Hayes & Kanan, CVPRW 2020 (ImageNet)"
    )
    p.add_argument("--config", type=str, default="config.yaml")
    # CLI overrides (only applied if explicitly set)
    p.add_argument("--data_root",  type=str, default=None)
    p.add_argument("--save_dir",   type=str, default=None)
    p.add_argument("--seed",       type=int, default=None)
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument(
        "--no_omega", action="store_true",
        help="Skip Omega_all computation (faster, skips offline LDA upper bound)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data_root:          cfg["data_root"] = args.data_root
    if args.save_dir:           cfg["save_dir"] = args.save_dir
    if args.seed is not None:   cfg["seed"] = args.seed
    if args.resume_checkpoint:  cfg["resume_checkpoint"] = args.resume_checkpoint
    if args.no_omega:           cfg["compute_omega_all"] = False

    print("\n[SLDA] Configuration:")
    print(json.dumps(cfg, indent=2))

    run_experiment(cfg)


if __name__ == "__main__":
    main()

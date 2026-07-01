"""
RanPAC – Training & Evaluation Script
=======================================
Faithful reproduction of the NeurIPS 2023 paper experiments.

Paper:   McDonnell et al., "RanPAC: Random Projections and Pre-trained
         Models for Continual Learning", NeurIPS 2023.

Protocol (paper Section 4):
1.  A pre-trained backbone (ViT-B/16 IN-21k by default) is loaded and
    frozen. All features are extracted through this frozen backbone.
2.  (Optional Phase 1) On the FIRST task only, an adapter/PETL method
    can be fine-tuned via SGD with cosine annealing LR. After Phase 1,
    the backbone is frozen permanently. This is controlled by
    model_name='adapter' in config.
3.  (Phase 2 -- core contribution) For each task:
    a. Extract features from the frozen backbone.
    b. Apply frozen random projection W_rand + ReLU.
    c. Accumulate G += H^T @ H and Q += H^T @ Y across all tasks.
    d. Optimise ridge parameter lambda via 80/20 grid search.
    e. Solve W_o = (G + lambda*I)^{-1} Q via torch.linalg.solve.
    f. Evaluate on all classes seen so far.
4.  Primary metric: Average Incremental Accuracy (paper Table 1).

Usage:
    python train.py --config config.yaml
"""

import argparse
import json
import logging
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from torch import optim
from torch.utils.data import DataLoader

from backbone import get_backbone
from model import RanPACClassifier
from utils import (
    AverageMeter,
    RemappedDataset,
    accuracy,
    get_class_order,
    get_test_transform,
    get_train_transform,
    load_dataset,
    make_class_loader,
)


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
    backbone: nn.Module,
    device: str,
    desc: str = "Extracting",
) -> tuple:
    """
    Forward all images through the frozen backbone, return (features, labels).

    Returns
    -------
    features : Tensor (N, feature_dim)
    labels   : Tensor (N,) long
    """
    backbone.eval()
    feats_list, labels_list = [], []
    n_done = 0
    for batch in loader:
        if len(batch) == 2:
            imgs, labels = batch
        elif len(batch) == 3:
            # Some data managers return (index, imgs, labels)
            _, imgs, labels = batch
        else:
            imgs, labels = batch[0], batch[1]

        imgs = imgs.to(device)
        feats = backbone(imgs)  # (B, feature_dim)
        feats_list.append(feats.cpu())
        labels_list.append(labels)
        n_done += imgs.size(0)
        print(f"\r[{desc}] {n_done}/{len(loader.dataset)}", end="", flush=True)
    print()
    return torch.cat(feats_list, 0), torch.cat(labels_list, 0)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    classifier: RanPACClassifier,
    backbone: nn.Module,
    loader: DataLoader,
    device: str,
    num_classes_seen: int,
) -> tuple:
    """
    Compute top-1 and top-5 accuracy of the RanPAC classifier.

    Returns
    -------
    top1, top5 : float (percentages)
    """
    backbone.eval()
    top1_m, top5_m = AverageMeter(), AverageMeter()
    for batch in loader:
        if len(batch) == 2:
            imgs, labels = batch
        elif len(batch) == 3:
            _, imgs, labels = batch
        else:
            imgs, labels = batch[0], batch[1]

        feats = backbone(imgs.to(device))
        logits = classifier.predict(feats, num_classes_seen=num_classes_seen)
        t1, t5 = accuracy(logits, labels, topk=(1, min(5, num_classes_seen)))
        top1_m.update(t1.item(), imgs.size(0))
        top5_m.update(t5.item(), imgs.size(0))
    return top1_m.avg, top5_m.avg


# ---------------------------------------------------------------------------
# Optional Phase 1: PETL (Adapter) training on first task
# ---------------------------------------------------------------------------

def train_petl_first_task(
    backbone: nn.Module,
    train_loader: DataLoader,
    test_loader: DataLoader,
    num_classes_seen: int,
    cfg: dict,
    device: str,
) -> nn.Module:
    """
    Phase 1: Fine-tune adapter parameters on the first task via SGD.

    Official code (_init_train):
        optimizer = SGD(network.parameters(), lr=body_lr, momentum=0.9, weight_decay=wd)
        scheduler = CosineAnnealingLR(optimizer, T_max=tuned_epoch, eta_min=min_lr)
        for epoch in range(tuned_epoch):
            for (_, inputs, targets) in train_loader:
                logits = network(inputs)["logits"]
                loss = cross_entropy(logits, targets)
                loss.backward()
                optimizer.step()

    After Phase 1, the backbone is frozen permanently.

    Parameters
    ----------
    backbone : nn.Module -- the backbone (may contain trainable adapter params)
    train_loader, test_loader : DataLoader
    num_classes_seen : int
    cfg : dict -- configuration
    device : str

    Returns
    -------
    nn.Module -- the fine-tuned (then frozen) backbone
    """
    tuned_epoch = cfg.get("tuned_epoch", 20)
    body_lr = cfg.get("body_lr", 0.01)
    weight_decay = cfg.get("weight_decay", 0.0005)
    min_lr = cfg.get("min_lr", 1e-8)

    if tuned_epoch <= 0 or body_lr <= 0:
        logging.info("[RanPAC] Phase 1 skipped (tuned_epoch=0 or body_lr=0)")
        return backbone

    logging.info(
        f"[RanPAC] Phase 1: PETL training for {tuned_epoch} epochs, "
        f"lr={body_lr}, wd={weight_decay}"
    )

    # Create a temporary linear head for Phase 1 cross-entropy training
    feature_dim = cfg["feature_dim"]
    phase1_head = nn.Linear(feature_dim, num_classes_seen).to(device)

    # Only unfreeze adapter parameters in backbone
    # For simplicity and faithfulness, we train all backbone params that
    # have requires_grad=True (already set by the adapter architecture)
    trainable_params = [p for p in backbone.parameters() if p.requires_grad]
    trainable_params += list(phase1_head.parameters())

    if len(trainable_params) == 0:
        logging.info("[RanPAC] No trainable parameters in backbone. Skipping Phase 1.")
        return backbone

    optimizer = optim.SGD(
        trainable_params,
        lr=body_lr,
        momentum=0.9,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=tuned_epoch, eta_min=min_lr
    )

    backbone.train()
    for epoch in range(tuned_epoch):
        losses = 0.0
        correct, total = 0, 0
        for batch in train_loader:
            if len(batch) == 2:
                imgs, labels = batch
            else:
                _, imgs, labels = batch

            imgs, labels = imgs.to(device), labels.to(device)

            features = backbone(imgs)
            logits = phase1_head(features)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            losses += loss.item()
            _, preds = torch.max(logits, dim=1)
            correct += preds.eq(labels).cpu().sum().item()
            total += len(labels)

        scheduler.step()
        train_acc = np.around(correct * 100.0 / total, decimals=2)
        logging.info(
            f"[RanPAC] Phase 1 Epoch {epoch+1}/{tuned_epoch}: "
            f"Loss={losses/len(train_loader):.3f}, Train_acc={train_acc:.2f}%"
        )

    # Freeze backbone permanently after Phase 1
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()
    logging.info("[RanPAC] Phase 1 complete. Backbone frozen permanently.")

    # Discard Phase 1 head (it was only for cross-entropy training)
    del phase1_head

    return backbone


# ---------------------------------------------------------------------------
# Main experiment loop
# ---------------------------------------------------------------------------

def run_experiment(cfg: dict) -> None:
    """
    Execute the class-incremental RanPAC experiment from the paper.

    Implements the full evaluation protocol:
    - Class-incremental learning with shuffled class order
    - Phase 1 (optional PETL) on first task
    - Phase 2 (RP + ridge regression) on all tasks
    - Average incremental accuracy reporting
    """
    device = cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if isinstance(device, list):
        device = f"cuda:{device[0]}" if torch.cuda.is_available() else "cpu"
    elif isinstance(device, int):
        device = f"cuda:{device}" if torch.cuda.is_available() else "cpu"

    set_seed(cfg["seed"])
    logging.info(f"\n[RanPAC] Device: {device} | Seed: {cfg['seed']}")
    logging.info(f"[RanPAC] Dataset: {cfg['dataset']}")

    save_dir = cfg["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    # -----------------------------------------------------------------------
    # Dataset
    # -----------------------------------------------------------------------
    train_dataset, test_dataset, num_classes = load_dataset(
        cfg["dataset"], cfg["data_root"], cfg.get("image_size", 224)
    )

    # Override num_classes if specified in config
    if "num_classes" in cfg:
        num_classes = cfg["num_classes"]

    # -----------------------------------------------------------------------
    # Class order (shuffled with seed, matching official code)
    # -----------------------------------------------------------------------
    shuffle_classes = cfg.get("shuffle", True)
    class_order = get_class_order(num_classes, shuffle=shuffle_classes, seed=cfg["seed"])
    logging.info(f"[RanPAC] Class order (first 20): {class_order[:20]}")

    # Remap dataset labels according to class_order
    train_dataset = RemappedDataset(train_dataset, class_order)
    test_dataset = RemappedDataset(test_dataset, class_order)

    # -----------------------------------------------------------------------
    # Backbone
    # -----------------------------------------------------------------------
    backbone, feature_dim = get_backbone(
        arch=cfg["backbone"],
        pretrained=True,
    )
    backbone = backbone.to(device)

    # Verify feature_dim matches config
    if "feature_dim" in cfg and feature_dim != cfg["feature_dim"]:
        logging.warning(
            f"[RanPAC] backbone yields feature_dim={feature_dim} but "
            f"config says feature_dim={cfg['feature_dim']}. Using {feature_dim}."
        )
    cfg["feature_dim"] = feature_dim

    # -----------------------------------------------------------------------
    # Classifier
    # -----------------------------------------------------------------------
    use_RP = cfg.get("use_RP", True)
    M = cfg.get("M", 10000)

    classifier = RanPACClassifier(
        feature_dim=feature_dim,
        num_classes=num_classes,
        use_RP=use_RP,
        M=M,
        device=device,
    )

    # Resume from checkpoint if specified
    resume_ckpt = cfg.get("resume_checkpoint", None)
    if resume_ckpt is not None:
        ckpt_dir, ckpt_fname = os.path.split(resume_ckpt)
        classifier.load_model(ckpt_dir, ckpt_fname.replace(".pth", ""))

    # -----------------------------------------------------------------------
    # Incremental learning protocol
    # -----------------------------------------------------------------------
    init_cls = cfg.get("init_cls", 10)        # Paper: 10
    increment = cfg.get("increment", 10)      # Paper: 10
    batch_size = cfg.get("batch_size", 128)    # Paper: 128
    num_workers = cfg.get("num_workers", 8)

    # Build task schedule
    # Official code: init_cls classes first, then increment per task
    # If init_cls == increment: all tasks have the same size
    tasks = []
    if init_cls == increment:
        for start in range(0, num_classes, increment):
            end = min(start + increment, num_classes)
            tasks.append(list(range(start, end)))
    else:
        tasks.append(list(range(0, init_cls)))
        for start in range(init_cls, num_classes, increment):
            end = min(start + increment, num_classes)
            tasks.append(list(range(start, end)))

    T = len(tasks)
    seen_classes: list = []
    all_task_accs = []

    results = {
        "task_top1": [],
        "task_top5": [],
        "classes_seen": [],
        "avg_incremental_acc": None,
        "config": cfg,
    }

    start_time = time.time()

    for task_idx, task_classes in enumerate(tasks):
        classes_seen_so_far = seen_classes + task_classes
        num_classes_seen = len(classes_seen_so_far)

        logging.info(f"\n{'='*60}")
        logging.info(
            f"[RanPAC] Task {task_idx+1}/{T}: "
            f"classes {task_classes[0]}-{task_classes[-1]} "
            f"(total seen: {num_classes_seen})"
        )

        # -------------------------------------------------------------------
        # Build data loaders for this task
        # -------------------------------------------------------------------
        train_loader = make_class_loader(
            train_dataset, task_classes, batch_size, shuffle=True,
            num_workers=num_workers,
        )
        # Test on ALL seen classes
        test_loader = make_class_loader(
            test_dataset, classes_seen_so_far, batch_size, shuffle=False,
            num_workers=num_workers,
        )

        # -------------------------------------------------------------------
        # Optional Phase 1: PETL training on first task only
        # -------------------------------------------------------------------
        model_name = cfg.get("model_name", "ncm")
        if task_idx == 0 and model_name not in ("ncm", "joint_linear", "joint_full"):
            backbone = train_petl_first_task(
                backbone, train_loader, test_loader,
                num_classes_seen, cfg, device,
            )

        # -------------------------------------------------------------------
        # Phase 2: Feature extraction + RP + ridge regression
        # -------------------------------------------------------------------
        logging.info("[RanPAC] Extracting features for current task...")
        task_feats, task_labels = extract_features(
            train_loader, backbone, device, desc=f"Task {task_idx+1} features"
        )

        if use_RP:
            # Accumulate G and Q
            classifier.accumulate(task_feats, task_labels)

            # Optimise ridge parameter on current task data
            ridge = classifier.optimise_ridge_parameter(task_feats, task_labels)

            # Solve for classifier weights (uses accumulated G and Q)
            classifier.solve(ridge)
            logging.info(
                f"[RanPAC] Ridge regression solved with lambda={ridge:.2e}, "
                f"G shape={classifier.G.shape}, Q shape={classifier.Q.shape}"
            )
        else:
            # NCM baseline: update class prototypes
            classifier.update_prototypes(task_feats, task_labels)
            logging.info("[RanPAC] NCM prototypes updated.")

        # -------------------------------------------------------------------
        # Evaluate on all seen classes
        # -------------------------------------------------------------------
        top1, top5 = evaluate(
            classifier, backbone, test_loader, device, num_classes_seen
        )
        logging.info(f"[RanPAC] Task {task_idx+1}: top-1={top1:.2f}%  top-5={top5:.2f}%")

        seen_classes = classes_seen_so_far
        all_task_accs.append(top1)

        # -------------------------------------------------------------------
        # Log results
        # -------------------------------------------------------------------
        results["task_top1"].append(top1)
        results["task_top5"].append(top5)
        results["classes_seen"].append(num_classes_seen)

        with open(os.path.join(save_dir, "results.json"), "w") as f:
            json.dump(results, f, indent=2)

        # Save checkpoint after each task
        ckpt_name = f"ranpac_after_task_{task_idx+1:02d}_classes_{num_classes_seen}"
        classifier.save_model(save_dir, ckpt_name)

    # -----------------------------------------------------------------------
    # Final results
    # -----------------------------------------------------------------------
    elapsed = time.time() - start_time

    avg_inc_acc = float(np.mean(all_task_accs))
    results["avg_incremental_acc"] = avg_inc_acc
    results["final_top1"] = results["task_top1"][-1]
    results["final_top5"] = results["task_top5"][-1]
    results["total_seconds"] = elapsed

    with open(os.path.join(save_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    classifier.save_model(save_dir, "ranpac_final")

    logging.info(f"\n{'='*60}")
    logging.info(
        f"[RanPAC] Average Incremental Accuracy = {avg_inc_acc:.2f}%  "
        f"(primary paper metric)"
    )
    logging.info(
        f"[RanPAC] Final accuracy: top-1={results['final_top1']:.2f}%  "
        f"top-5={results['final_top5']:.2f}%"
    )
    logging.info(f"[RanPAC] Total time: {elapsed/60:.1f} min")
    logging.info(f"[RanPAC] Results saved to: {save_dir}/results.json")

    _print_summary(results, T)


def _print_summary(results: dict, T: int) -> None:
    """Print a clean summary table of per-task results."""
    print(f"\n{'='*60}")
    print(f"{'Task':>6} {'Classes':>8} {'Top-1':>8} {'Top-5':>8}")
    print(f"{'-'*60}")
    for i, (c, a1, a5) in enumerate(zip(
        results["classes_seen"],
        results["task_top1"],
        results["task_top5"],
    )):
        print(f"{i+1:>6} {c:>8} {a1:>7.2f}% {a5:>7.2f}%")
    print(f"{'='*60}")
    if results["avg_incremental_acc"] is not None:
        print(f"Average Incremental Accuracy = {results['avg_incremental_acc']:.2f}%")
    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="RanPAC – McDonnell et al., NeurIPS 2023"
    )
    p.add_argument("--config", type=str, default="config.yaml")
    # CLI overrides
    p.add_argument("--data_root", type=str, default=None)
    p.add_argument("--save_dir", type=str, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resume_checkpoint", type=str, default=None)
    p.add_argument("--dataset", type=str, default=None)
    p.add_argument("--M", type=int, default=None, help="RP dimension (paper: 10000)")
    p.add_argument(
        "--no_rp", action="store_true",
        help="Disable random projection (NCM baseline)."
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides
    if args.data_root:          cfg["data_root"] = args.data_root
    if args.save_dir:           cfg["save_dir"] = args.save_dir
    if args.seed is not None:   cfg["seed"] = args.seed
    if args.resume_checkpoint:  cfg["resume_checkpoint"] = args.resume_checkpoint
    if args.dataset:            cfg["dataset"] = args.dataset
    if args.M is not None:      cfg["M"] = args.M
    if args.no_rp:              cfg["use_RP"] = False

    logging.info("\n[RanPAC] Configuration:")
    logging.info(json.dumps(cfg, indent=2))

    run_experiment(cfg)


if __name__ == "__main__":
    main()

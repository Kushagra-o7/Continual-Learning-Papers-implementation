# RanPAC – Faithful Paper Reproduction

**Paper:** *RanPAC: Random Projections and Pre-trained Models for Continual Learning*  
Mark D. McDonnell, Dong Gong, Amin Parveneh, Ehsan Abbasnejad, Anton van den Hengel  
NeurIPS 2023  
[📄 Paper](https://arxiv.org/abs/2307.02251) | [💻 Official code](https://github.com/McDonnell-Research-Lab/RanPAC)

---

## Paper Overview

RanPAC is a continual learning method for pre-trained models that avoids catastrophic forgetting
entirely by using **frozen random projections** and **closed-form ridge regression** instead of
gradient-based parameter updates.

### What the paper actually uses

| Item | Paper |
|------|-------|
| **Primary backbone** | ViT-B/16 (ImageNet-21k pretrained). **Paper's main results.** |
| **Ablation backbones** | ResNet-50 (IN-1k), ResNet-152 (IN-1k), CLIP ViT-B/32 (LAION-2B) |
| **CIL benchmarks** | CIFAR-100 (224), ImageNet-A, ImageNet-R, CUB-200, OmniBenchmark, VTAB, Stanford Cars |
| **Primary metric** | Average Incremental Accuracy (mean of top-1 accuracy after each task) |
| **RP dimension M** | 10,000 (default) |
| **Ridge parameter λ** | Optimised per task via grid search over {10⁻⁸, ..., 10⁸} |
| **Class order** | Shuffled with seed 1993 |
| **Protocol** | 10 classes/task for CIFAR-100, 20 classes/task for 200-class datasets |

---

## Algorithm

### Phase 1: Optional PETL (first task only)

For the first task only, an adapter module can be fine-tuned via SGD with cosine
annealing LR. After Phase 1, the entire backbone is frozen permanently.

This is optional and controlled by `model_name` in the config. Set to `'ncm'` to
skip Phase 1 entirely (pure Phase 2 only).

### Phase 2: Random Projection + Ridge Regression (all tasks)

**Random projection** (training-free, frozen):

$$\mathbf{H} = \text{ReLU}(\mathbf{F} \cdot \mathbf{W}_{\text{rand}})$$

where $\mathbf{F} \in \mathbb{R}^{N \times L}$ are backbone features and
$\mathbf{W}_{\text{rand}} \in \mathbb{R}^{L \times M}$ is a frozen Gaussian random matrix.

**Accumulation** (across ALL tasks, no forgetting):

$$\mathbf{G} \mathrel{+}= \mathbf{H}^\top \mathbf{H}, \quad \mathbf{Q} \mathrel{+}= \mathbf{H}^\top \mathbf{Y}$$

where $\mathbf{Y}$ is the one-hot encoding of labels.

**Ridge regression** (closed-form solution):

$$\mathbf{W}_o = (\mathbf{G} + \lambda \mathbf{I})^{-1} \mathbf{Q}$$

solved via `torch.linalg.solve(G + λI, Q)` for numerical stability.

**λ optimisation** (per-task grid search):

- Grid: $\lambda \in \{10^{-8}, 10^{-7}, \ldots, 10^{8}\}$
- 80/20 train/val split on current task features
- MSE loss on validation set
- Best λ selected by argmin

### Prediction

$$\hat{y} = \arg\max_k \; \mathbf{H} \cdot \mathbf{W}_o^\top$$

---

## File Structure

```
ranpac/
├── model.py         # RanPACClassifier (RP + ridge regression + accumulation)
├── backbone.py      # ViT-B/16, ResNet-50/152 via timm
├── train.py         # Training + evaluation loop
├── utils.py         # Dataset loaders, metrics, transforms
├── config.yaml      # Paper-faithful config (all hyperparameters)
├── requirements.txt
├── run.sh
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run (CIFAR-100, auto-downloaded)

```bash
python train.py --config config.yaml
```

Or:
```bash
bash run.sh
```

### 3. For other datasets

Download datasets and set `data_root` in config:
```yaml
dataset: "imageneta"
data_root: "./data/imagenet-a"    # Must contain train/ and test/
num_classes: 200
init_cls: 20
increment: 20
```

---

## Configuration

| Key | Paper Value | Description |
|-----|-------------|-------------|
| `dataset` | `cifar224` | Dataset name (see paper benchmarks) |
| `data_root` | `./data` | Path to dataset root |
| `backbone` | `pretrained_vit_b16_224_in21k` | Pre-trained backbone (paper primary) |
| `feature_dim` | `768` | Backbone output dimension |
| `use_RP` | `true` | Enable random projection + ridge regression |
| `M` | `10000` | RP output dimension (paper default) |
| `init_cls` | `10` | Classes in first task |
| `increment` | `10` | New classes per subsequent task |
| `shuffle` | `true` | Shuffle class order |
| `seed` | `1993` | Random seed |
| `model_name` | `ncm` | `'ncm'` (no PETL) or `'adapter'` (Phase 1 PETL) |
| `tuned_epoch` | `20` | Phase 1 SGD epochs (if using PETL) |
| `body_lr` | `0.01` | Phase 1 learning rate |
| `batch_size` | `128` | Batch size |

---

## Expected Results (from paper Table 1)

### CIFAR-100 (10-task CIL, ViT-B/16 backbone)

| Method | Avg Inc. Acc |
|--------|-------------|
| NCM (no PETL) | ~86.3% |
| RanPAC w/o RP, no PETL (M=0) | ~88.7% |
| **RanPAC (M=10000), no PETL** | **~90.9%** |
| RanPAC + Adapter PETL | ~92.7% |

### Other datasets (ViT-B/16, RanPAC + Adapter)

| Dataset | Avg Inc. Acc |
|---------|-------------|
| ImageNet-A | ~66.3% |
| ImageNet-R | ~82.6% |
| CUB-200 | ~89.3% |
| OmniBenchmark | ~80.4% |
| VTAB | ~85.7% |
| Stanford Cars | ~76.3% |

### ResNet-50 backbone (no PETL)

| Method | Avg Inc. Acc (CIFAR-100) |
|--------|------------------------|
| NCM | ~63.1% |
| RanPAC w/o RP (M=0) | ~72.1% |
| RanPAC (M=10000) | ~72.1% |

---

## What Is and Is Not Implemented

### Implemented (faithful to paper)

| ✅ | Detail |
|----|--------|
| Frozen random projection (W_rand) | Gaussian i.i.d., M=10000, with ReLU activation |
| Gram matrix accumulation (G) | `G += H^T @ H` across all tasks |
| Prototype accumulation (Q) | `Q += H^T @ Y` across all tasks |
| Ridge regression (closed-form) | `W_o = solve(G + λI, Q)` via `torch.linalg.solve` |
| λ optimisation | Grid search {10⁻⁸,...,10⁸}, 80/20 split, MSE loss |
| NCM baseline | Cosine similarity with class-mean prototypes |
| Decorrelation-only mode (M=0) | Ridge regression on raw features (no RP) |
| ViT-B/16 (IN-21k) backbone | Via timm, matching official code |
| ResNet-50/152 backbones | Via timm, matching official code |
| All 7 CIL benchmarks | CIFAR-100, ImageNet-A/R, CUB-200, OmniBenchmark, VTAB, Cars |
| Class-order shuffling | Seeded random permutation |
| Deterministic seeds | torch, numpy, random, cuDNN |
| Checkpoint save/load | All state (G, Q, W_rand, W_o, hyperparameters) |
| Average incremental accuracy | Paper's primary metric |

### Not Implemented

| ❌ | Reason |
|----|--------|
| **Full adapter/SSF/VPT architectures** | These require modifying the ViT architecture with trainable adapter layers. A simplified Phase 1 PETL is provided, but the full timm-based adapter variants (e.g., `pretrained_vit_b16_224_in21k_adapter`) require custom ViT modifications from the official code. |
| **Domain-incremental learning (DIL)** | Paper tests Core50, CDDB, DomainNet for DIL. These have a different protocol (same classes, different domains) and are not implemented here. |
| **Joint training baselines** | Joint linear probe and joint full fine-tuning (paper rows 0-1) are offline upper bounds, not CL methods. |
| **CLIP backbone** | `vit_base_patch32_224_clip_laion2b` requires a CLIP-specific timm variant. The loader is provided but may need timm version adjustment. |
| **Multiple GPU training** | Paper's DataParallel wrapper is not reproduced. Single GPU is used. |

---

## Assumptions

1. **Frozen backbone throughout Phase 2**: No gradient updates to the backbone after Phase 1 (or ever, if no PETL).
2. **Gaussian random projection**: W_rand ~ N(0, 1), i.i.d., frozen after initialisation.
3. **Class-ordered tasks**: Classes are assigned to tasks in a shuffled (seeded) order, then presented task-by-task.
4. **Ridge regression replaces SGD**: No backpropagation is used for the classifier head in Phase 2. The solution is computed in closed form.
5. **Accumulation = no forgetting**: Since G and Q are summed across all tasks, the ridge regression solution at any point is identical to fitting on ALL data seen so far.
6. **λ optimised per task**: The ridge parameter is re-optimised using current-task data at each task boundary, then applied to the accumulated G and Q.

---

## Expected Runtime

### CIFAR-100 (10 tasks, GPU: RTX 3090)

| Phase | Time |
|-------|------|
| Backbone loading (ViT-B/16 IN-21k) | ~10 sec |
| Feature extraction per task | ~30 sec |
| Ridge regression solve per task | ~2 sec |
| λ optimisation per task | ~5 sec |
| **Total (10 tasks, no PETL)** | **~10 min** |
| **Total (10 tasks, with Phase 1 PETL)** | **~20 min** |

### ImageNet-A/R (10 tasks, GPU: RTX 3090)

| Phase | Time |
|-------|------|
| Feature extraction per task | ~2 min |
| **Total (10 tasks)** | **~30 min** |

---

## Citation

```bibtex
@inproceedings{mcdonnell2023ranpac,
    title     = {RanPAC: Random Projections and Pre-trained Models for Continual Learning},
    author    = {McDonnell, Mark D. and Gong, Dong and Parveneh, Amin
                 and Abbasnejad, Ehsan and van den Hengel, Anton},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year      = {2023}
}
```

# Deep SLDA – Faithful Paper Reproduction

**Paper:** *Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis*  
Tyler L. Hayes & Christopher Kanan  
CVPR Workshops (CLVision), 2020  
[📄 PDF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf) | [💻 Official code](https://github.com/tyler-hayes/Deep_SLDA)

---

## Paper Overview

Deep SLDA couples a **completely frozen ResNet-18** backbone with a streaming LDA classifier
to enable class-incremental learning from a one-pass data stream, with no replay buffer.

### What the paper actually uses

| Item | Paper |
|------|-------|
| **Dataset** | ImageNet-1000 (ILSVRC 2012). **Only dataset in the paper.** |
| **Backbone** | ResNet-18. **Only architecture in the paper.** |
| **Backbone init** | Offline training on **first 100 ImageNet classes** (shared by all methods). |
| **Features** | From `layer4.1` of ResNet-18, spatially mean-pooled → 512-d. |
| **Primary metric** | Ω_all = (1/T) Σ_t (α_t / α_offline,t) |
| **Secondary metric** | Top-1 / Top-5 accuracy after each 100-class increment |
| **Baselines** | Fine-tuning (θF), Fine-tuning (θF+θG), ExStream, iCaRL, End-to-End |

---

## Algorithm

### Streaming update (one sample at a time)

$$\delta = \frac{n}{n+1}(\mathbf{x} - \boldsymbol{\mu}_k)^\top(\mathbf{x} - \boldsymbol{\mu}_k)$$

$$\boldsymbol{\Sigma} \leftarrow \frac{n \cdot \boldsymbol{\Sigma} + \delta}{n+1}, \quad \boldsymbol{\mu}_k \leftarrow \boldsymbol{\mu}_k + \frac{\mathbf{x} - \boldsymbol{\mu}_k}{c_k + 1}$$

### Prediction (LDA discriminant rule)

$$\boldsymbol{\Lambda} = \bigl[(1-\varepsilon)\boldsymbol{\Sigma} + \varepsilon \mathbf{I}\bigr]^{-1}$$

$$\hat{y} = \arg\max_k \;\bigl(\mathbf{x}^\top \boldsymbol{\Lambda} \boldsymbol{\mu}_k - \tfrac{1}{2}\boldsymbol{\mu}_k^\top \boldsymbol{\Lambda} \boldsymbol{\mu}_k\bigr)$$

**Note on Λ computation:** The paper uses a regular matrix inverse `[...]^{-1}` (not pseudo-inverse),
because `(1-ε)Σ + εI` is positive definite for any ε > 0. This implementation uses
`torch.linalg.inv` accordingly.

### Primary Metric: Ω_all

$$\Omega_{all} = \frac{1}{T} \sum_{t=1}^{T} \frac{\alpha_t}{\alpha_{offline,t}}$$

- **α_t** — accuracy of the streaming SLDA model after t increments
- **α_offline,t** — accuracy of an **offline LDA** trained on ALL data seen up to t (upper bound)
- **T** — total increments (9 for ImageNet with base_classes=100, class_increment=100)

This metric is computed by default (`compute_omega_all: true` in config).

---

## File Structure

```
slda/
├── model.py        # StreamingLDA class (core algorithm)
├── train.py        # Training + evaluation with Omega_all
├── backbone.py     # ResNet-18 feature extractor (paper-faithful)
├── utils.py        # ImageNet loader, metrics
├── config.yaml     # Paper-faithful config (ImageNet, all hyperparameters)
├── requirements.txt
├── run.sh
└── README.md
```

---

## Quick Start

### 1. Get the data and checkpoint

```bash
# ImageNet-1000 must be manually downloaded from https://image-net.org/
# After downloading, your data_root must look like:
#   /path/to/imagenet/train/n01440764/...
#   /path/to/imagenet/val/n01440764/...

# The authors' backbone checkpoint (ResNet-18 trained on first 100 ImageNet classes):
# Download from: https://github.com/tyler-hayes/Deep_SLDA
# Place at: ./imagenet_files/imagenet_100_class_ckpt.pth
```

### 2. Set your data path

Edit `config.yaml`:
```yaml
data_root: "/your/path/to/imagenet"
backbone_checkpoint: "./imagenet_files/imagenet_100_class_ckpt.pth"
```

### 3. Run

```bash
pip install -r requirements.txt
python train.py --config config.yaml
```

Or:
```bash
bash run.sh
```

---

## Configuration

| Key | Paper Value | Description |
|-----|-------------|-------------|
| `dataset` | `imagenet` | **Must be `imagenet` for paper reproduction** |
| `data_root` | — | Path to ImageNet root (`train/` + `val/`) |
| `num_classes` | `1000` | Total ImageNet classes |
| `backbone` | `resnet18` | **ResNet-18 only in paper** |
| `feature_dim` | `512` | ResNet-18 layer4.1 mean-pooled output |
| `backbone_checkpoint` | authors' ckpt | ResNet-18 trained on first 100 classes |
| `imagenet_pretrained` | `false` | Must be false when using checkpoint |
| `shrinkage_param` | `1e-4` | ε in Λ = [(1-ε)Σ + εI]⁻¹ |
| `streaming_update_sigma` | `true` | Online Σ update (main); `false` = ablation |
| `base_classes` | `100` | Classes in OAS-init base increment |
| `class_increment` | `100` | New classes per streaming increment |
| `shuffle_data` | `false` | Class-ordered stream (no shuffling) |
| `compute_omega_all` | `true` | Compute Ω_all (paper's primary metric) |
| `seed` | `0` | Random seed |

---

## Expected Results (from paper Table 1)

| Method | Ω_all | Final Top-1 | Final Top-5 |
|--------|-------|-------------|-------------|
| Deep SLDA (streaming Σ) | ~0.95 | ~63.5% | ~83.4% |
| Deep SLDA (fixed Σ)     | ~0.97 | ~65.0% | ~84.5% |

Results shown above require the authors' backbone checkpoint.
Using a full ImageNet-pretrained ResNet-18 will give different numbers.

---

## Backbone Initialization Protocol

The paper explicitly states: *"all models use the same offline base CNN initialization procedure."*

This means:
1. A ResNet-18 is trained **offline** (full data, full epochs) on the **first 100 ImageNet classes**.
2. This trained network becomes the **frozen feature extractor** for ALL streaming experiments.
3. The training data for these 100 classes is then re-used to initialize SLDA via `fit_base()`.

This implementation handles this via:
- `backbone_checkpoint`: path to the offline-trained ResNet-18 (authors provide it).
- `fit_base()`: OAS covariance estimation on all base-100-class features.

Without the authors' checkpoint, results will **not** match the paper.

---

## What Is and Is Not Implemented

### Implemented (faithful to paper)

| ✅ | Detail |
|----|--------|
| Streaming covariance update (Welford-style) | Exactly as in paper / official code |
| Online class-mean update | Exact formula |
| OAS base initialization | `sklearn.covariance.OAS(assume_centered=True)` |
| Precision matrix via regular inverse | `torch.linalg.inv` — matches paper formula |
| Layer4.1 extraction + spatial mean pool | Hook-based, matches `retrieve_any_layer.py` |
| Omega_all metric | (1/T)Σ(α_t / α_offline_t), computed at each increment |
| Offline LDA upper bound | Fresh OAS-LDA on all seen data at each t |
| Checkpoint save/load | Sufficient statistics + hyperparameters |
| Deterministic seeds | torch, numpy, random, cuDNN |
| Class-ordered stream | shuffle_data=false |

### Not Implemented

| ❌ | Reason |
|----|--------|
| **Comparison baselines** (iCaRL, ExStream, Fine-tuning, End-to-End) | Separate methods; outside the scope of this SLDA reproduction. Each would require its own implementation. |
| **Per-class covariance** | Paper mentions it briefly but uses tied Σ as main contribution |
| **Online training of backbone** | Paper explicitly uses frozen backbone; this is intentional |
| **Multiple datasets** | Paper uses **only ImageNet**. CIFAR-100/Tiny-ImageNet in the extension config are provided for development convenience only — they will NOT reproduce any paper number. |

---

## Assumptions

1. **Frozen backbone throughout**: No gradient updates to ResNet-18 at any point.
2. **Tied covariance**: One shared Σ across all classes (standard LDA assumption).
3. **Class-ordered stream**: Data arrives class by class (not shuffled), matching the paper.
4. **One sample at a time**: `fit()` is called individually for each streaming sample.
5. **OAS for base**: `sklearn OAS(assume_centered=True)` as used in official `fit_base()`.
6. **Offline upper bound**: α_offline,t is an OAS-LDA re-fit from scratch on all seen features.

---

## Expected Runtime (ImageNet, RTX 3090)

| Phase | Time |
|-------|------|
| Pre-extracting all features (for Ω_all) | ~20 min |
| Base init (100 classes, OAS) | ~2 min |
| Each streaming increment (100 classes) | ~5 min feature extraction |
| Offline LDA upper bound per increment | ~2 min |
| **Total** | **~90–120 min** |

Skip Ω_all computation (faster, no offline upper bound):
```bash
python train.py --config config.yaml --no_omega
```

---

## Citation

```bibtex
@InProceedings{Hayes_2020_CVPR_Workshops,
    author    = {Hayes, Tyler L. and Kanan, Christopher},
    title     = {Lifelong Machine Learning With Deep Streaming Linear Discriminant Analysis},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision
                 and Pattern Recognition (CVPR) Workshops},
    month     = {June},
    year      = {2020}
}
```

# Deep SLDA – Streaming Linear Discriminant Analysis

**Paper reproduced:** *Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis*  
Tyler L. Hayes & Christopher Kanan  
CVPR Workshops (CLVision), 2020  
[📄 PDF](https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf) | [💻 Official code](https://github.com/tyler-hayes/Deep_SLDA)

---

## Overview

**Deep SLDA** couples a *completely frozen* deep backbone with a streaming LDA classifier.  It is designed for the **class-incremental** (data stream) scenario where:

- Each sample is seen only **once** (no replay buffer).
- No gradients are backpropagated through the backbone.
- The classifier updates in **O(d²)** time per sample, where d is the feature dimension.

### Core Algorithm

The classifier maintains three sufficient statistics:

| Symbol | Shape | Description |
|--------|-------|-------------|
| `muK`  | (C, d) | Per-class mean vectors |
| `cK`   | (C,)  | Per-class sample counts |
| `Sigma`| (d, d) | Shared (tied) covariance matrix |

**Streaming update** (one sample at a time):

$$\delta = \frac{n}{n+1}(\mathbf{x} - \boldsymbol{\mu}_k)^\top(\mathbf{x} - \boldsymbol{\mu}_k)$$

$$\boldsymbol{\Sigma} \leftarrow \frac{n \cdot \boldsymbol{\Sigma} + \delta}{n+1}$$

$$\boldsymbol{\mu}_k \leftarrow \boldsymbol{\mu}_k + \frac{\mathbf{x} - \boldsymbol{\mu}_k}{c_k + 1}$$

**Prediction** (LDA discriminant scores):

$$\boldsymbol{\Lambda} = \text{pinv}\bigl((1-\varepsilon)\boldsymbol{\Sigma} + \varepsilon \mathbf{I}\bigr)$$

$$\hat{y} = \arg\max_k \;\bigl(\boldsymbol{\Lambda}\boldsymbol{\mu}_k^\top \mathbf{x} - \tfrac{1}{2}\boldsymbol{\mu}_k \boldsymbol{\Lambda}\boldsymbol{\mu}_k^\top\bigr)$$

where ε = `shrinkage_param` (default 1e-4).

---

## File Structure

```
slda/
├── model.py          # StreamingLDA class (core algorithm)
├── train.py          # Training + evaluation script
├── backbone.py       # Frozen feature extractors (ResNet, VGG)
├── utils.py          # Datasets, metrics, transforms
├── config.yaml       # Reproducible hyperparameter configuration
├── requirements.txt  # Python dependencies
└── run.sh            # Single-command execution script
```

---

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run (single command)

```bash
python train.py --config config.yaml
```

Or using the shell script:

```bash
bash run.sh config.yaml
```

### 3. Custom dataset / overrides

You can override any config key from the CLI:

```bash
python train.py --config config.yaml \
    --dataset cifar100 \
    --data_root ./data \
    --save_dir ./results/my_run \
    --seed 42
```

---

## Configuration (`config.yaml`)

| Key | Default | Description |
|-----|---------|-------------|
| `dataset` | `cifar100` | Dataset: `cifar100`, `tiny-imagenet`, `imagenet` |
| `data_root` | `./data` | Where data lives / is downloaded |
| `num_classes` | `100` | Total classes in the stream |
| `backbone` | `resnet18` | Backbone arch: `resnet18`, `resnet50`, `vgg16_bn`, … |
| `feature_dim` | `512` | Feature dimension (must match backbone) |
| `imagenet_pretrained` | `true` | Load ImageNet-pretrained backbone weights |
| `backbone_checkpoint` | `null` | Custom backbone .pth path (overrides pretrained) |
| `shrinkage_param` | `1e-4` | Tikhonov regularisation ε for precision matrix |
| `streaming_update_sigma` | `true` | Update Σ online; `false` freezes it after base-init |
| `base_classes` | `10` | Classes used in OAS-init (first increment) |
| `class_increment` | `10` | Classes added per streaming increment |
| `batch_size` | `256` | Feature extraction / evaluation batch size |
| `test_batch_size` | `1024` | Internal predict() mini-batch size |
| `shuffle_data` | `false` | Whether to shuffle training stream (paper: false) |
| `seed` | `0` | Global random seed |
| `save_dir` | `./results/slda_cifar100` | Output directory for checkpoints & JSON results |
| `resume_checkpoint` | `null` | Path to a .pth file to resume from |

---

## ImageNet Experiment (paper's main result)

The paper runs SLDA on ImageNet-1K with a ResNet-18 backbone pre-trained on the first 100 classes.  To reproduce:

```yaml
# config.yaml (ImageNet)
dataset: imagenet
data_root: /path/to/imagenet
num_classes: 1000
backbone: resnet18
feature_dim: 512
imagenet_pretrained: false
backbone_checkpoint: ./imagenet_files/imagenet_100_class_ckpt.pth
base_classes: 100
class_increment: 100
save_dir: ./results/slda_imagenet
```

```bash
python train.py --config config.yaml
```

**Expected results (from paper, Table 1):**

| Method | Top-1 | Top-5 |
|--------|-------|-------|
| Deep SLDA (streaming Σ) | ~63.5% | ~83.4% |
| Deep SLDA (fixed Σ)     | ~65.0% | ~84.5% |

---

## Checkpointing

The model is saved after each class-increment:

```
results/slda_cifar100/
├── config.json
├── accuracies.json
├── slda_min0_max10.pth
├── slda_min0_max20.pth
├── ...
└── slda_final.pth
```

**Resume from a checkpoint:**

```bash
python train.py --config config.yaml \
    --resume_checkpoint ./results/slda_cifar100/slda_min0_max50.pth
```

**Load in Python:**

```python
from model import StreamingLDA

model = StreamingLDA.from_checkpoint(
    save_path="./results/slda_cifar100",
    save_name="slda_final",
)
```

---

## Assumptions

1. **Frozen backbone**: The backbone is always fully frozen. No gradient updates to the feature extractor are performed (faithful to the paper).
2. **Tied covariance**: A single shared Σ is used across all classes (LDA assumption). The paper also ablates class-specific covariance (not implemented here).
3. **OAS for base init**: The first increment uses Oracle Approximating Shrinkage (OAS) from scikit-learn for better-conditioned Σ initialisation, matching the paper's `fit_base`.
4. **Class-ordered stream**: Data arrives in class-ordered batches (`shuffle_data: false`). The streaming equations are derived for this protocol.
5. **`fit` called once per sample**: The inner loop in `fit_batch` calls `fit(xi, yi)` for each sample individually, as done in the official code.

---

## Unsupported Features

| Feature | Status | Notes |
|---------|--------|-------|
| Per-class covariance | ❌ Not implemented | Paper mentions it briefly; tied Σ is the main contribution |
| Task boundaries required | ❌ Not required | SLDA is task-agnostic; labels are always observed at train time |
| Data augmentation at test time | ❌ Not implemented | Paper doesn't use TTA |
| Distributed training | ❌ Not applicable | SLDA has no backprop, distributed training isn't relevant |
| Multi-GPU feature extraction | ⚠️ Untested | Single-GPU feature extraction is assumed |

---

## Expected Runtime

| Dataset | GPU | Backbone | Expected Time |
|---------|-----|----------|---------------|
| CIFAR-100 | RTX 3090 | ResNet-18 (pretrained) | ~3–5 min |
| Tiny-ImageNet | RTX 3090 | ResNet-18 (pretrained) | ~10–15 min |
| ImageNet-1K | RTX 3090 | ResNet-18 (custom ckpt) | ~60–90 min |

SLDA training time is dominated by feature extraction (forward pass through the frozen backbone), not the LDA updates themselves.

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

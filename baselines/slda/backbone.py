"""
Deep SLDA - Backbone: ResNet-18 Feature Extractor
==================================================
The paper (Hayes & Kanan, CVPRW 2020) uses ONE backbone architecture:

    ResNet-18, pre-trained offline on the base 100 ImageNet classes.

Feature extraction follows the paper's official code (retrieve_any_layer.py):
    1. Forward input through the ResNet-18 up to and including layer4.1
       (the last residual block, before the global average pooling layer).
    2. Spatially mean-pool the (N, 512, H, W) output -> (N, 512).

This is equivalent to using the global average pooling output of ResNet-18
(feature_dim = 512), which is what the paper reports.

Other architectures (resnet50, etc.) are NOT used in the paper.
They are included here only as extensions for users who want to experiment.
The paper results are ONLY reproducible with resnet18 + the provided checkpoint.

Backbone initialization protocol (paper Section 4 / Section 3.1):
------------------------------------------------------------------
All methods in the paper share the SAME offline base CNN initialization:
  - ResNet-18 is trained offline on the FIRST 100 ImageNet classes.
  - The resulting checkpoint is then used as the FROZEN feature extractor
    for ALL streaming experiments (SLDA, iCaRL, ExStream, Fine-tuning, etc.).
  - The official checkpoint is provided by the authors at:
    https://github.com/tyler-hayes/Deep_SLDA (imagenet_files/imagenet_100_class_ckpt.pth)
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# ---------------------------------------------------------------------------
# Forward hook helper (mirrors official retrieve_any_layer.py)
# ---------------------------------------------------------------------------

class _HookCapture:
    """Registers a forward hook on a named submodule and stores its output."""

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.output = None
        module = dict(model.named_modules())[layer_name]
        module.register_forward_hook(self._hook)

    def _hook(self, module, inp, out) -> None:
        self.output = out.detach()


# ---------------------------------------------------------------------------
# Paper backbone: ResNet-18 with layer4.1 extraction + spatial mean pool
# ---------------------------------------------------------------------------

class ResNet18FeatureExtractor(nn.Module):
    """
    ResNet-18 feature extractor as used in the paper.

    Extracts from 'layer4.1' (the last residual block) and
    spatially mean-pools to produce a 512-d vector -- identical to the
    output of ResNet-18's global average pooling layer.

    This is the ONLY backbone used in the paper experiments.
    """

    FEATURE_DIM = 512  # ResNet-18 layer4 channel width

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.model = base
        # Register hook on layer4.1 (paper: retrieve_any_layer with 'layer4.1')
        self._capture = _HookCapture(self.model, "layer4.1")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, 3, H, W)

        Returns
        -------
        Tensor (N, 512) -- spatially mean-pooled features from layer4.1
        """
        self.model(x)                              # trigger hook
        feat = self._capture.output                # (N, 512, h, w)
        # Spatial mean pool: paper's pool_feat()
        # feat shape: (N, C, h, w) -> (N, h*w, C) -> mean over spatial -> (N, C)
        N, C, h, w = feat.shape
        feat = feat.permute(0, 2, 3, 1)            # (N, h, w, C)
        feat = feat.reshape(N, h * w, C)           # (N, h*w, C)
        feat = feat.mean(dim=1)                    # (N, C)
        return feat


# ---------------------------------------------------------------------------
# Extension backbones (NOT in the paper -- clearly marked)
# ---------------------------------------------------------------------------

class _ResNetAvgPoolExtractor(nn.Module):
    """
    Generic ResNet backbone using the standard global average pooling output.
    NOT used in the paper. Provided as a convenience extension only.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        # Strip the final FC layer; keep everything up to and including avgpool
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x).flatten(1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Paper backbone
_PAPER_ARCH = "resnet18"

# Registry: arch_name -> (build_fn, feature_dim)
# resnet18 uses the hook-based extractor to exactly match the paper.
# Others use avgpool output and are NOT paper-faithful.
_REGISTRY = {
    "resnet18": None,   # handled separately below (hook-based)
    # --- Extensions (not in paper) ---
    "resnet34":  (lambda pt: _ResNetAvgPoolExtractor(models.resnet34(
        weights=models.ResNet34_Weights.IMAGENET1K_V1 if pt else None)), 512),
    "resnet50":  (lambda pt: _ResNetAvgPoolExtractor(models.resnet50(
        weights=models.ResNet50_Weights.IMAGENET1K_V1 if pt else None)), 2048),
    "resnet101": (lambda pt: _ResNetAvgPoolExtractor(models.resnet101(
        weights=models.ResNet101_Weights.IMAGENET1K_V1 if pt else None)), 2048),
}


def get_backbone(
    arch: str = "resnet18",
    pretrained: bool = False,
    checkpoint: str = None,
) -> tuple[nn.Module, int]:
    """
    Build and return (backbone, feature_dim).

    Parameters
    ----------
    arch : str
        Architecture name. 'resnet18' is the paper's backbone.
        Others are extensions not validated against the paper.
    pretrained : bool
        Load torchvision ImageNet-1k weights.
        For paper reproduction: set False and provide checkpoint instead.
    checkpoint : str or None
        Path to a .pth checkpoint. For the paper's exact results, use the
        authors' checkpoint: imagenet_files/imagenet_100_class_ckpt.pth
        When provided, pretrained is ignored.

    Returns
    -------
    (nn.Module, int)  -- (backbone in eval mode, feature_dim)
    """
    arch = arch.lower()

    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone '{arch}'. "
            f"Available: {list(_REGISTRY.keys())}. "
            f"Paper uses 'resnet18'."
        )

    if arch != _PAPER_ARCH:
        print(
            f"[Backbone] WARNING: '{arch}' is NOT used in the paper. "
            f"Paper backbone is 'resnet18'. Results will NOT reproduce paper numbers."
        )

    # Build base ResNet-18 (paper architecture)
    if arch == "resnet18":
        if checkpoint is not None:
            # Load from checkpoint (paper protocol: offline-trained on base 100 classes)
            base = models.resnet18(weights=None)
            feature_dim = ResNet18FeatureExtractor.FEATURE_DIM
            backbone = ResNet18FeatureExtractor(base)
            _load_checkpoint(backbone.model, checkpoint)
            print(
                "[Backbone] Loaded paper-protocol checkpoint (ResNet-18 "
                f"offline-trained on base classes): {checkpoint}"
            )
        elif pretrained:
            # Full ImageNet-1k pretrained (NOT what the paper uses, but valid for
            # experiments without the authors' checkpoint)
            print(
                "[Backbone] WARNING: Using full ImageNet-1k pretrained ResNet-18. "
                "Paper uses a ResNet-18 trained ONLY on the first 100 classes. "
                "Results will differ from the paper."
            )
            base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            feature_dim = ResNet18FeatureExtractor.FEATURE_DIM
            backbone = ResNet18FeatureExtractor(base)
        else:
            # Random init (for debugging only)
            print("[Backbone] WARNING: ResNet-18 with random weights (no checkpoint, no pretrain).")
            base = models.resnet18(weights=None)
            feature_dim = ResNet18FeatureExtractor.FEATURE_DIM
            backbone = ResNet18FeatureExtractor(base)
    else:
        builder, feature_dim = _REGISTRY[arch]
        backbone = builder(pretrained and checkpoint is None)
        if checkpoint is not None:
            _load_checkpoint(backbone, checkpoint)

    return backbone.eval(), feature_dim


def _load_checkpoint(model: nn.Module, path: str) -> None:
    """
    Load weights from a checkpoint into model.
    Handles 'state_dict' and 'model_state' keys, and strips 'module.' prefix
    from DataParallel-wrapped checkpoints.
    """
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Strip DataParallel prefix
    state = {k.replace("module.", ""): v for k, v in state.items()}

    missing, unexpected = model.load_state_dict(state, strict=False)
    n_miss = len(missing)
    n_unex = len(unexpected)
    if n_miss > 0:
        print(f"[Backbone] {n_miss} missing keys (expected if FC head is absent).")
    if n_unex > 0:
        print(f"[Backbone] {n_unex} unexpected keys.")
    print(f"[Backbone] Checkpoint loaded from: {path}")

"""
Deep SLDA – Backbone Module
============================
Provides frozen feature extractors compatible with Deep SLDA.

The paper uses ResNet-18 with the final FC layer removed so that the
output of the global average pooling layer (feature_dim=512) is used as
the feature vector fed into StreamingLDA.

Supported architectures
------------------------
  - resnet18   (default, paper)  → feature_dim = 512
  - resnet34                      → feature_dim = 512
  - resnet50                      → feature_dim = 2048
  - resnet101                     → feature_dim = 2048
  - vgg16_bn                      → feature_dim = 4096  (penultimate FC)

Adding a new backbone:
  1. Implement a function that returns a ``nn.Module``.
  2. The module's forward() must accept an image tensor (N, C, H, W)
     and return a flat feature vector (N, d).
  3. Register it in ``_REGISTRY``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision import models


# --------------------------------------------------------------------------- #
# Generic ResNet wrapper – removes the FC layer
# --------------------------------------------------------------------------- #

class ResNetFeatureExtractor(nn.Module):
    """
    ResNet backbone with the final classification head removed.

    The network is registered in .eval() mode and its parameters are
    frozen outside this module (the caller is responsible).

    forward() returns the output of global average pooling, shape (N, d).
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        # Keep everything except the final fully-connected layer
        self.features = nn.Sequential(*list(base.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # (N, C, H, W) → (N, d, 1, 1) → (N, d)
        out = self.features(x)
        return out.flatten(1)


# --------------------------------------------------------------------------- #
# VGG16 with BN – penultimate hidden layer (4096-d)
# --------------------------------------------------------------------------- #

class VGG16BNFeatureExtractor(nn.Module):
    """
    VGG-16 with batch norm.  Returns the 4096-d penultimate activation.
    """

    def __init__(self, base: nn.Module) -> None:
        super().__init__()
        self.features = base.features
        self.avgpool = base.avgpool
        # Classifier up to (but not including) the last linear layer
        # default VGG16: classifier = [Linear, ReLU, Dropout, Linear, ReLU, Dropout, Linear]
        self.classifier = nn.Sequential(*list(base.classifier.children())[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

def _build_resnet18(pretrained: bool) -> ResNetFeatureExtractor:
    base = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
    return ResNetFeatureExtractor(base)


def _build_resnet34(pretrained: bool) -> ResNetFeatureExtractor:
    base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)
    return ResNetFeatureExtractor(base)


def _build_resnet50(pretrained: bool) -> ResNetFeatureExtractor:
    base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
    return ResNetFeatureExtractor(base)


def _build_resnet101(pretrained: bool) -> ResNetFeatureExtractor:
    base = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
    return ResNetFeatureExtractor(base)


def _build_vgg16bn(pretrained: bool) -> VGG16BNFeatureExtractor:
    base = models.vgg16_bn(weights=models.VGG16_BN_Weights.IMAGENET1K_V1 if pretrained else None)
    return VGG16BNFeatureExtractor(base)


_REGISTRY = {
    "resnet18":  (_build_resnet18,  512),
    "resnet34":  (_build_resnet34,  512),
    "resnet50":  (_build_resnet50,  2048),
    "resnet101": (_build_resnet101, 2048),
    "vgg16_bn":  (_build_vgg16bn,   4096),
}


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #

def get_backbone(
    arch: str,
    pretrained: bool = False,
    checkpoint: str | None = None,
    feature_layer: str = "avgpool",     # kept for API compat; unused for ResNet
) -> nn.Module:
    """
    Build and return a frozen feature extractor.

    Parameters
    ----------
    arch : str
        Architecture name (see ``_REGISTRY``).
    pretrained : bool
        If True, load ImageNet-pretrained weights from torchvision.
        Ignored when ``checkpoint`` is provided.
    checkpoint : str | None
        Path to a .pth checkpoint whose 'state_dict' or 'model_state'
        key will be loaded into the backbone (as in the paper's
        ImageNet-trained ResNet-18 for the base-100 experiment).
    feature_layer : str
        Unused; kept for forward compatibility.

    Returns
    -------
    nn.Module in eval mode (parameters NOT frozen here – done in train.py).
    """
    arch = arch.lower()
    if arch not in _REGISTRY:
        raise ValueError(
            f"Unknown backbone '{arch}'. "
            f"Available: {list(_REGISTRY.keys())}"
        )

    builder, feature_dim = _REGISTRY[arch]
    backbone = builder(pretrained=(pretrained and checkpoint is None))

    if checkpoint is not None:
        _load_checkpoint(backbone, checkpoint)

    return backbone.eval()


def _load_checkpoint(model: nn.Module, path: str) -> None:
    """Load weights from a checkpoint file into ``model``."""
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model_state" in ckpt:
            state = ckpt["model_state"]
        else:
            state = ckpt
    else:
        state = ckpt

    # Strip 'module.' prefix added by DataParallel
    state = {k.replace("module.", ""): v for k, v in state.items()}

    # Load into the feature extractor sub-module if keys match
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        # Try to load into the inner .features sub-module
        inner = getattr(model, "features", model)
        missing2, unexpected2 = inner.load_state_dict(state, strict=False)
        if missing2:
            print(
                f"[Backbone] Warning: {len(missing2)} missing keys after "
                f"trying both model and model.features."
            )
    print(f"[Backbone] Loaded checkpoint from: {path}")


def feature_dim_for(arch: str) -> int:
    """Return the feature dimensionality for a given backbone architecture."""
    arch = arch.lower()
    if arch not in _REGISTRY:
        raise ValueError(f"Unknown backbone '{arch}'.")
    return _REGISTRY[arch][1]

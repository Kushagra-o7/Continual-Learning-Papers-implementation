"""
RanPAC – Backbone: Pre-trained Feature Extractors
===================================================
The paper (McDonnell et al., NeurIPS 2023) uses the following backbones:

    Primary:   ViT-B/16 pre-trained on ImageNet-21k (paper's main results)
    Ablations: ResNet-50  (ImageNet-1k pretrained)
               ResNet-152 (ImageNet-1k pretrained)
               CLIP ViT-B/32 (LAION-2B pretrained)

The official code uses the `timm` library (version 0.6.12) to load backbones.
We faithfully reproduce this.

Backbone initialization protocol (paper Section 4):
----------------------------------------------------
All backbones are FROZEN after loading. No backbone parameters are updated
during the continual learning phase (Phase 2).

Optional Phase 1 (PETL):
    For the first task only, an adapter module can be fine-tuned via SGD.
    After Phase 1, the entire backbone (including adapter) is frozen.
    This is handled in train.py, not here.

Feature extraction:
    - ViT-B/16: CLS token output -> 768-d
    - ResNet-50: Global average pool -> 2048-d
    - ResNet-152: Global average pool -> 2048-d
    - CLIP ViT-B/32: CLS token output -> 768-d (via timm)
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn

try:
    import timm
except ImportError:
    raise ImportError(
        "timm is required for RanPAC backbones. Install with: pip install timm>=0.6.12"
    )


# ---------------------------------------------------------------------------
# Feature dimension registry (matches official code / timm defaults)
# ---------------------------------------------------------------------------

_FEATURE_DIMS = {
    "vit_base_patch16_224": 768,
    "vit_base_patch16_224_in21k": 768,
    "pretrained_vit_b16_224_in21k": 768,        # paper's name
    "resnet50": 2048,
    "pretrained_resnet50": 2048,                 # paper's name
    "resnet152": 2048,
    "pretrained_resnet152": 2048,                # paper's name
    "vit_base_patch32_224_clip_laion2b": 768,    # CLIP backbone
}


# ---------------------------------------------------------------------------
# Backbone wrapper (timm-based, matches official code)
# ---------------------------------------------------------------------------

class TimmBackbone(nn.Module):
    """
    Wraps a timm model to extract features (no classification head).

    The official code (inc_net.py) uses timm.create_model with num_classes=0
    to strip the classification head and return features directly.
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor (N, 3, H, W)

        Returns
        -------
        Tensor (N, feature_dim) -- CLS token for ViT, avgpool for ResNet
        """
        return self.model(x)


# ---------------------------------------------------------------------------
# Input normalization (for ResNet backbones)
# ---------------------------------------------------------------------------

class InputNormalize(nn.Module):
    """
    Per-channel normalization of input features (not images).

    Official code (inc_net.py) applies this to ResNet features:
        features = (features - mean) / std

    This is the 'use_input_norm' flag in the official CSV configs.
    Applied to ResNet backbone outputs, NOT to ViT outputs.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mean = None
        self.std = None
        self.fitted = False

    @torch.no_grad()
    def fit(self, features: torch.Tensor) -> None:
        """Compute mean and std from a batch of features."""
        self.mean = features.mean(dim=0)
        self.std = features.std(dim=0).clamp(min=1e-8)
        self.fitted = True

    @torch.no_grad()
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            return features
        return (features - self.mean.to(features.device)) / self.std.to(features.device)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

# Paper primary backbone
_PAPER_PRIMARY = "pretrained_vit_b16_224_in21k"

# Mapping from paper names to timm model names
_PAPER_TO_TIMM = {
    "pretrained_vit_b16_224_in21k": "vit_base_patch16_224",
    "pretrained_resnet50": "resnet50",
    "pretrained_resnet152": "resnet152",
    "vit_base_patch32_224_clip_laion2b": "vit_base_patch32_224_clip_laion2b",
    # Direct timm names
    "vit_base_patch16_224": "vit_base_patch16_224",
    "resnet50": "resnet50",
    "resnet152": "resnet152",
}


def get_backbone(
    arch: str = "pretrained_vit_b16_224_in21k",
    pretrained: bool = True,
) -> tuple[nn.Module, int]:
    """
    Build and return (backbone, feature_dim).

    Parameters
    ----------
    arch : str
        Architecture name. Use paper names for faithful reproduction:
        - 'pretrained_vit_b16_224_in21k' (paper primary, ViT-B/16 IN-21k)
        - 'pretrained_resnet50' (paper ablation)
        - 'pretrained_resnet152' (paper ablation)
        - 'vit_base_patch32_224_clip_laion2b' (paper ablation)
    pretrained : bool
        Load pretrained weights (should always be True for paper reproduction).

    Returns
    -------
    (nn.Module, int) -- (backbone in eval mode, feature_dim)
    """
    arch_lower = arch.lower()

    if arch_lower not in _PAPER_TO_TIMM:
        raise ValueError(
            f"Unknown backbone '{arch}'. "
            f"Available: {list(_PAPER_TO_TIMM.keys())}. "
            f"Paper primary: '{_PAPER_PRIMARY}'."
        )

    if arch_lower != _PAPER_PRIMARY:
        logging.warning(
            f"[Backbone] '{arch}' is not the paper's primary backbone. "
            f"Paper primary: '{_PAPER_PRIMARY}'. Results may differ."
        )

    timm_name = _PAPER_TO_TIMM[arch_lower]

    # Determine feature_dim
    feature_dim = _FEATURE_DIMS.get(arch_lower)
    if feature_dim is None:
        feature_dim = _FEATURE_DIMS.get(timm_name, 768)

    # Handle ViT-B/16 with IN-21k weights specifically
    if arch_lower == "pretrained_vit_b16_224_in21k":
        # Official code loads ViT-B/16 pretrained on ImageNet-21k
        # timm uses 'vit_base_patch16_224' with pretrained=True (IN-1k by default)
        # For IN-21k: timm uses 'vit_base_patch16_224.augreg_in21k' or similar
        try:
            model = timm.create_model(
                "vit_base_patch16_224.augreg_in21k",
                pretrained=pretrained,
                num_classes=0,  # Strip classification head
            )
            logging.info(
                "[Backbone] Loaded ViT-B/16 pretrained on ImageNet-21k "
                "(vit_base_patch16_224.augreg_in21k)"
            )
        except Exception:
            # Fallback: use standard IN-1k pretrained
            logging.warning(
                "[Backbone] IN-21k variant not available in this timm version. "
                "Falling back to IN-1k pretrained ViT-B/16."
            )
            model = timm.create_model(
                "vit_base_patch16_224",
                pretrained=pretrained,
                num_classes=0,
            )
            logging.info("[Backbone] Loaded ViT-B/16 pretrained on ImageNet-1k (fallback)")
    else:
        model = timm.create_model(
            timm_name,
            pretrained=pretrained,
            num_classes=0,  # Strip classification head -> returns features
        )
        logging.info(f"[Backbone] Loaded '{timm_name}' (pretrained={pretrained})")

    backbone = TimmBackbone(model)

    # Freeze all parameters
    for p in backbone.parameters():
        p.requires_grad_(False)
    backbone.eval()

    logging.info(
        f"[Backbone] Feature dim: {feature_dim} | "
        f"All parameters frozen ({sum(p.numel() for p in backbone.parameters()):,} params)"
    )

    return backbone, feature_dim

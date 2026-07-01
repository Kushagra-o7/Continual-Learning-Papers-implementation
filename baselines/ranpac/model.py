"""
RanPAC: Random Projections and Pre-trained Models for Continual Learning
=========================================================================
Faithful implementation of:

    McDonnell, M. D., Gong, D., Parveneh, A., Abbasnejad, E.,
    & van den Hengel, A. (2023).
    RanPAC: Random Projections and Pre-trained Models for Continual Learning.
    NeurIPS 2023.

    Paper: https://arxiv.org/abs/2307.02251
    Official code: https://github.com/McDonnell-Research-Lab/RanPAC

Algorithm (Phase 2 -- core contribution)
-----------------------------------------
Given a frozen pre-trained backbone producing L-dimensional features:

1. **Random Projection**: A frozen Gaussian random matrix W_rand in R^{L x M}
   is applied followed by ReLU activation:
       H = ReLU(features @ W_rand)         # (N, M)

2. **Accumulation**: Two matrices are accumulated across ALL tasks:
       G += H^T @ H                         # (M, M) -- Gram matrix
       Q += H^T @ Y                         # (M, C) -- feature-label cross-correlation
   where Y is the one-hot encoding of labels.

3. **Ridge Regression**: Classifier weights are solved in closed form:
       W_o = solve(G + lambda * I, Q)^T     # (C, M)
   using torch.linalg.solve for numerical stability (NOT matrix inverse).

4. **Lambda Optimisation**: The ridge parameter lambda is optimised per-task
   via grid search over {10^{-8}, ..., 10^{8}} using an 80/20 validation split
   of the CURRENT task's projected features, minimising MSE loss.

Key insight: since G and Q are accumulated, the ridge regression solution at
any point is IDENTICAL to training on ALL data seen so far -- no forgetting.

When M=0 (no RP, decorrelation only):
    H = features (no projection), still uses ridge regression on raw features.

When use_RP=False (NCM baseline):
    Uses cosine similarity with class-mean prototypes (no ridge regression).
"""

import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class RanPACClassifier(nn.Module):
    """
    RanPAC classifier head implementing random projection + ridge regression.

    This corresponds to the CosineLinear + replace_fc logic from the official code,
    restructured into a clean standalone module.

    Parameters
    ----------
    feature_dim : int
        Dimensionality L of input features from the frozen backbone.
        Paper: 768 for ViT-B/16, 2048 for ResNet-50, 2048 for ResNet-152.
    num_classes : int
        Total number of classes C across all tasks.
    use_RP : bool
        True  -> use random projection + ridge regression (paper's method).
        False -> use cosine similarity NCM baseline.
    M : int
        Random projection output dimensionality.
        Paper default: 10000.
        M=0  -> decorrelation only (no RP, but still ridge regression).
        M=-1 -> no RP, no ridge regression (NCM only).
    device : str or None
        Compute device. Auto-detected if None.
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        use_RP: bool = True,
        M: int = 10000,
        device: str = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.feature_dim = feature_dim   # L in the paper
        self.num_classes = num_classes   # C total
        self.use_RP = use_RP
        self.M = M

        # ------------------------------------------------------------------
        # Determine effective projection dimension
        # ------------------------------------------------------------------
        if use_RP and M > 0:
            # Random projection: W_rand in R^{L x M}
            self.proj_dim = M
        elif use_RP and M == 0:
            # Decorrelation only (no RP): use raw features
            self.proj_dim = feature_dim
        else:
            # NCM baseline: M=-1 or use_RP=False
            self.proj_dim = feature_dim

        # ------------------------------------------------------------------
        # Random projection matrix (frozen, Gaussian i.i.d.)
        # Paper Section 3.2: "W_rand is a fixed random matrix"
        # Official code: torch.randn(in_features, M)
        # ------------------------------------------------------------------
        if use_RP and M > 0:
            # Register as buffer so it's saved with the model but not trained
            self.register_buffer(
                "W_rand",
                torch.randn(feature_dim, M, device=device),
            )
        else:
            self.W_rand = None

        # ------------------------------------------------------------------
        # Accumulation matrices (paper Section 3.2)
        # G: Gram matrix / feature auto-correlation  (proj_dim, proj_dim)
        # Q: Feature-label cross-correlation          (proj_dim, num_classes)
        # ------------------------------------------------------------------
        if use_RP:
            self.register_buffer(
                "G", torch.zeros(self.proj_dim, self.proj_dim, device=device)
            )
            self.register_buffer(
                "Q", torch.zeros(self.proj_dim, num_classes, device=device)
            )

        # ------------------------------------------------------------------
        # Classifier weights  (num_classes, proj_dim)
        # When use_RP=True:  set via ridge regression (not trained by SGD)
        # When use_RP=False: set to class-mean prototypes (cosine similarity)
        # ------------------------------------------------------------------
        self.register_buffer(
            "W_o",
            torch.zeros(num_classes, self.proj_dim, device=device),
        )

        # Cosine similarity scaling factor (sigma in official CosineLinear)
        # Only used in NCM mode (use_RP=False)
        if not use_RP:
            self.sigma = nn.Parameter(torch.ones(1, device=device))
        else:
            self.sigma = None

        # Track per-class sample counts (for NCM prototype averaging)
        self.register_buffer(
            "class_counts",
            torch.zeros(num_classes, device=device),
        )

        # Track the current ridge parameter
        self.current_ridge = 1.0
        # Track number of total samples seen
        self.num_updates = 0

    # ------------------------------------------------------------------
    # Random projection (paper Section 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def project(self, features: torch.Tensor) -> torch.Tensor:
        """
        Apply random projection + ReLU to features.

        Paper: H = ReLU(features @ W_rand)
        Official code (CosineLinear.forward):
            inn = torch.nn.functional.relu(input @ self.W_rand)

        Parameters
        ----------
        features : Tensor (N, L)

        Returns
        -------
        Tensor (N, M) or (N, L) if M=0
        """
        if self.W_rand is not None:
            return F.relu(features @ self.W_rand)
        else:
            return features

    # ------------------------------------------------------------------
    # Accumulation (paper Section 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def accumulate(self, features: torch.Tensor, labels: torch.Tensor) -> None:
        """
        Accumulate the Gram matrix G and cross-correlation matrix Q
        from a batch of features for the current task.

        Official code (replace_fc):
            Features_h = ReLU(Features_f @ W_rand)
            self.Q = self.Q + Features_h.T @ Y
            self.G = self.G + Features_h.T @ Features_h

        Parameters
        ----------
        features : Tensor (N, L) -- raw backbone features
        labels   : Tensor (N,)   -- integer class labels
        """
        assert self.use_RP, "accumulate() is only for use_RP=True mode"

        features = features.to(self.device, dtype=torch.float32)
        labels = labels.long().to(self.device)

        # Project features
        H = self.project(features)  # (N, proj_dim)

        # One-hot encode labels -> Y (N, C)
        Y = torch.zeros(
            labels.size(0), self.num_classes, device=self.device, dtype=torch.float32
        )
        Y.scatter_(1, labels.unsqueeze(1), 1.0)

        # Accumulate (paper equations)
        self.Q += H.t() @ Y          # (proj_dim, C)
        self.G += H.t() @ H          # (proj_dim, proj_dim)
        self.num_updates += features.size(0)

    # ------------------------------------------------------------------
    # Ridge parameter optimisation (paper Section 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def optimise_ridge_parameter(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> float:
        """
        Optimise the ridge parameter lambda via grid search.

        Official code (optimise_ridge_parameter):
            ridges = 10.0**np.arange(-8, 9)
            num_val_samples = int(Features.shape[0] * 0.8)
            # 80% train, 20% val split
            # For each ridge: solve, compute MSE on val set
            # Return best ridge

        Parameters
        ----------
        features : Tensor (N, L) -- raw backbone features for current task
        labels   : Tensor (N,)   -- integer class labels

        Returns
        -------
        float -- optimal ridge parameter
        """
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.long().to(self.device)

        # Project features
        H = self.project(features)  # (N, proj_dim)

        # One-hot encode labels -> Y (N, C)
        Y = torch.zeros(
            labels.size(0), self.num_classes, device=self.device, dtype=torch.float32
        )
        Y.scatter_(1, labels.unsqueeze(1), 1.0)

        # 80/20 split (paper / official code)
        num_val_samples = int(H.shape[0] * 0.8)

        Q_val = H[:num_val_samples].t() @ Y[:num_val_samples]    # (proj_dim, C)
        G_val = H[:num_val_samples].t() @ H[:num_val_samples]    # (proj_dim, proj_dim)

        # Grid search over ridges (official code: 10.0**np.arange(-8, 9))
        ridges = 10.0 ** np.arange(-8, 9)
        losses = []
        eye = torch.eye(G_val.size(0), device=self.device)

        for ridge in ridges:
            # Solve ridge regression on 80% split
            Wo = torch.linalg.solve(G_val + ridge * eye, Q_val).t()  # (C, proj_dim)
            # Predict on 20% validation split
            Y_pred = H[num_val_samples:] @ Wo.t()   # (N_val, C)
            # MSE loss
            loss = F.mse_loss(Y_pred, Y[num_val_samples:])
            losses.append(loss.item())

        best_ridge = ridges[np.argmin(losses)]
        logging.info(f"[RanPAC] Optimal lambda: {best_ridge}")
        self.current_ridge = float(best_ridge)
        return float(best_ridge)

    # ------------------------------------------------------------------
    # Solve classifier weights (paper Section 3.2)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def solve(self, ridge: float = None) -> None:
        """
        Solve for classifier weights via ridge regression.

        Official code (replace_fc):
            Wo = torch.linalg.solve(
                self.G + ridge * torch.eye(self.G.size(dim=0)), self.Q
            ).T

        Parameters
        ----------
        ridge : float or None
            Ridge parameter lambda. Uses self.current_ridge if None.
        """
        assert self.use_RP, "solve() is only for use_RP=True mode"

        if ridge is None:
            ridge = self.current_ridge

        eye = torch.eye(self.G.size(0), device=self.device)
        # Paper: W_o = (G + lambda * I)^{-1} Q
        # Official code uses torch.linalg.solve for numerical stability
        Wo = torch.linalg.solve(self.G + ridge * eye, self.Q).t()  # (C, proj_dim)
        self.W_o = Wo

    # ------------------------------------------------------------------
    # NCM (cosine similarity) class prototype update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_prototypes(
        self, features: torch.Tensor, labels: torch.Tensor
    ) -> None:
        """
        Update class-mean prototypes for NCM (cosine similarity) baseline.

        Official code (replace_fc when use_RP=False):
            class_prototype = Features_f[data_index].mean(0)
            self._network.fc.weight.data[class_index] = class_prototype

        Parameters
        ----------
        features : Tensor (N, L)
        labels   : Tensor (N,)
        """
        features = features.to(self.device, dtype=torch.float32)
        labels = labels.long().to(self.device)

        for class_idx in torch.unique(labels):
            mask = labels == class_idx
            class_prototype = features[mask].mean(dim=0)
            self.W_o[class_idx] = class_prototype
            self.class_counts[class_idx] = mask.sum().float()

        self.num_updates += features.size(0)

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, features: torch.Tensor, num_classes_seen: int = None
    ) -> torch.Tensor:
        """
        Compute logits for input features.

        When use_RP=True (paper's method):
            H = ReLU(features @ W_rand)
            logits = H @ W_o.T

        When use_RP=False (NCM baseline):
            logits = sigma * cosine_similarity(features, W_o)

        Official code (CosineLinear.forward):
            if use_RP: out = F.linear(inn, self.weight)
            else:      out = F.linear(F.normalize(input), F.normalize(self.weight))

        Parameters
        ----------
        features         : Tensor (N, L) -- raw backbone features
        num_classes_seen : int or None
            If set, only return logits for classes 0..num_classes_seen-1.

        Returns
        -------
        Tensor (N, C) or (N, num_classes_seen) logits on CPU.
        """
        features = features.to(self.device, dtype=torch.float32)

        if self.use_RP:
            H = self.project(features)  # (N, proj_dim)
            logits = F.linear(H, self.W_o)  # (N, C)
        else:
            # Cosine similarity (NCM)
            logits = F.linear(
                F.normalize(features, p=2, dim=1),
                F.normalize(self.W_o, p=2, dim=1),
            )
            if self.sigma is not None:
                logits = self.sigma * logits

        if num_classes_seen is not None:
            logits = logits[:, :num_classes_seen]

        return logits.cpu()

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_model(self, save_path: str, save_name: str) -> None:
        """
        Persist all state to a .pth file.
        Saves: W_rand, G, Q, W_o, class_counts, hyperparameters.
        """
        os.makedirs(save_path, exist_ok=True)
        ckpt = {
            "W_o": self.W_o.cpu(),
            "class_counts": self.class_counts.cpu(),
            "num_updates": self.num_updates,
            "current_ridge": self.current_ridge,
            # Hyperparameters for reconstruction
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "use_RP": self.use_RP,
            "M": self.M,
        }
        if self.use_RP:
            ckpt["G"] = self.G.cpu()
            ckpt["Q"] = self.Q.cpu()
        if self.W_rand is not None:
            ckpt["W_rand"] = self.W_rand.cpu()
        if self.sigma is not None:
            ckpt["sigma"] = self.sigma.data.cpu()

        path = os.path.join(save_path, save_name + ".pth")
        torch.save(ckpt, path)
        logging.info(f"[RanPAC] Checkpoint saved: {path}")

    def load_model(self, save_path: str, save_name: str) -> None:
        """
        Load state from a .pth checkpoint into this instance.
        """
        path = os.path.join(save_path, save_name + ".pth")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.W_o = ckpt["W_o"].to(self.device)
        self.class_counts = ckpt["class_counts"].to(self.device)
        self.num_updates = ckpt["num_updates"]
        self.current_ridge = ckpt["current_ridge"]
        if self.use_RP:
            self.G = ckpt["G"].to(self.device)
            self.Q = ckpt["Q"].to(self.device)
        if "W_rand" in ckpt and self.W_rand is not None:
            self.W_rand = ckpt["W_rand"].to(self.device)
        if "sigma" in ckpt and self.sigma is not None:
            self.sigma.data = ckpt["sigma"].to(self.device)
        logging.info(f"[RanPAC] Checkpoint loaded: {path}")

    @classmethod
    def from_checkpoint(cls, save_path: str, save_name: str, device: str = None):
        """
        Construct a RanPACClassifier directly from a checkpoint file.
        """
        path = os.path.join(save_path, save_name + ".pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            feature_dim=ckpt["feature_dim"],
            num_classes=ckpt["num_classes"],
            use_RP=ckpt["use_RP"],
            M=ckpt["M"],
            device=device,
        )
        instance.load_model(save_path, save_name)
        return instance

    def __repr__(self) -> str:
        return (
            f"RanPACClassifier(feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes}, "
            f"use_RP={self.use_RP}, M={self.M}, "
            f"proj_dim={self.proj_dim}, "
            f"num_updates={self.num_updates})"
        )

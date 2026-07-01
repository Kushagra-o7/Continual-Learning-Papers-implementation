"""
Deep Streaming Linear Discriminant Analysis (Deep SLDA)
========================================================
Faithful implementation of:

    Hayes, T. L., & Kanan, C. (2020).
    Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis.
    CVPR Workshops (CLVision), 2020.

    Paper: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/
           Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf

Algorithm Summary
-----------------
Deep SLDA couples a *frozen* deep feature extractor with a streaming LDA
classifier.  The classifier maintains:

  * muK  (C × d) – per-class mean vectors
  * cK   (C,)    – per-class sample counts
  * Sigma (d × d)– shared (tied) covariance matrix (updated online or fixed)

Given a new sample (x, y):

  Covariance update (streaming):
      delta  = (x - muK[y])^T (x - muK[y]) * n / (n + 1)
      Sigma  = (n * Sigma + delta) / (n + 1)

  Mean update:
      muK[y] += (x - muK[y]) / (cK[y] + 1)
      cK[y]  += 1

Prediction (LDA discriminant scores):
      Lambda = pinv((1 - eps) * Sigma + eps * I)    # precision matrix
      W      = Lambda @ muK.T                        # (d × C)
      c      = 0.5 * diag(muK @ W)                  # bias terms  (C,)
      score(x) = x @ W - c                          # (N × C)
      y_hat  = argmax(score(x), dim=1)

Base Initialization:
  When sufficient data are available at startup, the covariance is estimated
  via Oracle Approximating Shrinkage (OAS) from sklearn, which is more
  accurate than the streaming one-pass estimator on large batches.
"""

import os
import torch
import torch.nn as nn


class StreamingLDA(nn.Module):
    """
    Streaming Linear Discriminant Analysis classifier.

    This module is NOT a PyTorch neural network in the conventional sense –
    it does not use backprop.  Inheriting from nn.Module lets us use
    ``save_model`` / ``load_model`` seamlessly alongside PyTorch checkpoints.

    Parameters
    ----------
    feature_dim : int
        Dimensionality d of the input feature vectors.
    num_classes : int
        Total number of output classes C (must be known in advance).
    shrinkage_param : float
        Tikhonov/shrinkage regularisation ε used when computing the
        precision matrix Lambda = pinv((1-ε)Σ + εI).
        Paper default: 1e-4.
    streaming_update_sigma : bool
        If True, Σ is updated online (one sample at a time) after the
        base-init phase.  If False, Σ is frozen after base-init.
        Paper ablation: both variants are studied; streaming is default.
    test_batch_size : int
        Mini-batch size used internally during inference to avoid OOM on
        large test sets.
    device : str | None
        'cuda', 'cpu', or None (auto-detect).
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        shrinkage_param: float = 1e-4,
        streaming_update_sigma: bool = True,
        test_batch_size: int = 1024,
        device: str = None,
    ):
        super().__init__()

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.shrinkage_param = shrinkage_param
        self.streaming_update_sigma = streaming_update_sigma
        self.test_batch_size = test_batch_size

        # ------------------------------------------------------------------ #
        # Sufficient statistics maintained by the classifier
        # ------------------------------------------------------------------ #
        # muK  : per-class mean vectors           shape (C, d)
        # cK   : per-class sample counts          shape (C,)
        # Sigma: shared tied covariance matrix    shape (d, d)
        # ------------------------------------------------------------------ #
        self.muK = torch.zeros(num_classes, feature_dim, device=device)
        self.cK = torch.zeros(num_classes, device=device)
        # Initialised to identity (Sigma = I) so the first predictions are
        # reasonable before any covariance estimate is available.
        self.Sigma = torch.eye(feature_dim, device=device)
        self.num_updates = 0  # total samples seen (across all classes)

        # Cache the precision matrix so we only recompute it when Sigma changes
        self.Lambda = torch.zeros(feature_dim, feature_dim, device=device)
        self._prev_num_updates = -1  # sentinel for cache invalidation

    # ---------------------------------------------------------------------- #
    # Training methods
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the classifier with a *single* new (feature, label) pair.

        This is the core streaming update.  Call this once per sample in
        the order they arrive from the stream.

        Parameters
        ----------
        x : Tensor of shape (d,) or (1, d)
            Feature vector extracted from the frozen backbone.
        y : Tensor scalar or shape (1,)
            Integer class label.
        """
        x = x.to(self.device, dtype=torch.float32)
        y = y.long().to(self.device)

        # Ensure correct shapes
        if x.dim() < 2:
            x = x.unsqueeze(0)          # (1, d)
        if y.dim() == 0:
            y = y.unsqueeze(0)          # (1,)

        # ------------------------------------------------------------------ #
        # Streaming covariance update (Eq. 2 in paper / Welford-style)
        #
        #   delta = (x - mu_k)^T (x - mu_k) * n / (n + 1)
        #   Sigma = (n * Sigma + delta) / (n + 1)
        #
        # where n = self.num_updates (total updates so far)
        # ------------------------------------------------------------------ #
        if self.streaming_update_sigma:
            x_minus_mu = x - self.muK[y]                   # (1, d)
            mult = x_minus_mu.t().mm(x_minus_mu)           # (d, d)
            n = self.num_updates
            delta = mult * n / (n + 1)
            self.Sigma = (n * self.Sigma + delta) / (n + 1)

        # ------------------------------------------------------------------ #
        # Class-mean update (online mean formula)
        #
        #   muK[y] += (x - muK[y]) / (cK[y] + 1)
        # ------------------------------------------------------------------ #
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def fit_batch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the classifier one sample at a time over a batch.

        Convenience wrapper – identical to calling ``fit`` in a loop.
        Used during the incremental (non-base) streaming phase.

        Parameters
        ----------
        X : Tensor of shape (N, d)
        y : Tensor of shape (N,) or (N, 1)
        """
        X = X.to(self.device, dtype=torch.float32)
        y = y.view(-1).long().to(self.device)
        for xi, yi in zip(X, y):
            self.fit(xi, yi.view(1))

    @torch.no_grad()
    def fit_base(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Initialise the classifier from a batch of base-class data.

        This replaces the streaming estimator for the first (base) increment:
          * Class means are computed exactly from the full base batch.
          * Σ is estimated using Oracle Approximating Shrinkage (OAS) from
            sklearn, which gives a better-conditioned estimate than Welford
            on a moderate-sized batch.

        Parameters
        ----------
        X : Tensor of shape (N, d)
            All feature vectors for the base classes.
        y : Tensor of shape (N,) or (N, 1)
            Corresponding integer labels.
        """
        X = X.to(self.device, dtype=torch.float32)
        y = y.squeeze().long().to(self.device)

        print("[SLDA] Fitting base initialisation ...")

        # Exact class-mean computation
        for k in torch.unique(y):
            mask = y == k
            self.muK[k] = X[mask].mean(dim=0)
            self.cK[k] = mask.sum().float()

        self.num_updates = X.shape[0]

        # OAS covariance estimation on mean-centred features
        print("[SLDA] Estimating initial covariance matrix via OAS ...")
        from sklearn.covariance import OAS

        X_centered = (X - self.muK[y]).cpu().numpy()
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        self.Sigma = (
            torch.from_numpy(oas.covariance_).float().to(self.device)
        )
        print("[SLDA] Base init complete.")

    # ---------------------------------------------------------------------- #
    # Inference
    # ---------------------------------------------------------------------- #

    @torch.no_grad()
    def predict(
        self, X: torch.Tensor, return_probas: bool = False
    ) -> torch.Tensor:
        """
        Compute LDA discriminant scores (or softmax probabilities).

        LDA decision rule:
            Lambda = pinv((1 - eps) * Sigma + eps * I)
            W      = Lambda @ muK.T               (d × C)
            c      = 0.5 * diag(muK @ W)          bias (C,)
            score  = X @ W - c                    (N × C)

        Parameters
        ----------
        X : Tensor of shape (N, d)
        return_probas : bool
            If True, return softmax(scores) instead of raw scores.

        Returns
        -------
        Tensor of shape (N, C) on CPU.
        """
        X = X.to(self.device, dtype=torch.float32)
        num_samples = X.shape[0]

        # Recompute precision matrix only when model has been updated
        if self._prev_num_updates != self.num_updates:
            print("\n[SLDA] Model updated - recomputing Lambda (precision matrix)...")
            eps = self.shrinkage_param
            reg_sigma = (1.0 - eps) * self.Sigma + eps * torch.eye(
                self.feature_dim, device=self.device
            )
            self.Lambda = torch.linalg.pinv(reg_sigma)
            self._prev_num_updates = self.num_updates

        # Precompute shared weight matrix and bias
        M = self.muK.t()                                    # (d, C)
        W = self.Lambda @ M                                 # (d, C)
        c = 0.5 * (M * W).sum(dim=0)                       # (C,)

        # Inference in mini-batches to avoid OOM
        mb = min(self.test_batch_size, num_samples)
        scores = torch.empty(num_samples, self.num_classes, device="cpu")

        for start in range(0, num_samples, mb):
            end = min(start + mb, num_samples)
            x_mb = X[start:end]                             # (≤mb, d)
            scores[start:end] = (x_mb @ W - c).cpu()       # (≤mb, C)

        if return_probas:
            return torch.softmax(scores, dim=1)
        return scores

    # ---------------------------------------------------------------------- #
    # Checkpoint helpers
    # ---------------------------------------------------------------------- #

    def save_model(self, save_path: str, save_name: str) -> None:
        """
        Persist the SLDA sufficient statistics to disk.

        Saves: muK, cK, Sigma, num_updates, feature_dim, num_classes,
               shrinkage_param, streaming_update_sigma.

        Parameters
        ----------
        save_path : str   directory where the file will be written
        save_name : str   filename (without extension)
        """
        os.makedirs(save_path, exist_ok=True)
        ckpt = {
            "muK": self.muK.cpu(),
            "cK": self.cK.cpu(),
            "Sigma": self.Sigma.cpu(),
            "num_updates": self.num_updates,
            # Hyperparameters needed to reconstruct the model
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "shrinkage_param": self.shrinkage_param,
            "streaming_update_sigma": self.streaming_update_sigma,
        }
        path = os.path.join(save_path, save_name + ".pth")
        torch.save(ckpt, path)
        print(f"[SLDA] Checkpoint saved -> {path}")

    def load_model(self, save_path: str, save_name: str) -> None:
        """
        Load SLDA sufficient statistics from a checkpoint.

        Parameters
        ----------
        save_path : str   directory containing the checkpoint
        save_name : str   filename (without extension)
        """
        path = os.path.join(save_path, save_name + ".pth")
        ckpt = torch.load(path, map_location=self.device)
        self.muK = ckpt["muK"].to(self.device)
        self.cK = ckpt["cK"].to(self.device)
        self.Sigma = ckpt["Sigma"].to(self.device)
        self.num_updates = ckpt["num_updates"]
        # Reset precision cache so it is recomputed on next predict()
        self._prev_num_updates = -1
        print(f"[SLDA] Checkpoint loaded <- {path}")

    @classmethod
    def from_checkpoint(cls, save_path: str, save_name: str, device: str = None):
        """
        Construct a StreamingLDA instance directly from a checkpoint.

        Parameters
        ----------
        save_path, save_name : str
            As in ``load_model``.
        device : str | None
            Target device; auto-detected if None.

        Returns
        -------
        StreamingLDA
        """
        path = os.path.join(save_path, save_name + ".pth")
        ckpt = torch.load(path, map_location="cpu")
        instance = cls(
            feature_dim=ckpt["feature_dim"],
            num_classes=ckpt["num_classes"],
            shrinkage_param=ckpt["shrinkage_param"],
            streaming_update_sigma=ckpt["streaming_update_sigma"],
            device=device,
        )
        instance.load_model(save_path, save_name)
        return instance

    # ---------------------------------------------------------------------- #
    # Utility
    # ---------------------------------------------------------------------- #

    def __repr__(self) -> str:
        return (
            f"StreamingLDA("
            f"feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes}, "
            f"shrinkage={self.shrinkage_param}, "
            f"streaming_sigma={self.streaming_update_sigma}, "
            f"num_updates={self.num_updates})"
        )

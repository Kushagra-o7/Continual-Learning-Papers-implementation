"""
Deep Streaming Linear Discriminant Analysis (Deep SLDA)
========================================================
Faithful implementation of:

    Hayes, T. L., & Kanan, C. (2020).
    Lifelong Machine Learning with Deep Streaming Linear Discriminant Analysis.
    CVPR Workshops (CLVision), 2020.

    Paper: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w15/
           Hayes_Lifelong_Machine_Learning_With_Deep_Streaming_Linear_Discriminant_Analysis_CVPRW_2020_paper.pdf

Dataset in paper: ImageNet-1000 (ILSVRC 2012). Only dataset used in all paper experiments.

Algorithm
---------
The classifier maintains three sufficient statistics:

  muK  (C x d)  -- per-class feature mean vectors
  cK   (C,)     -- per-class sample counts
  Sigma (d x d) -- shared (tied) covariance matrix; updated online or frozen

Streaming update for one sample (x, y):
    # covariance (Welford-style, paper Eq.):
    delta  = (x - muK[y])^T (x - muK[y]) * n / (n + 1)
    Sigma  = (n * Sigma + delta) / (n + 1)
    # class mean (online mean formula):
    muK[y] += (x - muK[y]) / (cK[y] + 1)
    cK[y]  += 1

Prediction (LDA discriminant rule, paper Section 3):
    Lambda = [(1 - eps) * Sigma + eps * I]^{-1}   # NOT pseudo-inverse; paper uses exact inverse
    W      = Lambda @ muK.T                         # (d, C)
    c      = 0.5 * diag(muK @ W)                   # bias  (C,)
    score(x) = x @ W - c                           # (N, C)

Base initialization (first increment only):
    Class means: exact mean over base batch per class.
    Sigma: Oracle Approximating Shrinkage (OAS) from sklearn on mean-centred features.
    This matches the paper's "offline base CNN initialization procedure."
"""

import os
import torch
import torch.nn as nn


class StreamingLDA(nn.Module):
    """
    Streaming Linear Discriminant Analysis classifier.

    Parameters
    ----------
    feature_dim : int
        Dimensionality d of input feature vectors.
        Paper uses 512 (ResNet-18 global average pooling output).
    num_classes : int
        Total classes C known in advance.
        Paper: 1000 (ImageNet).
    shrinkage_param : float
        Regularisation epsilon in Lambda = [(1-eps)*Sigma + eps*I]^{-1}.
        Paper default: 1e-4.
    streaming_update_sigma : bool
        True  -> update Sigma online one sample at a time (main paper result).
        False -> freeze Sigma after OAS base-init (ablation in paper Table 2).
    test_batch_size : int
        Mini-batch size for predict() to avoid GPU OOM on large test sets.
    device : str or None
        Compute device. Auto-detected if None.
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

        # ------------------------------------------------------------------
        # Sufficient statistics
        # ------------------------------------------------------------------
        # muK  : per-class mean vectors  (C, d)
        # cK   : per-class counts         (C,)
        # Sigma: shared covariance matrix (d, d)
        # Lambda: cached precision matrix (d, d) -- recomputed lazily
        # ------------------------------------------------------------------
        self.muK = torch.zeros(num_classes, feature_dim, device=device)
        self.cK = torch.zeros(num_classes, device=device)
        # Init to identity so predictions are valid before first update
        self.Sigma = torch.eye(feature_dim, device=device)
        self.num_updates = 0

        # Precision matrix cache -- recomputed only when Sigma changes
        self.Lambda = torch.zeros(feature_dim, feature_dim, device=device)
        self._prev_num_updates = -1  # sentinel for cache invalidation

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    @torch.no_grad()
    def fit(self, x: torch.Tensor, y: torch.Tensor) -> None:
        """
        Update the classifier with a single new (feature, label) pair.

        This implements the paper's streaming update equations exactly.
        Call once per sample in arrival order.

        Parameters
        ----------
        x : Tensor shape (d,) or (1, d)
        y : scalar Tensor or shape (1,)
        """
        x = x.to(self.device, dtype=torch.float32)
        y = y.long().to(self.device)

        if x.dim() < 2:
            x = x.unsqueeze(0)   # (1, d)
        if y.dim() == 0:
            y = y.unsqueeze(0)   # (1,)

        # ------------------------------------------------------------------
        # Streaming covariance update (Welford-style one-pass estimator)
        # Paper: delta = (x - mu_k)^T (x - mu_k) * n / (n+1)
        #        Sigma = (n * Sigma + delta) / (n+1)
        # NOTE: update uses OLD muK[y] (before mean update below).
        # ------------------------------------------------------------------
        if self.streaming_update_sigma:
            x_minus_mu = x - self.muK[y]               # (1, d)
            mult = x_minus_mu.t().mm(x_minus_mu)        # (d, d)
            n = self.num_updates
            delta = mult * n / (n + 1)
            self.Sigma = (n * self.Sigma + delta) / (n + 1)

        # ------------------------------------------------------------------
        # Class-mean update (online mean formula)
        # muK[y] += (x - muK[y]) / (cK[y] + 1)
        # ------------------------------------------------------------------
        self.muK[y, :] += (x - self.muK[y, :]) / (self.cK[y] + 1).unsqueeze(1)
        self.cK[y] += 1
        self.num_updates += 1

    @torch.no_grad()
    def fit_batch(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Stream a batch of (feature, label) pairs one sample at a time.

        Identical to calling fit() in a loop -- this is the correct protocol.
        The paper fits one sample at a time; batching is NOT done in the SLDA
        update itself.

        Parameters
        ----------
        X : Tensor (N, d)
        y : Tensor (N,) or (N, 1)
        """
        X = X.to(self.device, dtype=torch.float32)
        y = y.view(-1).long().to(self.device)
        for xi, yi in zip(X, y):
            self.fit(xi, yi.view(1))

    @torch.no_grad()
    def fit_base(self, X: torch.Tensor, y: torch.Tensor) -> None:
        """
        Base-class initialisation using the full base batch (first increment only).

        This matches the paper's "offline base CNN initialization procedure":
          - Class means computed exactly from the full base batch.
          - Sigma estimated via Oracle Approximating Shrinkage (OAS) from sklearn
            on the mean-centred features of all base samples.

        Parameters
        ----------
        X : Tensor (N, d)  -- all features for base classes
        y : Tensor (N,)    -- corresponding integer labels
        """
        X = X.to(self.device, dtype=torch.float32)
        y = y.squeeze().long().to(self.device)

        print("[SLDA] fit_base: computing exact class means ...")
        for k in torch.unique(y):
            mask = y == k
            self.muK[k] = X[mask].mean(dim=0)
            self.cK[k] = mask.sum().float()
        self.num_updates = X.shape[0]

        print("[SLDA] fit_base: estimating covariance via OAS ...")
        from sklearn.covariance import OAS
        X_centered = (X - self.muK[y]).cpu().numpy()
        oas = OAS(assume_centered=True)
        oas.fit(X_centered)
        self.Sigma = torch.from_numpy(oas.covariance_).float().to(self.device)
        print("[SLDA] fit_base: done.")

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict(
        self, X: torch.Tensor, return_probas: bool = False
    ) -> torch.Tensor:
        """
        LDA discriminant scores (or softmax probabilities).

        Paper formula (Section 3):
            Lambda = [(1 - eps) * Sigma + eps * I]^{-1}
            W      = Lambda @ muK.T                  (d, C)
            c      = 0.5 * diag(muK @ W)             (C,)
            score  = X @ W - c                       (N, C)

        NOTE: We use torch.linalg.inv (exact inverse) because
        (1-eps)*Sigma + eps*I is positive definite by construction (eps > 0),
        so a pseudo-inverse (pinv) is unnecessary and not what the paper does.

        Parameters
        ----------
        X            : Tensor (N, d)
        return_probas: bool -- if True return softmax(scores) else raw scores

        Returns
        -------
        Tensor (N, C) on CPU.
        """
        X = X.to(self.device, dtype=torch.float32)
        num_samples = X.shape[0]

        # Recompute Lambda only when Sigma has changed
        if self._prev_num_updates != self.num_updates:
            print("[SLDA] Recomputing precision matrix Lambda ...")
            eps = self.shrinkage_param
            reg_sigma = (1.0 - eps) * self.Sigma + eps * torch.eye(
                self.feature_dim, device=self.device
            )
            # Paper: Lambda = [(1-eps)*Sigma + eps*I]^{-1}  (exact inverse)
            self.Lambda = torch.linalg.inv(reg_sigma)
            self._prev_num_updates = self.num_updates

        # Shared weight matrix and bias
        M = self.muK.t()             # (d, C)
        W = self.Lambda @ M          # (d, C)
        c = 0.5 * (M * W).sum(dim=0) # (C,)

        # Mini-batch inference to avoid OOM
        mb = min(self.test_batch_size, num_samples)
        scores = torch.empty(num_samples, self.num_classes, device="cpu")
        for start in range(0, num_samples, mb):
            end = min(start + mb, num_samples)
            scores[start:end] = (X[start:end] @ W - c).cpu()

        if return_probas:
            return torch.softmax(scores, dim=1)
        return scores

    # ------------------------------------------------------------------
    # Checkpoint save / load
    # ------------------------------------------------------------------

    def save_model(self, save_path: str, save_name: str) -> None:
        """
        Persist sufficient statistics + hyperparameters to a .pth file.
        Saves: muK, cK, Sigma, num_updates + all constructor hyperparameters.
        """
        os.makedirs(save_path, exist_ok=True)
        ckpt = {
            "muK": self.muK.cpu(),
            "cK": self.cK.cpu(),
            "Sigma": self.Sigma.cpu(),
            "num_updates": self.num_updates,
            # Hyperparameters needed to reconstruct
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
            "shrinkage_param": self.shrinkage_param,
            "streaming_update_sigma": self.streaming_update_sigma,
        }
        path = os.path.join(save_path, save_name + ".pth")
        torch.save(ckpt, path)
        print(f"[SLDA] Checkpoint saved: {path}")

    def load_model(self, save_path: str, save_name: str) -> None:
        """
        Load sufficient statistics from a .pth checkpoint into this instance.
        """
        path = os.path.join(save_path, save_name + ".pth")
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.muK = ckpt["muK"].to(self.device)
        self.cK = ckpt["cK"].to(self.device)
        self.Sigma = ckpt["Sigma"].to(self.device)
        self.num_updates = ckpt["num_updates"]
        self._prev_num_updates = -1  # force Lambda recompute on next predict
        print(f"[SLDA] Checkpoint loaded: {path}")

    @classmethod
    def from_checkpoint(cls, save_path: str, save_name: str, device: str = None):
        """
        Construct a StreamingLDA directly from a checkpoint file.
        """
        path = os.path.join(save_path, save_name + ".pth")
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        instance = cls(
            feature_dim=ckpt["feature_dim"],
            num_classes=ckpt["num_classes"],
            shrinkage_param=ckpt["shrinkage_param"],
            streaming_update_sigma=ckpt["streaming_update_sigma"],
            device=device,
        )
        instance.load_model(save_path, save_name)
        return instance

    def __repr__(self) -> str:
        return (
            f"StreamingLDA(feature_dim={self.feature_dim}, "
            f"num_classes={self.num_classes}, "
            f"eps={self.shrinkage_param}, "
            f"streaming_sigma={self.streaming_update_sigma}, "
            f"num_updates={self.num_updates})"
        )

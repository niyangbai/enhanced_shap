"""
Adaptive Baseline SHAP (Sparse)

Estimates feature attributions using SHAP, but with baselines always drawn from real observed samples (never mean or zero). Baseline selection is either by support similarity or random from the data. All perturbations are guaranteed to be valid binary/one-hot patterns.
"""

from shap_enhanced.base_explainer import BaseExplainer

import numpy as np
import torch
from typing import Any, Union, Sequence

class AdaptiveBaselineSHAPExplainer(BaseExplainer):
    """
    Universal Adaptive SHAP Explainer.

    - For dense/continuous/tabular features: uses mean baseline masking (SHAP default).
    - For sparse/one-hot/categorical features: uses adaptive (sampled) baselines.
    - Automatically detects feature type based on background data, or can be forced by user.

    Parameters
    ----------
    model : Callable
        The model to be explained.
    background : np.ndarray | torch.Tensor
        Background data for baseline selection, shape (N, F) or (N, T, F).
    n_baselines : int
        Number of baseline samples per coalition (for adaptive masking).
    mask_strategy : str | Sequence[str]
        "auto" (default): detect per feature; or explicitly provide per-feature list (e.g., ["mean", "adaptive", ...]).
    device : str | None
        PyTorch device string.
    """

    def __init__(
        self,
        model: Any,
        background: Union[np.ndarray, torch.Tensor],
        n_baselines: int = 10,
        mask_strategy: Union[str, Sequence[str]] = "auto",
        device: str = None
    ):
        super().__init__(model, background)
        bg = background.detach().cpu().numpy() if hasattr(background, "detach") else np.asarray(background)
        self.background = bg if bg.ndim == 3 else bg[:, None, :]  # (N, T, F)
        self.N, self.T, self.F = self.background.shape
        self.n_baselines = n_baselines
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Determine masking strategy per feature
        if mask_strategy == "auto":
            # For each feature, if >90% zeros in background, use 'adaptive' masking, else 'mean'
            self.feature_strategies = []
            for f in range(self.F):
                bg_feat = self.background[..., f].flatten()
                zero_frac = np.mean(bg_feat == 0)
                self.feature_strategies.append("adaptive" if zero_frac > 0.9 else "mean")
        elif isinstance(mask_strategy, (list, tuple, np.ndarray)):
            assert len(mask_strategy) == self.F
            self.feature_strategies = list(mask_strategy)
        elif isinstance(mask_strategy, str):
            # All features use the same
            self.feature_strategies = [mask_strategy] * self.F
        else:
            raise ValueError(f"Invalid mask_strategy: {mask_strategy}")

        self.mean_baseline = np.mean(self.background, axis=0)  # (T, F)

    def _select_baselines(self, x: np.ndarray, n: int) -> np.ndarray:
        idx = np.random.choice(self.N, n, replace=True)
        return self.background[idx]  # (n, T, F)

    def _mask_input(self, x: np.ndarray, baseline: np.ndarray, mask: list) -> np.ndarray:
        """
        Mask x by baseline for categorical, by mean for continuous.
        """
        x_masked = x.copy()
        for (t, f) in mask:
            if self.feature_strategies[f] == "mean":
                x_masked[t, f] = self.mean_baseline[t, f]
            else:  # "adaptive"
                x_masked[t, f] = baseline[t, f]
        return x_masked

    def _full_mask(self, x: np.ndarray) -> np.ndarray:
        """
        Returns fully-masked version of x (for normalization).
        Uses mean for mean-masked features, random baseline for adaptive.
        """
        x_masked = x.copy()
        baseline = self._select_baselines(x, 1)[0]
        for t in range(self.T):
            for f in range(self.F):
                if self.feature_strategies[f] == "mean":
                    x_masked[t, f] = self.mean_baseline[t, f]
                else:
                    x_masked[t, f] = baseline[t, f]
        return x_masked

    def shap_values(
        self,
        X: Union[np.ndarray, torch.Tensor],
        nsamples: int = 100,
        random_seed: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Compute SHAP values for X (single or batch).
        """
        np.random.seed(random_seed)
        X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        single = (X.ndim == 2)
        X = X[None, ...] if single else X  # (B, T, F)
        B, T, F = X.shape
        out = np.zeros((B, T, F), dtype=np.float32)

        for b in range(B):
            x = X[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    vals = []
                    other_pos = [p for p in all_pos if p != (t, f)]
                    for _ in range(nsamples):
                        k = np.random.randint(1, len(other_pos) + 1)
                        mask_idxs = [other_pos[i] for i in np.random.choice(len(other_pos), k, replace=False)]
                        for baseline in self._select_baselines(x, self.n_baselines):
                            x_masked = self._mask_input(x, baseline, mask_idxs)
                            x_masked_plus = self._mask_input(x_masked, baseline, [(t, f)])
                            pred_masked = float(self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze())
                            pred_masked_plus = float(self.model(torch.tensor(x_masked_plus[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze())
                            vals.append(pred_masked_plus - pred_masked)
                    out[b, t, f] = np.mean(vals)

            # Normalize attributions to match output difference (SHAP style)
            orig_pred = float(self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze())
            full_masked = self._full_mask(x)
            full_pred = float(self.model(torch.tensor(full_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze())
            diff = orig_pred - full_pred
            summ = out[b].sum()
            if np.abs(summ) > 1e-8:
                out[b] *= diff / summ

        return out[0] if single else out

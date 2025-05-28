"""
Temporal SHAP (TS-SHAP) Explainer for Sequential Models.

Implements TS-SHAP: Shapley value estimation for time series models using context-aware masking.
Reference: https://arxiv.org/abs/2106.05303
"""

from typing import Any, Optional, List
import torch
import random
import numpy as np
from scipy.signal import savgol_filter

from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_subsets


class TSShapExplainer(BaseExplainer):
    """Temporal SHAP (TS-SHAP) Explainer.

    Estimates timestep-level Shapley values via masked perturbation and background-based imputation.

    .. math::

        \\phi_t = \\mathbb{E}_{S \\subseteq T \\setminus \\{t\\}} \\left[ f(x_{S \\cup \\{t\\}}) - f(x_S) \\right]
    """

    def __init__(
        self,
        model: Any,
        background: torch.Tensor,
        nsamples: int = 100,
        window_size: int = 1,
        random_seed: Optional[int] = None,
        target_index: Optional[int] = 0,
        mask_strategy: str = "background",
        smoothing: bool = True,
    ) -> None:
        """
        :param model: Model taking [batch, time, features] input.
        :param background: Background data for imputation [N, time, features].
        :param nsamples: Number of Shapley subset samples.
        :param window_size: Size of window to perturb (currently not used).
        :param random_seed: Optional seed for reproducibility.
        :param target_index: Output index to explain.
        :param mask_strategy: "background", "mean", or "interp".
        :param smoothing: Whether to apply Savitzky–Golay filter to reduce noise.
        """
        super().__init__(model)
        self.background = background
        self.nsamples = nsamples
        self.window_size = window_size
        self.random_seed = random_seed
        self.target_index = target_index
        self.mask_strategy = mask_strategy.lower()
        self.smoothing = smoothing

        if self.random_seed is not None:
            random.seed(self.random_seed)
            torch.manual_seed(self.random_seed)

        if self.mask_strategy not in ["background", "mean", "interp"]:
            raise ValueError("Invalid mask_strategy. Choose 'background', 'mean', or 'interp'.")

        if self.mask_strategy == "mean":
            self.mean_background = background.mean(dim=0, keepdim=True)

    def explain(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate TS-SHAP values for each timestep.

        :param X: Input batch [B, T, F].
        :return: Shapley attributions [B, T].
        """
        if X.shape[1:] != self.background.shape[1:]:
            raise ValueError("Input and background must have same [T, F] shape.")

        batch_size, time_steps, _ = X.shape
        attributions = torch.zeros((batch_size, time_steps), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]
            shap_vals = self._ts_shap(x, time_steps)
            if self.smoothing:
                shap_vals = savgol_filter(shap_vals, window_length=5, polyorder=2, mode="nearest")
            attributions[i] = torch.tensor(shap_vals, device=X.device)

        return self._format_output(attributions)

    def _ts_shap(self, x: torch.Tensor, time_steps: int) -> np.ndarray:
        """TS-SHAP sequential perturbation using masking."""
        subsets = sample_subsets(time_steps, self.nsamples)
        shap_values = torch.zeros(time_steps, device=x.device)

        for subset in subsets:
            S = list(subset)
            x_S = self._mask_timesteps(x, [t for t in range(time_steps) if t not in S])
            out_S = self._predict(x_S)

            for t in range(time_steps):
                if t in S:
                    continue
                S_t = S + [t]
                x_S_t = self._mask_timesteps(x, [tt for tt in range(time_steps) if tt not in S_t])
                out_S_t = self._predict(x_S_t)
                shap_values[t] += (out_S_t - out_S).squeeze().item()

        return (shap_values / self.nsamples).cpu().numpy()

    def _mask_timesteps(self, x: torch.Tensor, mask_indices: List[int]) -> torch.Tensor:
        """Apply masking strategy to selected time steps."""
        x_masked = x.clone()
        T = x.size(1)
        for t in mask_indices:
            if self.mask_strategy == "background":
                idx = torch.randint(0, self.background.shape[0], (1,), device=x.device)
                x_masked[:, t, :] = self.background[idx, t, :]
            elif self.mask_strategy == "mean":
                x_masked[:, t, :] = self.mean_background.to(x.device)[:, t, :]
            elif self.mask_strategy == "interp":
                if 0 < t < T - 1:
                    x_masked[:, t, :] = 0.5 * (x[:, t - 1, :] + x[:, t + 1, :])
                else:
                    x_masked[:, t, :] = 0.0
        return x_masked

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run model and extract scalar output."""
        output = self.model(x)
        if output.ndim == 2:
            return output[:, self.target_index or 0]
        elif output.ndim == 1:
            return output.unsqueeze(1)
        else:
            raise ValueError(f"Unexpected model output shape: {output.shape}")

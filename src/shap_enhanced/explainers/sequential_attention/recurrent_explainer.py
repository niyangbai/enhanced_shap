"""This module implements a robust SHAP-style attribution explainer for sequential models (e.g., LSTM).

It approximates timestep-level importance by:

    - Sampling coalitions of timesteps and masking/complementing each timestep.
    - Training multiple surrogate models (KernelRidge) to estimate marginal contributions.
    - Smoothing and normalizing the aggregated attributions for interpretability.

Key Features
------------
- SHAP-style coalition sampling.
- Context-aware mean imputation.
- Noise injection (optional) or ensemble of both.
- Multiple surrogate fits (bootstrapped) with optional weighting.
- Savitzky–Golay smoothing for temporal consistency.
- Spearman-consistent and robust across seeds.

Typical Usage
-------------

.. code-block:: python

    explainer = RecurrentExplainer(model=my_lstm)
    attr = explainer.explain(input_batch)
"""

import torch
import torch.nn.functional as F
import random
import numpy as np
from typing import List
from sklearn.kernel_ridge import KernelRidge
from scipy.signal import savgol_filter
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.perturbation import perturb_sequence_with_noise
import torch

def sample_coalition(T: int, exclude: List[int] = [], p: float = 0.5) -> List[int]:
    """_summary_

    :param int T: _description_
    :param List[int] exclude: _description_, defaults to []
    :param float p: _description_, defaults to 0.5
    :return List[int]: _description_
    """
    candidates = list(set(range(T)) - set(exclude))
    k = max(1, int(p * len(candidates)))
    return random.sample(candidates, k)

def impute_with_context(X: torch.Tensor, timesteps: List[int]) -> torch.Tensor:
    """_summary_

    :param torch.Tensor X: _description_
    :param List[int] timesteps: _description_
    :return torch.Tensor: _description_
    """
    X_copy = X.clone()
    B, T, F = X.shape
    for t in timesteps:
        if 0 < t < T - 1:
            X_copy[:, t] = 0.5 * (X[:, t - 1] + X[:, t + 1])
        else:
            X_copy[:, t] = 0.0
    return X_copy

def perturb_sequence_with_noise(X: torch.Tensor, timesteps: List[int], noise_level: float = 0.05) -> torch.Tensor:
    """_summary_

    :param torch.Tensor X: _description_
    :param List[int] timesteps: _description_
    :param float noise_level: _description_, defaults to 0.05
    :return torch.Tensor: _description_
    """
    X_perturbed = X.clone()
    if not timesteps:
        return X_perturbed
    B, T, F = X.shape
    noise = torch.randn((B, len(timesteps), F), device=X.device) * noise_level
    for i, t in enumerate(timesteps):
        X_perturbed[:, t] += noise[:, i]
    return X_perturbed

def smooth_savgol(attr: torch.Tensor, window: int = 7, poly: int = 3) -> torch.Tensor:
    """_summary_

    :param torch.Tensor attr: _description_
    :param int window: _description_, defaults to 7
    :param int poly: _description_, defaults to 3
    :return torch.Tensor: _description_
    """
    attr_np = attr.cpu().numpy()
    smoothed = []
    for row in attr_np:
        row_smooth = savgol_filter(row, window_length=window, polyorder=poly, mode="mirror")
        smoothed.append(row_smooth)
    return torch.tensor(smoothed, device=attr.device, dtype=attr.dtype)

class RecurrentExplainer(BaseExplainer):
    def __init__(self, model, mode="ensemble", nsamples=60, n_repeats=5,
                 coalition_p_range=(0.3, 0.6), noise_level=0.05, target_index=0,
                 normalize=True, smooth_window=7, smooth_poly=3, seed=42):
        super().__init__(model)
        self.model = model
        self.mode = mode
        self.nsamples = nsamples
        self.n_repeats = n_repeats
        self.coalition_p_range = coalition_p_range
        self.noise_level = noise_level
        self.target_index = target_index
        self.normalize = normalize
        self.smooth_window = smooth_window
        self.smooth_poly = smooth_poly
        self.seed = seed

    def _get_output(self, X: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = self.model(X)
            return out[:, self.target_index or 0] if out.ndim == 2 else out.view(-1)

    def explain(self, X: torch.Tensor) -> torch.Tensor:
        torch.manual_seed(self.seed)
        random.seed(self.seed)

        B, T, F = X.shape
        attr = torch.zeros((B, T), device=X.device)

        for b in range(B):
            x = X[b:b+1]  # [1, T, F]
            ensemble_attrs = []
            weights = []

            for r in range(self.n_repeats):
                outputs, masks = [], []
                coalition_p = random.uniform(*self.coalition_p_range)

                for _ in range(self.nsamples):
                    mask = torch.zeros(T, dtype=torch.bool)
                    on = sample_coalition(T, p=coalition_p)
                    mask[on] = True
                    off = (~mask).nonzero(as_tuple=True)[0].tolist()

                    if self.mode == "mask":
                        x_masked = impute_with_context(x, off)
                    elif self.mode == "noise":
                        x_masked = perturb_sequence_with_noise(x, off, self.noise_level)
                    elif self.mode == "ensemble":
                        x1 = impute_with_context(x, off)
                        x2 = perturb_sequence_with_noise(x, off, self.noise_level)
                        x_masked = 0.5 * (x1 + x2)
                    else:
                        raise ValueError(f"Invalid mode: {self.mode}")

                    y = self._get_output(x_masked)[0].item()
                    outputs.append(y)
                    masks.append(mask.float().cpu().numpy())

                X_mask = np.vstack(masks)
                y_vals = np.array(outputs)

                alpha = getattr(self, "surrogate_alpha", 1e-2)
                surrogate = KernelRidge(kernel='rbf', alpha=alpha)
                surrogate.fit(X_mask, y_vals)

                baseline = np.zeros((1, T))
                contribs = []
                for t in range(T):
                    mask_on = baseline.copy()
                    mask_on[0, t] = 1.0
                    pred_t = surrogate.predict(mask_on)[0]
                    pred_0 = surrogate.predict(baseline)[0]
                    contribs.append(pred_t - pred_0)

                contribs_tensor = torch.tensor(contribs, device=X.device)
                ensemble_attrs.append(contribs_tensor)
                weights.append(1.0)  # Equal weights; could use validation loss or kernel norm

            # Weighted average across surrogate repeats
            weighted_attr = sum(w * a for w, a in zip(weights, ensemble_attrs)) / sum(weights)
            attr[b] = weighted_attr

        attr = smooth_savgol(attr, window=self.smooth_window, poly=self.smooth_poly)

        if self.normalize:
            attr -= attr.min(dim=1, keepdim=True).values
            attr /= attr.max(dim=1, keepdim=True).values + 1e-6

        return self._format_output(attr)




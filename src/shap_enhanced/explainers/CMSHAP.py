"""
Contextual Masking SHAP (CM-SHAP) Explainer for Sequential Models
=================================================================

Theoretical Explanation
-----------------------

CM-SHAP is a SHAP-style feature attribution method designed for sequential (time series) models.
Instead of masking features with zeros or mean values, CM-SHAP uses context-aware imputation:
masked values are replaced by interpolating between their immediate temporal neighbors (forward/backward average).
This preserves the temporal structure and context, leading to more realistic and faithful perturbations for sequential data.

Key Concepts
^^^^^^^^^^^^

- **Contextual Masking**: When masking a feature at time ``t``, its value is replaced by the average of its neighboring time steps (``t-1`` and ``t+1``). For boundary cases (first or last time step), only the available neighbor is used.
- **Coalition Sampling**: For each feature-time position ``(t, f)``, random coalitions (subsets of all other positions) are sampled. The marginal contribution of ``(t, f)`` is estimated by measuring the change in model output when ``(t, f)`` is added to the coalition.
- **Additivity Normalization**: Attributions are normalized so their sum matches the model output difference between the original and fully-masked input.

Algorithm
---------

1. **Initialization**:
    - Accepts a model and device.

2. **Contextual Masking**:
    - For each coalition (subset of features to mask), masked positions are replaced by the average of their immediate temporal neighbors.

3. **SHAP Value Estimation**:
    - For each feature-time position ``(t, f)``, repeatedly:
        - Sample a random coalition of other positions.
        - Mask the coalition using contextual interpolation.
        - Mask the coalition plus ``(t, f)`` using contextual interpolation.
        - Compute the model output difference.
        - Average these differences to estimate the marginal contribution of ``(t, f)``.

    - Normalize attributions so their sum matches the difference between the original and fully-masked model output.
"""

from typing import Any

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class ContextualMaskingSHAPExplainer(BaseExplainer):
    r"""
    Contextual Masking SHAP (CM-SHAP) Explainer for Sequential Models

    Estimates SHAP values for sequential inputs by replacing masked feature values with
    interpolated values from neighboring time steps. This context-aware masking strategy
    preserves temporal coherence and enables more realistic feature perturbation in time-series data.

    :param model: Model to explain. Must accept NumPy arrays or PyTorch tensors.
    :type model: Any
    :param device: Device to perform computations on ('cpu' or 'cuda'). Defaults to 'cuda' if available.
    :type device: Optional[str]
    """

    def __init__(self, model: Any, device: str | None = None):
        super().__init__(model, background=None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _interpolate_mask(X, idxs):
        """
        Apply contextual interpolation to mask specified time-feature pairs.

        Replaces each selected feature at time `t` with the average of its adjacent
        time steps (t-1 and t+1). Handles edge cases by copying the available neighbor.

        :param X: Input of shape (T, F)
        :type X: np.ndarray or torch.Tensor
        :param idxs: List of (t, f) index pairs to interpolate.
        :type idxs: list[tuple[int, int]]
        :return: Interpolated input with same shape as X.
        :rtype: Same as input type
        """
        X_interp = X.copy() if isinstance(X, np.ndarray) else X.clone()
        T, F = X_interp.shape
        for t, f in idxs:
            if t == 0:
                X_interp[t, f] = X_interp[t + 1, f]
            elif t == T - 1:
                X_interp[t, f] = X_interp[t - 1, f]
            else:
                X_interp[t, f] = 0.5 * (X_interp[t - 1, f] + X_interp[t + 1, f])
        return X_interp

    def _get_model_output(self, X):
        """
        Forward-pass utility to handle input conversion and model inference.

        Ensures that the input is on the correct device and returned as a NumPy array.

        :param X: Input array or tensor (T, F) or (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :return: Model output in NumPy format.
        :rtype: np.ndarray or float
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        elif isinstance(X, torch.Tensor):
            X = X.to(self.device)
        else:
            raise ValueError("Input must be np.ndarray or torch.Tensor.")

        with torch.no_grad():
            out = self.model(X)
            return out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)

    def shap_values(
        self,
        X: np.ndarray | torch.Tensor,
        nsamples: int = 100,
        check_additivity: bool = True,
        random_seed: int = 42,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Estimate SHAP values using contextual (interpolated) masking.

        Each feature-time pair (t, f) is evaluated by sampling coalitions of other
        positions, applying context-aware masking, and averaging the difference
        in model outputs when (t, f) is added to the coalition.

        Interpolation strategy ensures continuity in time series by replacing masked
        values with averages of adjacent time steps:

        .. math::
            x_{t,f}^{masked} =
            \begin{cases}
                x_{t+1,f}, & \text{if } t = 0 \\
                x_{t-1,f}, & \text{if } t = T-1 \\
                \frac{x_{t-1,f} + x_{t+1,f}}{2}, & \text{otherwise}
            \end{cases}

        Final attributions are normalized such that:

        .. math::
            \sum_{t=1}^T \sum_{f=1}^F \phi_{t,f} \approx f(x) - f(x_{masked})

        :param X: Input array of shape (T, F) or (B, T, F)
        :type X: np.ndarray or torch.Tensor
        :param nsamples: Number of sampled coalitions per position.
        :type nsamples: int
        :param check_additivity: Whether to normalize SHAP values to match output difference.
        :type check_additivity: bool
        :param random_seed: Random seed for reproducibility.
        :type random_seed: int
        :return: SHAP values with same shape as input.
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)

        is_torch = isinstance(X, torch.Tensor)
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        shape = X_in.shape
        if len(shape) == 2:  # (T, F)
            X_in = X_in[None, ...]
        B, T, F = X_in.shape

        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_in[b]
            for t in range(T):
                for f in range(F):
                    contribs = []
                    all_pos = [
                        (i, j) for i in range(T) for j in range(F) if (i, j) != (t, f)
                    ]
                    for _ in range(nsamples):
                        # Sample random coalition
                        k = np.random.randint(1, len(all_pos) + 1)
                        C_idxs = np.random.choice(len(all_pos), size=k, replace=False)
                        C_idxs = [all_pos[idx] for idx in C_idxs]

                        # Mask coalition C (using interpolation)
                        x_C = self._interpolate_mask(x_orig, C_idxs)
                        # Mask coalition plus (t, f)
                        x_C_tf = self._interpolate_mask(x_C, [(t, f)])

                        out_C = self._get_model_output(x_C[None])[0]
                        out_C_tf = self._get_model_output(x_C_tf[None])[0]

                        contribs.append(out_C_tf - out_C)
                    shap_vals[b, t, f] = np.mean(contribs)

            # Additivity normalization
            orig_pred = self._get_model_output(x_orig[None])[0]
            x_all_masked = self._interpolate_mask(
                x_orig, [(ti, fi) for ti in range(T) for fi in range(F)]
            )
            masked_pred = self._get_model_output(x_all_masked[None])[0]
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if len(shape) == 2 else shap_vals

        if check_additivity:
            print(
                f"[CM-SHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}"
            )

        return shap_vals

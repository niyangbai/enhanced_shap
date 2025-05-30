"""
Contextual Masking SHAP (CM-SHAP) Explainer for Sequential Models

## Theoretical Explanation

CM-SHAP is a SHAP-style feature attribution method designed for sequential (time series) models. Instead of masking features with zeros or mean values, CM-SHAP uses context-aware imputation: masked values are replaced by interpolating between their immediate temporal neighbors (forward/backward average). This preserves the temporal structure and context, leading to more realistic and faithful perturbations for sequential data.

### Key Concepts

- **Contextual Masking:** When masking a feature at time t, its value is replaced by the average of its neighboring time steps (t-1 and t+1). For boundary cases (first or last time step), only the available neighbor is used.
- **Coalition Sampling:** For each feature-time position (t, f), random coalitions (subsets of all other positions) are sampled. The marginal contribution of (t, f) is estimated by measuring the change in model output when (t, f) is added to the coalition.
- **Additivity Normalization:** Attributions are normalized so their sum matches the model output difference between the original and fully-masked input.

## Algorithm

1. **Initialization:**
    - Accepts a model and device.
2. **Contextual Masking:**
    - For each coalition (subset of features to mask), masked positions are replaced by the average of their immediate temporal neighbors.
3. **SHAP Value Estimation:**
    - For each feature-time position (t, f), repeatedly:
        - Sample a random coalition of other positions.
        - Mask the coalition using contextual interpolation.
        - Mask the coalition plus (t, f) using contextual interpolation.
        - Compute the model output difference.
        - Average these differences to estimate the marginal contribution of (t, f).
    - Normalize attributions so their sum matches the difference between the original and fully-masked model output.
"""

from typing import Any, Optional, Union
import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer

class ContextualMaskingSHAPExplainer(BaseExplainer):
    """
    Contextual Masking SHAP (CM-SHAP) Explainer.

    Estimates Shapley values for sequential models by masking input positions using local interpolation
    (averaging immediate neighbors in time), preserving temporal structure and context.

    Parameters
    ----------
    model : Any
        The model to be explained.
    device : Optional[str]
        'cpu' or 'cuda' (PyTorch only).
    """

    def __init__(self, model: Any, device: Optional[str] = None):
        super().__init__(model, background=None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def _interpolate_mask(X, idxs):
        """
        Interpolates masked positions using the average of neighboring time steps.
        idxs: list of (t, f) pairs to mask.
        """
        X_interp = X.copy() if isinstance(X, np.ndarray) else X.clone()
        T, F = X_interp.shape
        for (t, f) in idxs:
            if t == 0:
                X_interp[t, f] = X_interp[t + 1, f]
            elif t == T - 1:
                X_interp[t, f] = X_interp[t - 1, f]
            else:
                X_interp[t, f] = 0.5 * (X_interp[t - 1, f] + X_interp[t + 1, f])
        return X_interp

    def _get_model_output(self, X):
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
        X: Union[np.ndarray, torch.Tensor],
        nsamples: int = 100,
        check_additivity: bool = True,
        random_seed: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Compute SHAP values by locally interpolating (rather than zero-masking) features.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input sequence(s), shape (T, F) or (B, T, F).
        nsamples : int
            Number of coalitions to sample per position.
        check_additivity : bool
            Whether to rescale for additivity.
        random_seed : int
            Random seed for reproducibility.

        Returns
        -------
        shap_vals : np.ndarray
            Attributions, shape (T, F) or (B, T, F).
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
                    all_pos = [(i, j) for i in range(T) for j in range(F) if (i, j) != (t, f)]
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
            x_all_masked = self._interpolate_mask(x_orig, [(ti, fi) for ti in range(T) for fi in range(F)])
            masked_pred = self._get_model_output(x_all_masked[None])[0]
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if len(shape) == 2 else shap_vals

        if check_additivity:
            print(f"[CM-SHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")

        return shap_vals

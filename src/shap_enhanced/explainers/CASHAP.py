"""
Coalition-Aware SHAP (CASHAP) Explainer for Sequential Models.

Implements the CASHAP method: Shapley value estimation by sampling coalitions (groups of time-feature pairs) and measuring their marginal contributions to model output.
Supports LSTM/sequence models and generalizes to tabular data with masking or imputation for context-aware explanations.

Algorithm:
----------
For each input position (t, f):
    - Sample random coalitions C ⊆ (T × F) \ {(t, f)}
    - Compute marginal contribution:
        φ_{t, f} ≈ (1/N) Σ_i [f(x_{C ∪ (t, f)}) - f(x_C)]
    - Aggregate over N samples per (t, f)

Supports masking (zero or mean-imputation) and both numpy and PyTorch input.

Warnings:
---------
- For LSTM models, context-aware imputation is recommended for best results.
- This method can be computationally intensive for long sequences or high nsamples.
"""

from typing import Any, Optional, Union
from collections.abc import Callable
import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer

class CoalitionAwareSHAPExplainer(BaseExplainer):
    """
    Coalition-Aware SHAP (CASHAP) Explainer.

    Estimates Shapley values for sequential models by sampling random coalitions of (t, f) pairs,
    masking/imputing them, and measuring the marginal effect of each position on the model output.

    Parameters
    ----------
    model : Any
        The model to be explained. Must support __call__ or .predict with batched input.
    background : Optional[np.ndarray, torch.Tensor]
        Background data for mean imputation.
    mask_strategy : str
        'zero' for zero masking, 'mean' for mean imputation, or 'custom' for a user-supplied imputer.
    imputer : Optional[Callable]
        A callable function to perform imputation. Used only if mask_strategy == 'custom'.
    device : Optional[str]
        'cpu' or 'cuda' (only relevant for PyTorch).
    """

    def __init__(
        self,
        model: Any,
        background: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mask_strategy: str = "zero",
        imputer: Optional[Callable] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, background)
        self.mask_strategy = mask_strategy
        self.imputer = imputer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Precompute mean if needed
        if mask_strategy == "mean":
            if background is None:
                raise ValueError("Mean imputation requires background data.")
            self._mean = (
                background.mean(axis=0) if isinstance(background, np.ndarray)
                else background.float().mean(dim=0)
            )
        else:
            self._mean = None

    def _mask(self, X, idxs, value=None):
        """
        Mask (or impute) positions in X at indices given by idxs.
        idxs: list of (t, f) pairs (or (batch, t, f) if batched)
        value: value to use for masking/imputation
        """
        X_masked = X.copy() if isinstance(X, np.ndarray) else X.clone()
        for (t, f) in idxs:
            if isinstance(X_masked, np.ndarray):
                X_masked[t, f] = value if value is not None else 0.0
            else:
                X_masked[:, t, f] = value if value is not None else 0.0
        return X_masked

    def _impute(self, X, idxs):
        """
        Impute positions in X at indices given by idxs using the selected strategy.
        """
        if self.mask_strategy == "zero":
            return self._mask(X, idxs, value=0.0)
        elif self.mask_strategy == "mean":
            mean_val = (
                self._mean if isinstance(X, np.ndarray)
                else self._mean.unsqueeze(0).expand_as(X)
            )
            X_imp = X.copy() if isinstance(X, np.ndarray) else X.clone()
            for (t, f) in idxs:
                if isinstance(X_imp, np.ndarray):
                    X_imp[t, f] = mean_val[t, f]
                else:
                    X_imp[:, t, f] = mean_val[t, f]
            return X_imp
        elif self.mask_strategy == "custom":
            assert self.imputer is not None, "Custom imputer must be provided."
            return self.imputer(X, idxs)
        else:
            raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")

    def _get_model_output(self, X):
        """
        Ensures model input is always a torch.Tensor on the correct device.
        Accepts (T, F) or (B, T, F), returns numpy array or float.
        """
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        elif isinstance(X, torch.Tensor):
            X = X.to(self.device)
        else:
            raise ValueError("Input must be np.ndarray or torch.Tensor.")

        with torch.no_grad():
            out = self.model(X)
            # Out can be (B,), (B,1), or scalar. Always return numpy
            return out.cpu().numpy() if hasattr(out, "cpu") else np.asarray(out)

    def shap_values(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        nsamples: int = 100,
        coalition_size: Optional[int] = None,
        mask_strategy: Optional[str] = None,
        check_additivity: bool = True,
        random_seed: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Improved CASHAP Shapley estimation.
        """
        np.random.seed(random_seed)
        mask_strategy = mask_strategy or self.mask_strategy

        is_torch = isinstance(X, torch.Tensor)
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        shape = X_in.shape
        if len(shape) == 2:  # (T, F)
            X_in = X_in[None, ...]  # add batch dim
        B, T, F = X_in.shape

        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_in[b]
            for t in range(T):
                for f in range(F):
                    contribs = []
                    all_pos = [(i, j) for i in range(T) for j in range(F) if (i, j) != (t, f)]
                    for _ in range(nsamples):
                        # Improved: Systematic coalition size
                        if coalition_size is not None:
                            k = coalition_size
                        else:
                            k = np.random.randint(1, len(all_pos) + 1)
                        C_idxs = list(np.random.choice(len(all_pos), size=k, replace=False))
                        C_idxs = [all_pos[idx] for idx in C_idxs]

                        # Mask coalition (C) only
                        x_C = self._impute(x_orig, C_idxs)
                        # Mask coalition plus (t, f)
                        x_C_tf = self._impute(x_C, [(t, f)])

                        # Compute outputs
                        out_C = self._get_model_output(x_C[None])[0]
                        out_C_tf = self._get_model_output(x_C_tf[None])[0]

                        contrib = out_C_tf - out_C
                        contribs.append(contrib)
                    shap_vals[b, t, f] = np.mean(contribs)

            # Additivity correction per sample
            orig_pred = self._get_model_output(x_orig[None])[0]
            x_all_masked = self._impute(x_orig, [(ti, fi) for ti in range(T) for fi in range(F)])
            masked_pred = self._get_model_output(x_all_masked[None])[0]
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if len(shape) == 2 else shap_vals

        if check_additivity:
            print(f"[CASHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")

        return shap_vals

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np

    # --- Dummy LSTM model for demo ---
    class DummyLSTM(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=8, output_dim=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Ensure input is float tensor
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.float()
            # x: (B, T, F)
            out, _ = self.lstm(x)
            # Use last time step's output
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)  # (B,)

    # --- Generate synthetic data ---
    np.random.seed(0)
    torch.manual_seed(0)
    B, T, F = 2, 5, 3
    train_X = np.random.normal(0, 1, (20, T, F)).astype(np.float32)
    test_X = np.random.normal(0, 1, (B, T, F)).astype(np.float32)

    # --- Initialize model and explainer ---
    model = DummyLSTM(input_dim=F, hidden_dim=8, output_dim=1)
    model.eval()

    explainer = CoalitionAwareSHAPExplainer(
        model=model,
        background=train_X,
        mask_strategy="mean"
    )

    # --- Compute SHAP values ---
    shap_vals = explainer.shap_values(
        test_X,           # (B, T, F)
        nsamples=10,      # small for demo, increase for quality
        coalition_size=4, # mask 4 pairs at a time
        check_additivity=True
    )

    print("SHAP values shape:", shap_vals.shape)
    print("First sample SHAP values:\n", shap_vals[0])
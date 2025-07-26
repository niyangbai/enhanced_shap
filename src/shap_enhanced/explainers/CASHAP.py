"""
CASHAP: Coalition-Aware SHAP Explainer
======================================

Theoretical Explanation
-----------------------

CASHAP (Coalition-Aware SHAP) is a Shapley value estimation framework tailored for models that process sequential or structured inputs, such as LSTMs.
Unlike classical SHAP methods that treat features independently, CASHAP considers **feature-time pairs**—enabling attribution of both spatial and temporal components.

By explicitly sampling coalitions (subsets) of feature-time pairs and measuring marginal contributions, CASHAP provides granular, context-aware explanations.
It also supports multiple imputation strategies to ensure the perturbed inputs remain valid and interpretable.

Key Concepts
^^^^^^^^^^^^

- **Coalition Sampling**: For every feature-time pair \\((t, f)\\), random subsets of all other positions are sampled.
    The contribution of \\((t, f)\\) is assessed by adding it to each coalition and measuring the change in model output.
- **Masking/Imputation Strategies**:
    - **Zero masking**: Replace masked values with zero.
    - **Mean imputation**: Use feature-wise means from background data.
    - **Custom imputers**: Support for user-defined imputation functions.
- **Model-Agnostic & Domain-General**: While ideal for time-series and sequential models, CASHAP can also be applied to tabular data
    wherever structured coalition masking is appropriate.
- **Additivity Normalization**: Attribution scores are scaled such that their total sum equals the difference in model output
    between the original input and a fully-masked version.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background data for imputation, masking strategy, optional custom imputer, and device context.

2. **Coalition Sampling**:
    - For each feature-time pair \\((t, f)\\):
        - Sample coalitions \\( C \\subseteq (T \times F) \\setminus \\{(t, f)\\} \\).
        - For each coalition \\( C \\):
            - Impute features in \\( C \\) using the chosen strategy.
            - Impute features in \\( C \\cup \\{(t, f)\\} \\).
            - Compute and record the model output difference.

3. **Attribution Estimation**:
    - Average the output differences across coalitions to estimate the marginal contribution of \\((t, f)\\).

4. **Normalization**:
    - Normalize attributions so that their total matches the difference between the model's prediction
        on the original and the fully-masked input.

References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**  
  [SHAP foundation—coalitional feature attribution framework]

- **Jutte et al. (2025), “C‑SHAP for time series: An approach to high‑level temporal explanations”**  
  [Applies concept‑based SHAP to structured temporal data; treats temporal segments or concepts as coalition elements] :contentReference[oaicite:1]{index=1}

- **Schlegel et al. (2019), “Towards a Rigorous Evaluation of XAI Methods on Time Series”**  
  [Evaluates how SHAP and other methods behave for sequential/time‑series models, highlighting temporal structure challenges] :contentReference[oaicite:2]{index=2}

- **Franco de la Peña et al. (2025), “ShaTS: A Shapley‑based Explainability Method for Time Series Models”**  
  [Proposes temporally aware grouping for Shapley attribution in sequential IoT data, preserving temporal dependencies] :contentReference[oaicite:3]{index=3}

- **Molnar, “Interpretable Machine Learning” (2022), SHAP chapter**  
  [Describes masking and coalition sampling strategies, including dealing with structured or dependent features]
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class CoalitionAwareSHAPExplainer(BaseExplainer):
    """
    Coalition-Aware SHAP (CASHAP) Explainer

    Estimates Shapley values for models processing structured inputs (e.g., time-series, sequences)
    by sampling coalitions of feature-time pairs and computing their marginal contributions
    using various imputation strategies.

    :param model: Model to be explained.
    :type model: Any
    :param background: Background data used for mean imputation strategy.
    :type background: Optional[np.ndarray or torch.Tensor]
    :param str mask_strategy: Strategy for imputing/masking feature-time pairs.
                              Options: 'zero', 'mean', or 'custom'.
    :param imputer: Custom callable for imputation. Required if `mask_strategy` is 'custom'.
    :type imputer: Optional[Callable]
    :param device: Device on which computation runs. Defaults to 'cuda' if available.
    :type device: Optional[str]
    """

    def __init__(
        self,
        model: Any,
        background: np.ndarray | torch.Tensor | None = None,
        mask_strategy: str = "zero",
        imputer: Callable | None = None,
        device: str | None = None,
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
                background.mean(axis=0)
                if isinstance(background, np.ndarray)
                else background.float().mean(dim=0)
            )
        else:
            self._mean = None

    def _mask(self, X, idxs, value=None):
        """
        Mask specified feature-time positions in the input.

        :param X: Input array (T, F) or tensor.
        :type X: np.ndarray or torch.Tensor
        :param idxs: List of (t, f) index pairs to mask.
        :type idxs: list[tuple[int, int]]
        :param value: Value to replace at masked positions. Defaults to 0.0.
        :return: Masked version of the input.
        :rtype: Same as input type
        """
        X_masked = X.copy() if isinstance(X, np.ndarray) else X.clone()
        for t, f in idxs:
            if isinstance(X_masked, np.ndarray):
                X_masked[t, f] = value if value is not None else 0.0
            else:
                X_masked[:, t, f] = value if value is not None else 0.0
        return X_masked

    def _impute(self, X, idxs):
        """
        Apply imputation strategy to specified positions.

        Imputation method depends on the selected `mask_strategy`:
        - 'zero': Set masked values to 0.
        - 'mean': Use mean values computed from background data.
        - 'custom': Use user-defined callable function.

        :param X: Input data (T, F).
        :type X: np.ndarray or torch.Tensor
        :param idxs: Positions to impute, as (t, f) tuples.
        :type idxs: list[tuple[int, int]]
        :return: Imputed input.
        :rtype: Same as input type
        """
        if self.mask_strategy == "zero":
            return self._mask(X, idxs, value=0.0)
        elif self.mask_strategy == "mean":
            mean_val = (
                self._mean
                if isinstance(X, np.ndarray)
                else self._mean.unsqueeze(0).expand_as(X)
            )
            X_imp = X.copy() if isinstance(X, np.ndarray) else X.clone()
            for t, f in idxs:
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
        X: np.ndarray | torch.Tensor,
        nsamples: int = 100,
        coalition_size: int | None = None,
        mask_strategy: str | None = None,
        check_additivity: bool = True,
        random_seed: int = 42,
        **kwargs,
    ) -> np.ndarray:
        """
        Compute CASHAP Shapley values for structured inputs via coalition-aware sampling.

        For each feature-time pair \\((t, f)\\), randomly sample coalitions excluding \\((t, f)\\),
        compute model outputs with and without the pair added, and average the marginal contributions.
        Attribution values are normalized so their total matches the model output difference
        between the original and fully-masked input.

        .. math::
            \\phi_{t,f} \approx \\mathbb{E}_{C \\subseteq (T \times F) \\setminus \\{(t,f)\\}} \\left[
                f(C \\cup \\{(t,f)\\}) - f(C)
            \right]

        .. note::
            Normalization ensures:
            \\sum_{t=1}^T \\sum_{f=1}^F \\phi_{t,f} \approx f(x) - f(x_{\text{masked}})

        :param X: Input sample of shape (T, F) or batch (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :param nsamples: Number of coalitions sampled per (t, f).
        :type nsamples: int
        :param coalition_size: Fixed size of sampled coalitions. If None, varies randomly.
        :type coalition_size: Optional[int]
        :param mask_strategy: Override default masking strategy.
        :type mask_strategy: Optional[str]
        :param check_additivity: Print diagnostic SHAP sum vs. model delta.
        :type check_additivity: bool
        :param random_seed: Seed for reproducibility.
        :type random_seed: int
        :return: SHAP values of shape (T, F) or (B, T, F).
        :rtype: np.ndarray
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
                    all_pos = [
                        (i, j) for i in range(T) for j in range(F) if (i, j) != (t, f)
                    ]
                    for _ in range(nsamples):
                        # Improved: Systematic coalition size
                        if coalition_size is not None:
                            k = coalition_size
                        else:
                            k = np.random.randint(1, len(all_pos) + 1)
                        C_idxs = list(
                            np.random.choice(len(all_pos), size=k, replace=False)
                        )
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
            x_all_masked = self._impute(
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
                f"[CASHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}"
            )

        return shap_vals


if __name__ == "__main__":
    import numpy as np
    import torch
    import torch.nn as nn

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
        model=model, background=train_X, mask_strategy="mean"
    )

    # --- Compute SHAP values ---
    shap_vals = explainer.shap_values(
        test_X,  # (B, T, F)
        nsamples=10,  # small for demo, increase for quality
        coalition_size=4,  # mask 4 pairs at a time
        check_additivity=True,
    )

    print("SHAP values shape:", shap_vals.shape)
    print("First sample SHAP values:\n", shap_vals[0])

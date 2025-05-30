"""
ER-SHAP: Ensemble of Random SHAP Explainer

## Theoretical Explanation

ER-SHAP is a fast, ensemble-based approximation of SHAP values for sequential or tabular models. It estimates feature attributions by repeatedly sampling random coalitions (subsets) of feature–timestep pairs, computing their marginal contributions to the model output, and averaging the results. ER-SHAP supports uniform or weighted coalition sampling, allowing prior knowledge (e.g., feature importance) to guide the process.

### Key Concepts

- **Random Coalition Sampling:** For each feature–timestep position (t, f), random coalitions (subsets of all other positions) are sampled. The marginal contribution of (t, f) is estimated by measuring the change in model output when (t, f) is added to the coalition.
- **Weighted Sampling:** Coalitions can be sampled uniformly or with weights based on prior feature importance or frequency.
- **Flexible Masking:** Masked features can be imputed with zeros or mean values from the background data.
- **Additivity Normalization:** Attributions are normalized so their sum matches the model output difference between the original and fully-masked input.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data (for mean imputation), number of coalitions, masking strategy, weighting scheme, optional feature importance, and device.
2. **Coalition Sampling:**
    - For each feature–timestep position (t, f):
        - Sample random coalitions \( C \subseteq (T \times F) \setminus \{(t, f)\} \), with optional weighting.
        - For each coalition:
            - Mask (impute) the coalition \( C \) in the input.
            - Mask (impute) the coalition \( C \cup \{(t, f)\} \).
            - Compute the model output difference.
        - Average these differences to estimate the marginal contribution of (t, f).
3. **Normalization:**
    - Scale attributions so their sum matches the difference between the original and fully-masked model output.
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class ERSHAPExplainer(BaseExplainer):
    """
    ER-SHAP: Random Coalition Ensemble SHAP Explainer

    Parameters
    ----------
    model : Any
        Model to be explained.
    background : np.ndarray or torch.Tensor
        For mean/zero imputation (N, T, F).
    n_coalitions : int
        Number of random coalitions to sample per feature.
    mask_strategy : str
        'mean' or 'zero' (default: 'mean').
    weighting : str
        Coalition weighting: 'uniform', 'frequency', 'importance'.
    feature_importance : np.ndarray, optional
        (T, F) prior importances, used if weighting='importance'.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        background,
        n_coalitions=100,
        mask_strategy="mean",
        weighting="uniform",
        feature_importance=None,
        device=None
    ):
        super().__init__(model, background)
        self.n_coalitions = n_coalitions
        self.mask_strategy = mask_strategy
        self.weighting = weighting
        self.feature_importance = feature_importance
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if mask_strategy == "mean":
            self._mean = background.mean(axis=0)
        else:
            self._mean = None

    def _impute(self, X, idxs):
        X_imp = X.copy()
        for (t, f) in idxs:
            if self.mask_strategy == "zero":
                X_imp[t, f] = 0.0
            elif self.mask_strategy == "mean":
                X_imp[t, f] = self._mean[t, f]
            else:
                raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")
        return X_imp

    def _sample_coalition(self, available, k, weights=None):
        if weights is not None:
            weights = np.array([weights[idx] for idx in available])
            weights = weights / (weights.sum() + 1e-8)
            idxs = np.random.choice(len(available), size=k, replace=False, p=weights)
        else:
            idxs = np.random.choice(len(available), size=k, replace=False)
        return [available[i] for i in idxs]

    def shap_values(
        self,
        X,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        ER-SHAP: Random coalitions, marginal contributions, averaging.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            (T, F) or (B, T, F)
        Returns
        -------
        shap_vals : np.ndarray
            SHAP attributions (T, F) or (B, T, F)
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        shape = X_in.shape
        if len(shape) == 2:
            X_in = X_in[None, ...]
            single = True
        else:
            single = False
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_in[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            shap_matrix = np.zeros((T, F))
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    # Define weights for coalition sampling
                    weights = None
                    if self.weighting == "importance" and self.feature_importance is not None:
                        flat_imp = self.feature_importance.flatten()
                        idx_map = {idx: i for i, idx in enumerate(all_pos)}
                        weights = np.array([flat_imp[idx_map[idx]] for idx in available])
                        weights = weights / (weights.sum() + 1e-8)
                    elif self.weighting == "frequency":
                        weights = None  # Implemented as uniform, could use prior freq

                    for _ in range(self.n_coalitions):
                        # Uniform or weighted coalition size (avoid full/empty)
                        k = np.random.randint(1, len(available)+1)
                        C_idxs = self._sample_coalition(available, k, weights)
                        x_C = self._impute(x_orig, C_idxs)
                        x_C_tf = self._impute(x_C, [(t, f)])
                        out_C = self.model(torch.tensor(x_C[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        out_C_tf = self.model(torch.tensor(x_C_tf[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        mc.append(out_C_tf - out_C)
                    shap_matrix[t, f] = np.mean(mc)
            shap_vals[b] = shap_matrix

            # Additivity normalization per sample
            orig_pred = self.model(torch.tensor(x_orig[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            x_all_masked = self._impute(x_orig, all_pos)
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[ERSHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")
        return shap_vals

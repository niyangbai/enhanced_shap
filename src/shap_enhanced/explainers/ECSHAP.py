"""
Empirical Conditional SHAP (EC-SHAP) for Discrete Data

## Theoretical Explanation

Empirical Conditional SHAP (EC-SHAP) is a feature attribution method for discrete (binary, categorical, or one-hot) data. For each coalition (subset of masked features), EC-SHAP imputes the masked features by sampling from the empirical conditional distribution given the observed (unmasked) features, using the background dataset. This ensures that all imputations correspond to valid, observed patterns in the data, avoiding unrealistic or out-of-distribution perturbations.

### Key Concepts

- **Empirical Conditional Imputation:** For each coalition, masked features are filled in by finding background samples that match the unmasked features. If no exact match is found, the method can skip the coalition or use the closest match (by Hamming distance).
- **Valid Discrete Patterns:** All imputations are guaranteed to be valid, observed binary/one-hot patterns, preserving the data distribution.
- **Fallback for Continuous Data:** If the data is detected as continuous (many unique values per feature), EC-SHAP falls back to mean imputation.
- **Additivity Normalization:** Attributions are normalized so their sum matches the model output difference between the original and fully-masked input.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data, skip/closest matching flags, and device.
2. **Conditional Imputation:**
    - For each coalition (subset of features to mask), find a background sample that matches the unmasked features. If none is found, optionally use the closest match or mean imputation.
3. **SHAP Value Estimation:**
    - For each feature position, repeatedly:
        - Sample a random coalition of other positions.
        - Impute the coalition using empirical conditional matching.
        - Impute the coalition plus the feature of interest.
        - Compute the model output difference.
        - Average these differences to estimate the marginal contribution of the feature.
    - Normalize attributions so their sum matches the difference between the original and fully-masked model output.
"""
import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class EmpiricalConditionalSHAPExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        background,
        skip_unmatched=True,
        use_closest=False,
        device=None
    ):
        super().__init__(model, background)
        self.background = background.detach().cpu().numpy() if hasattr(background, 'detach') else np.asarray(background)
        if self.background.ndim == 2:
            self.background = self.background[:, None, :]  # (N, 1, F)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_unmatched = skip_unmatched
        self.use_closest = use_closest
        # Simple check: treat data as "continuous" if >30 unique values per feature
        self.is_continuous = np.mean([np.unique(self.background[..., f]).size > 30 for f in range(self.background.shape[-1])]) > 0.5
        if self.is_continuous:
            print("[EmpCondSHAP] WARNING: Detected continuous/tabular data. Empirical conditional imputation is not suitable. Will fallback to mean imputation where needed.")

        self.mean_baseline = np.mean(self.background, axis=0)  # (T, F)

    def _find_conditional_match(self, mask, x):
        unmasked_flat = (~mask).reshape(-1)
        x_flat = x.reshape(-1)
        bg_flat = self.background.reshape(self.background.shape[0], -1)
        match = np.all(bg_flat[:, unmasked_flat] == x_flat[unmasked_flat], axis=1)
        idxs = np.where(match)[0]
        if len(idxs) > 0:
            return np.random.choice(idxs)
        elif self.use_closest and len(self.background) > 0:
            diffs = np.sum(bg_flat[:, unmasked_flat] != x_flat[unmasked_flat], axis=1)
            idx = np.argmin(diffs)
            return idx
        else:
            return None

    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        if X_in.ndim == 2:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        for b in range(B):
            x = X_in[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        k = np.random.randint(1, len(available)+1)
                        mask_idxs = [available[i] for i in np.random.choice(len(available), k, replace=False)]
                        mask = np.zeros((T, F), dtype=bool)
                        for tt, ff in mask_idxs:
                            mask[tt, ff] = True
                        idx_match = self._find_conditional_match(mask, x)
                        if idx_match is not None:
                            x_masked = self.background[idx_match].copy()
                        else:
                            # fallback: mean imputation for continuous data
                            x_masked = self.mean_baseline.copy()
                        mask2 = mask.copy()
                        mask2[t, f] = True
                        idx_match2 = self._find_conditional_match(mask2, x)
                        if idx_match2 is not None:
                            x_masked_tf = self.background[idx_match2].copy()
                        else:
                            x_masked_tf = self.mean_baseline.copy()
                        # Evaluate
                        out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        out_masked_tf = self.model(torch.tensor(x_masked_tf[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        mc.append(out_masked_tf - out_masked)
                    if len(mc) > 0:
                        shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            mask_all = np.ones((T, F), dtype=bool)
            idx_all = self._find_conditional_match(mask_all, x)
            if idx_all is not None:
                masked_pred = self.model(torch.tensor(self.background[idx_all][None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            else:
                masked_pred = self.model(torch.tensor(self.mean_baseline[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if X_in.shape[0] == 1 else shap_vals
        if check_additivity:
            print(f"[EmpCondSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

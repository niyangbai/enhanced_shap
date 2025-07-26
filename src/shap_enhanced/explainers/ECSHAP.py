"""
EC-SHAP: Empirical Conditional SHAP for Discrete Data
=====================================================

Theoretical Explanation
-----------------------

Empirical Conditional SHAP (EC-SHAP) is a feature attribution method tailored for discrete data types,
including binary, categorical, and one-hot encoded features. Unlike classical SHAP methods that often rely
on unconditional sampling or simplistic imputations, EC-SHAP imputes masked features based on the
**empirical conditional distribution** derived from a background dataset.

For each coalition (a subset of masked features), EC-SHAP seeks background samples that match the unmasked features,
ensuring that imputed instances remain within the data manifold and reflect realistic, observed patterns.

Key Concepts
^^^^^^^^^^^^

- **Empirical Conditional Imputation**:
    Masked features are filled by matching the unmasked portion of an input to background data. If no exact match exists,
    the algorithm can either skip the coalition or use the closest match (by Hamming distance).

- **Valid Discrete Patterns**:
    All imputations correspond to real, observed combinations in the background dataset—preserving the statistical validity
    and interpretability of the perturbed inputs.

- **Fallback for Continuous Features**:
    If features appear continuous (e.g., many unique values), EC-SHAP automatically falls back to mean imputation.

- **Additivity Normalization**:
    Attributions are scaled such that their sum equals the difference in model outputs between the original
    and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, device context, and configuration for skipping or relaxing matches (e.g., using closest match).

2. **Conditional Imputation**:
    - For each coalition (subset of features to mask):
        - Identify background samples where the unmasked features match.
        - If a match exists, use it to fill in masked features.
        - If no match:
            - Optionally use the nearest match (by Hamming distance), or
            - Fallback to mean imputation (for continuous features), or
            - Skip the coalition.

3. **SHAP Value Estimation**:
    - For each feature:
        - Sample random coalitions of other features.
        - Impute both:
            - The coalition alone, and
            - The coalition plus the target feature.
        - Compute the difference in model outputs.
        - Average the differences to estimate marginal contribution.

4. **Normalization**:
   - Ensure the sum of feature attributions equals the difference in model output between the original and fully-masked input.
"""

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class EmpiricalConditionalSHAPExplainer(BaseExplainer):
    r"""
    Empirical Conditional SHAP (EC-SHAP) Explainer for Discrete Data

    This explainer estimates Shapley values for discrete (e.g., categorical, binary, or one-hot)
    feature inputs by imputing masked features from a background dataset using conditional matching.
    It ensures perturbed samples remain within the data manifold, preserving interpretability.

    :param model: Model to explain, must support PyTorch tensors as input.
    :type model: Any
    :param background: Background dataset used for empirical conditional imputation.
    :type background: np.ndarray or torch.Tensor
    :param skip_unmatched: If True, skip coalitions where no matching background sample exists.
    :type skip_unmatched: bool
    :param use_closest: If True, use the closest (Hamming distance) background sample when no exact match is found.
    :type use_closest: bool
    :param device: Device on which to run the model ('cpu' or 'cuda').
    :type device: Optional[str]
    """

    def __init__(
        self, model, background, skip_unmatched=True, use_closest=False, device=None
    ):
        super().__init__(model, background)
        self.background = (
            background.detach().cpu().numpy()
            if hasattr(background, "detach")
            else np.asarray(background)
        )
        if self.background.ndim == 2:
            self.background = self.background[:, None, :]  # (N, 1, F)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_unmatched = skip_unmatched
        self.use_closest = use_closest
        # Simple check: treat data as "continuous" if >30 unique values per feature
        self.is_continuous = (
            np.mean(
                [
                    np.unique(self.background[..., f]).size > 30
                    for f in range(self.background.shape[-1])
                ]
            )
            > 0.5
        )
        if self.is_continuous:
            print(
                "[EmpCondSHAP] WARNING: Detected continuous/tabular data. Empirical conditional imputation is not suitable. Will fallback to mean imputation where needed."
            )

        self.mean_baseline = np.mean(self.background, axis=0)  # (T, F)

    def _find_conditional_match(self, mask, x):
        r"""
        Find a background sample that matches the unmasked features of the input.

        If `use_closest` is enabled, falls back to the nearest background match
        (measured via Hamming distance) when no exact match is found.

        :param mask: Boolean mask array indicating masked positions (True).
        :type mask: np.ndarray
        :param x: Input array to match against background samples.
        :type x: np.ndarray
        :return: Index of matched background sample or None.
        :rtype: Optional[int]
        """
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
        self, X, nsamples=100, check_additivity=True, random_seed=42, **kwargs
    ):
        r"""
        Estimate SHAP values using empirical conditional imputation.

        For each feature-time index (t, f), this method:
        - Samples coalitions of other features.
        - Finds background samples matching the unmasked portion of the input.
        - Imputes masked values with corresponding values from the matched sample.
        - Computes model output with and without the target feature masked.
        - Averages the differences over multiple coalitions.

        Normalization ensures:

        .. math::
            \sum_{t=1}^T \sum_{f=1}^F \phi_{t,f} \approx f(x) - f(x_{\text{masked}})

        .. note::
            If no exact match is found and `use_closest` is False, the coalition may be skipped.
            For continuous-looking data, the method will fallback to mean imputation.

        :param X: Input data of shape (T, F) or (B, T, F)
        :type X: np.ndarray or torch.Tensor
        :param nsamples: Number of coalitions to sample per feature.
        :type nsamples: int
        :param check_additivity: Whether to rescale SHAP values to match model output difference.
        :type check_additivity: bool
        :param random_seed: Seed for reproducibility.
        :type random_seed: int
        :return: SHAP values of shape (T, F) or (B, T, F)
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, "detach")
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
                        k = np.random.randint(1, len(available) + 1)
                        mask_idxs = [
                            available[i]
                            for i in np.random.choice(len(available), k, replace=False)
                        ]
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
                        out_masked = (
                            self.model(
                                torch.tensor(
                                    x_masked[None],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        out_masked_tf = (
                            self.model(
                                torch.tensor(
                                    x_masked_tf[None],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        mc.append(out_masked_tf - out_masked)
                    if len(mc) > 0:
                        shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = (
                self.model(
                    torch.tensor(x[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            mask_all = np.ones((T, F), dtype=bool)
            idx_all = self._find_conditional_match(mask_all, x)
            if idx_all is not None:
                masked_pred = (
                    self.model(
                        torch.tensor(
                            self.background[idx_all][None],
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            else:
                masked_pred = (
                    self.model(
                        torch.tensor(
                            self.mean_baseline[None],
                            dtype=torch.float32,
                            device=self.device,
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if X_in.shape[0] == 1 else shap_vals
        if check_additivity:
            print(f"[EmpCondSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

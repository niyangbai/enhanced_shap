"""
Sparse Coalition SHAP Explainer
===============================

Theoretical Explanation
-----------------------

Sparse Coalition SHAP is a feature attribution method specifically designed for sparse, discrete,
or structured inputs—such as one-hot encodings and binary feature sets.
Unlike standard SHAP approaches that may generate invalid or unrealistic feature perturbations,
Sparse Coalition SHAP only considers **valid coalitions** that preserve the sparsity and logical structure
of the input space.

For one-hot encoded groups, masking a group zeroes out the entire set—representing "no selection"—without producing
fractional or ambiguous class encodings. For binary features, masking is performed element-wise while maintaining
input validity.

This approach ensures all perturbed inputs remain on-manifold, improving both interpretability and the validity
of model attributions in discrete domains.

Key Concepts
^^^^^^^^^^^^

- **Valid Sparse Coalitions**:
    Coalitions are restricted to those that produce syntactically valid inputs under the sparsity constraints.
    This avoids creating feature patterns that would never occur naturally.

- **One-Hot Group Support**:
    Groups of mutually exclusive features (e.g., one-hot encodings) are masked by setting the entire group to zero,
    simulating "no class selected."

- **Binary Feature Support**:
    Element-wise masking is applied to binary features, allowing localized coalitions across time and features.

- **Flexible Masking Strategies**:
    - Default: zero-masking.
    - Extensible to other strategies (e.g., pattern sampling from background data).

- **Additivity Normalization**:
    Final attributions are normalized so their total matches the difference between the model outputs
    of the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts the target model, background dataset, one-hot group definitions, masking strategy (default: zero),
        and device configuration.

2. **Coalition Sampling**:
    - For each one-hot group or binary feature:
        - Randomly sample subsets of other groups/features to form coalitions.
        - For each coalition:
            - Mask the selected features/groups in the input.
            - Mask the coalition plus the current target group/feature.
            - Compute the model outputs for both variants.
            - Record the output difference.

3. **SHAP Value Estimation**:
    - Average the output differences over many sampled coalitions to approximate the Shapley value
        (i.e., the marginal contribution) of each group/feature.

4. **Normalization**:
    - Scale all attributions so their sum equals the model output difference between
        the original and fully-masked inputs.

Use Case
--------

Ideal for models operating on:
    - Categorical variables represented via one-hot encoding.
    - Structured binary inputs (e.g., presence/absence features).
    - Sparse input spaces where validity and interpretability are critical.
"""

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class SparseCoalitionSHAPExplainer(BaseExplainer):
    r"""
    SparseCoalitionSHAPExplainer: Valid SHAP for Structured Sparse Inputs

    This explainer approximates Shapley values by sampling valid sparse coalitions of features.
    It ensures that perturbed inputs remain syntactically valid, especially for inputs with
    structured sparsity such as one-hot encodings or binary indicator features.

    .. note::
        One-hot groups are masked as entire sets to simulate "no class selected".
        General binary features are masked element-wise.

    :param model: Predictive model to explain.
    :type model: Any
    :param background: Background data (not directly used but required for base class).
    :type background: np.ndarray or torch.Tensor
    :param onehot_groups: List of one-hot index groups, e.g., [[0,1,2], [3,4]].
    :type onehot_groups: list[list[int]] or None
    :param mask_strategy: Currently supports only "zero" masking.
    :type mask_strategy: str
    :param device: Device context for evaluation (e.g., 'cuda' or 'cpu').
    :type device: str
    """

    def __init__(
        self, model, background, onehot_groups=None, mask_strategy="zero", device=None
    ):
        super().__init__(model, background)
        self.onehot_groups = onehot_groups  # e.g., [[0,1,2],[3,4,5],...]
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _mask(self, x, groups_to_mask):
        # x: (T, F)
        x_masked = x.copy()
        if self.onehot_groups is not None:
            # groups_to_mask: list of groups, each group is list of indices
            for group in groups_to_mask:
                for idx in group:
                    x_masked[:, idx] = 0
        else:
            # For general binary: groups_to_mask is a flat list of (t, f) tuples
            for t, f in groups_to_mask:
                x_masked[t, f] = 0
        return x_masked

    def shap_values(
        self, X, nsamples=100, check_additivity=True, random_seed=42, **kwargs
    ):
        r"""
        Estimate SHAP values using sparse-valid coalitions.

        For each input sample:
        - Iterates over all features (or one-hot groups).
        - Randomly samples subsets of other features/groups to form coalitions.
        - Computes model output difference when adding the current feature/group to the coalition.
        - Averages these differences to estimate the Shapley value.

        .. math::
            \phi_i = \mathbb{E}_{S \subseteq N \setminus \{i\}} \left[
                f(S \cup \{i\}) - f(S)
            \right]

        Final attributions are normalized such that:

        .. math::
            \sum_i \phi_i = f(x) - f(x_{\text{masked}})

        :param X: Input instance(s), shape (T, F) or (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of coalition samples per feature/group.
        :param bool check_additivity: If True, prints the additivity check.
        :param int random_seed: Seed for reproducible sampling.
        :return: SHAP attribution values, same shape as input.
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, "detach")
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = X_in.ndim == 2
        if single:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x = X_in[b]
            if self.onehot_groups is not None:
                # One-hot masking
                all_groups = self.onehot_groups
                for group in all_groups:
                    for idx in group:
                        contribs = []
                        groups_others = [g for g in all_groups if g != group]
                        for _ in range(nsamples):
                            # Sample random subset of other groups to mask
                            k = np.random.randint(0, len(groups_others) + 1)
                            C_idxs = np.random.choice(
                                len(groups_others), size=k, replace=False
                            )
                            mask_groups = [groups_others[i] for i in C_idxs]
                            # Mask C (other groups)
                            x_C = self._mask(x, mask_groups)
                            # Mask C + this group
                            x_C_g = self._mask(x_C, [group])
                            out_C = (
                                self.model(
                                    torch.tensor(
                                        x_C[None],
                                        dtype=torch.float32,
                                        device=self.device,
                                    )
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .squeeze()
                            )
                            out_C_g = (
                                self.model(
                                    torch.tensor(
                                        x_C_g[None],
                                        dtype=torch.float32,
                                        device=self.device,
                                    )
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .squeeze()
                            )
                            contribs.append(out_C - out_C_g)
                        # Assign SHAP value to all features in this group equally (or just to idx)
                        shap_vals[b, :, idx] = np.mean(contribs) / len(group)
            else:
                # General binary: per (t, f)
                all_pos = [(t, f) for t in range(T) for f in range(F)]
                for t in range(T):
                    for f in range(F):
                        contribs = []
                        available = [idx for idx in all_pos if idx != (t, f)]
                        for _ in range(nsamples):
                            # Mask random subset of others
                            k = np.random.randint(0, len(available) + 1)
                            C_idxs = np.random.choice(
                                len(available), size=k, replace=False
                            )
                            mask_idxs = [available[i] for i in C_idxs]
                            x_C = self._mask(x, mask_idxs)
                            x_C_tf = self._mask(x_C, [(t, f)])
                            out_C = (
                                self.model(
                                    torch.tensor(
                                        x_C[None],
                                        dtype=torch.float32,
                                        device=self.device,
                                    )
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .squeeze()
                            )
                            out_C_tf = (
                                self.model(
                                    torch.tensor(
                                        x_C_tf[None],
                                        dtype=torch.float32,
                                        device=self.device,
                                    )
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .squeeze()
                            )
                            contribs.append(out_C - out_C_tf)
                        shap_vals[b, t, f] = np.mean(contribs)
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
            if self.onehot_groups is not None:
                x_all_masked = self._mask(x, self.onehot_groups)
            else:
                all_pos = [(t, f) for t in range(T) for f in range(F)]
                x_all_masked = self._mask(x, all_pos)
            masked_pred = (
                self.model(
                    torch.tensor(
                        x_all_masked[None], dtype=torch.float32, device=self.device
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

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(
                f"[SparseCoalitionSHAP] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}"
            )
        return shap_vals

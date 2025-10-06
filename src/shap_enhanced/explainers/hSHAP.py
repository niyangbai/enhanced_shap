"""
h-SHAP: Hierarchical SHAP Explainer
===================================

Theoretical Explanation
-----------------------

Hierarchical SHAP (h-SHAP) is an extension of the SHAP framework that enables structured, group-wise attribution
of features in models operating over high-dimensional or structured input data, such as time series or grouped tabular features.

Instead of treating each feature independently, h-SHAP introduces a hierarchy of feature groups, allowing recursive
estimation of SHAP values at multiple levels—first over coarse groups, then over finer subgroups or individual features.
This promotes interpretability and computational efficiency in contexts where feature dimensions have natural structure.

Key Concepts
^^^^^^^^^^^^

- **Hierarchical Grouping**:
    Features are grouped into blocks (e.g., temporal windows, spatial zones, feature families), possibly in multiple nested levels.
    These groups define the hierarchy over which SHAP values are computed.

- **Recursive SHAP Estimation**:
    SHAP values are estimated first at the group level, and then recursively subdivided among subgroups or features
    within each group. This preserves hierarchical structure in the resulting attribution map.

- **Flexible Masking**:
    Features can be masked by:
        - Setting them to zero (hard masking).
        - Imputing with background mean values (soft masking), using a provided reference dataset.

- **Additivity Normalization**:
    Final attributions are normalized such that their total sum equals the model output difference
    between the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset for imputation, a user-defined hierarchy of feature groups,
        masking strategy (`'zero'` or `'mean'`), and a device context.

2. **Recursive Attribution**:
    - For each group in the hierarchy:
        - Sample coalitions of other groups.
        - Estimate the group’s marginal contribution by masking:
            - Only the coalition, and
            - The coalition plus the current group.
        - Compute the model output difference to get SHAP value.
        - If the group contains subgroups, repeat recursively.
        - If not, distribute the SHAP value equally among group members or subfeatures.

3. **Normalization**:
    - Rescale all SHAP values so that their sum matches the change in model output
        between the unmasked input and the fully-masked input.

References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**
  [SHAP foundation—coalitional feature attribution framework]

- **Teneggi, Luster & Sulam (2022), *Fast Hierarchical Games for Image Explanations (h‑Shap)***
  [Exact hierarchical Shapley values for groups of pixels, with exponential computational improvements] :contentReference[oaicite:1]{index=1}

- **Jullum, Redelmeier & Løland (2021), *groupShapley: Efficient prediction explanation with Shapley values for feature groups***
  [Introduces grouping features and computing Shapley values at group level for efficiency and interpretability] :contentReference[oaicite:2]{index=2}

- **Wang et al. (2025), *Group Shapley with Robust Significance Testing* **
  [Extends Shapley framework to hierarchical groups and provides statistical tests for group contributions] :contentReference[oaicite:3]{index=3}

- **SHAP PartitionExplainer documentation**
  [Implements recursive hierarchical coalitions and Owen‑value attribution via user‑defined feature hierarchies] :contentReference[oaicite:4]{index=4}

- **Molnar, *Interpretable Machine Learning* (2022), SHAP chapter**
  [Discusses hierarchical coalitions, Owen and Shapley values, and structured feature grouping strategies] :contentReference[oaicite:5]{index=5}
"""

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


def generate_hierarchical_groups(
    T, F, time_block=None, feature_block=None, nested=False
):
    r"""
    Generate a hierarchical grouping of features in a (T, F)-shaped input space.

    This utility is useful for constructing hierarchical feature groupings for use with hierarchical SHAP (h-SHAP).
    Depending on the specified block sizes, it partitions time steps, features, or both into groups. If `nested=True`,
    the function returns subgroups within each block.

    .. note::
        The output format supports both flat and nested group structures, which are compatible with recursive SHAP attribution.

    :param int T: Number of time steps (first input dimension).
    :param int F: Number of features (second input dimension).
    :param int or None time_block: Size of time blocks (e.g., 2 for groups of 2 time steps). If None, no time blocking.
    :param int or None feature_block: Size of feature blocks. If None, no feature blocking.
    :param bool nested: Whether to return nested groups (group → list of singleton subgroups).
    :return: A list of grouped (t, f) indices, either flat or nested depending on `nested`.
    :rtype: list[list[tuple[int, int]]] or list[list[list[tuple[int, int]]]]
    """
    # Block by time only
    if time_block is not None and feature_block is None:
        groups = [
            [(t, f) for t in range(i, min(i + time_block, T)) for f in range(F)]
            for i in range(0, T, time_block)
        ]
    # Block by feature only
    elif feature_block is not None and time_block is None:
        groups = [
            [(t, f) for f in range(j, min(j + feature_block, F)) for t in range(T)]
            for j in range(0, F, feature_block)
        ]
    # Block by both time and feature (nested grid)
    elif time_block is not None and feature_block is not None:
        groups = []
        for i in range(0, T, time_block):
            for j in range(0, F, feature_block):
                block = [
                    (t, f)
                    for t in range(i, min(i + time_block, T))
                    for f in range(j, min(j + feature_block, F))
                ]
                groups.append(block)
    else:
        # Default: each (t, f) as its own group
        groups = [[(t, f)] for t in range(T) for f in range(F)]

    if nested and (time_block is not None or feature_block is not None):
        # Example of a nested hierarchy: block-of-2 time, then each time point as a subgroup
        hierarchy = []
        for group in groups:
            # For each block, nest as smaller singleton subgroups
            subgroups = [[idx] for idx in group]
            hierarchy.append(subgroups)
        return hierarchy
    else:
        return groups


class HShapExplainer(BaseExplainer):
    r"""
    HShapExplainer: Hierarchical SHAP Explainer

    Implements the h-SHAP algorithm, which recursively computes SHAP values over structured
    groups of features using hierarchical masking. Suitable for time-series or block-structured
    feature inputs where interpretability benefits from grouped attributions.

    .. note::
        Features can be masked using hard zero-masking or soft imputation via background means.

    :param model: Model to explain.
    :param background: Background dataset for mean imputation. Shape: (N, T, F).
    :type background: np.ndarray or torch.Tensor
    :param hierarchy: Nested list of feature index groups (e.g., [[(t1, f1), (t2, f2)], ...]).
    :type hierarchy: list
    :param str mask_strategy: Either "mean" for imputation or "zero" for hard masking.
    :param str device: Device context, e.g., "cuda" or "cpu".
    """

    def __init__(self, model, background, hierarchy, mask_strategy="mean", device=None):
        super().__init__(model, background)
        self.hierarchy = hierarchy  # List of groups, or nested list
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if mask_strategy == "mean":
            self._mean = background.mean(axis=0)
        else:
            self._mean = None

    def _impute(self, X, idxs):
        r"""
        Mask or impute features in `X` at the given (t, f) indices according to the configured strategy.

        :param X: Input sample to be modified.
        :type X: np.ndarray
        :param idxs: List of (t, f) indices to mask.
        :type idxs: list of tuples
        :return: Modified input sample with masked/imputed values.
        :rtype: np.ndarray
        """
        X_imp = X.copy()
        for t, f in idxs:
            if self.mask_strategy == "zero":
                X_imp[t, f] = 0.0
            elif self.mask_strategy == "mean":
                X_imp[t, f] = self._mean[t, f]
            else:
                raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")
        return X_imp

    def _shap_group(self, x, group_idxs, rest_idxs, nsamples=50):
        r"""
        Estimate the SHAP value of a feature group by computing its marginal contribution
        compared to sampled subsets of other groups.

        .. math::
            \phi(S) = \mathbb{E}_{R \subseteq \text{rest}} \left[
                f(x_{\text{rest}}) - f(x_{\text{rest} \cup S})
            \right]

        :param x: Input sample (T, F).
        :type x: np.ndarray
        :param group_idxs: Indices of the current group.
        :type group_idxs: list of tuples
        :param rest_idxs: Indices of all other groups.
        :type rest_idxs: list of tuples
        :param int nsamples: Number of random coalitions to sample for marginal estimation.
        :return: Estimated SHAP value for the group.
        :rtype: float
        """
        contribs = []
        group_idxs + rest_idxs
        for _ in range(nsamples):
            # Sample subset of rest to mask
            k = np.random.randint(0, len(rest_idxs) + 1)
            if k > 0:
                idx_choices = np.random.choice(len(rest_idxs), size=k, replace=False)
                rest_sample = [rest_idxs[i] for i in idx_choices]
            else:
                rest_sample = []
            # Mask: (rest_sample only), then (rest_sample + group)
            x_rest = self._impute(x, rest_sample)
            x_both = self._impute(x_rest, group_idxs)
            out_rest = (
                self.model(
                    torch.tensor(x_rest[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            out_both = (
                self.model(
                    torch.tensor(x_both[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            contribs.append(out_rest - out_both)
        return np.mean(contribs)

    def _explain_recursive(self, x, groups, nsamples=50, attributions=None):
        r"""
        Recursively apply SHAP attribution over a hierarchical structure of groups.

        If subgroups exist, the SHAP value is divided equally among sub-elements.
        Accumulates attributions for each (t, f) index.

        :param x: Input instance to explain.
        :type x: np.ndarray
        :param groups: List of groups (can be nested).
        :type groups: list
        :param int nsamples: Number of samples per group SHAP estimation.
        :param attributions: Dictionary to accumulate attributions.
        :type attributions: dict or None
        :return: Dictionary mapping (t, f) indices to SHAP values.
        :rtype: dict
        """
        if attributions is None:
            attributions = {}
        group_indices = []
        for group in groups:
            if isinstance(group[0], tuple | list):
                # group is a nested group, recurse
                self._explain_recursive(x, group, nsamples, attributions)
                # group_indices += flatten(group)
                group_indices += [
                    idx
                    for g in group
                    for idx in (g if isinstance(g[0], tuple | list) else [g])
                ]
            else:
                group_indices += [group]

        # At this hierarchy level, estimate group SHAP for each group
        for group in groups:
            if isinstance(group[0], tuple | list):
                flat_group = [
                    idx
                    for g in group
                    for idx in (g if isinstance(g[0], tuple | list) else [g])
                ]
            else:
                flat_group = [group]
            rest = [idx for idx in group_indices if idx not in flat_group]
            phi = self._shap_group(x, flat_group, rest, nsamples=nsamples)
            # Split SHAP value equally among group members
            for idx in flat_group:
                attributions[idx] = attributions.get(idx, 0.0) + phi / len(flat_group)
        return attributions

    def shap_values(
        self, X, nsamples=50, check_additivity=True, random_seed=42, **kwargs
    ):
        r"""
        Compute hierarchical SHAP values for a batch of inputs.

        The method recursively attributes model output to hierarchical feature groups.
        It also ensures additivity via normalization of final attributions.

        .. math::
            \sum_{i=1}^{TF} \phi_i = f(x) - f(x_{\text{masked}})

        :param X: Input batch, shape (B, T, F) or single instance (T, F).
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of Monte Carlo samples per group.
        :param bool check_additivity: If True, prints additivity check summary.
        :param int random_seed: Seed for reproducible sampling.
        :return: SHAP values, same shape as `X`.
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, "detach")
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
            x = X_in[b]
            attr = self._explain_recursive(x, self.hierarchy, nsamples=nsamples)
            for (t, f), v in attr.items():
                shap_vals[b, t, f] = v
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
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            x_all_masked = self._impute(x, all_pos)
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
                f"[h-SHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}"
            )
        return shap_vals

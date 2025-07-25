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
        - Estimate the group's marginal contribution by masking:
            - Only the coalition, and
            - The coalition plus the current group.
        - Compute the model output difference to get SHAP value.
        - If the group contains subgroups, repeat recursively.
        - If not, distribute the SHAP value equally among group members or subfeatures.

3. **Normalization**:
    - Rescale all SHAP values so that their sum matches the change in model output  
        between the unmasked input and the fully-masked input.
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler
from shap_enhanced.algorithms.masking import ZeroMasker, MeanMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor


def generate_hierarchical_groups(
    T, F, 
    time_block=None, 
    feature_block=None,
    nested=False
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
            [(t, f) for t in range(i, min(i+time_block, T)) for f in range(F)]
            for i in range(0, T, time_block)
        ]
    # Block by feature only
    elif feature_block is not None and time_block is None:
        groups = [
            [(t, f) for f in range(j, min(j+feature_block, F)) for t in range(T)]
            for j in range(0, F, feature_block)
        ]
    # Block by both time and feature (nested grid)
    elif time_block is not None and feature_block is not None:
        groups = []
        for i in range(0, T, time_block):
            for j in range(0, F, feature_block):
                block = [(t, f) for t in range(i, min(i+time_block, T))
                                 for f in range(j, min(j+feature_block, F))]
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


class HierarchicalMasker:
    """Custom masker for hierarchical SHAP with group-based masking."""
    
    def __init__(self, mask_strategy="mean", background_mean=None):
        self.mask_strategy = mask_strategy
        self.background_mean = background_mean
    
    def mask_groups(self, x, group_positions):
        """Apply hierarchical group-based masking."""
        x_masked = x.copy()
        
        for group in group_positions:
            for t, f in group:
                if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                    if self.mask_strategy == "zero":
                        x_masked[t, f] = 0.0
                    elif self.mask_strategy == "mean":
                        x_masked[t, f] = self.background_mean[t, f]
        
        return x_masked


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
    def __init__(
        self,
        model,
        background,
        hierarchy,
        mask_strategy="mean",
        device=None
    ):
        super().__init__(model, background)
        self.hierarchy = hierarchy  # List of groups, or nested list
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process background data
        self.background = BackgroundProcessor.process_background(background)
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)
        
        if mask_strategy == "mean":
            bg_stats = BackgroundProcessor.compute_background_statistics(self.background)
            self.masker = HierarchicalMasker(mask_strategy, bg_stats['mean'])
        else:
            self.masker = HierarchicalMasker(mask_strategy)

    def _shap_group(self, x, group_idxs, rest_groups, nsamples=50):
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
        :param rest_groups: List of other groups.
        :type rest_groups: list of groups
        :param int nsamples: Number of random coalitions to sample for marginal estimation.
        :return: Estimated SHAP value for the group.
        :rtype: float
        """
        marginal_contributions = []
        
        for _ in range(nsamples):
            # Sample subset of rest groups to mask
            k = np.random.randint(0, len(rest_groups) + 1)
            if k > 0:
                idx_choices = np.random.choice(len(rest_groups), size=k, replace=False)
                rest_sample = [rest_groups[i] for i in idx_choices]
            else:
                rest_sample = []
            
            # Evaluate model with coalition
            x_coalition = self.masker.mask_groups(x, rest_sample)
            pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
            
            # Evaluate model with coalition + target group
            coalition_plus_target = rest_sample + [group_idxs]
            x_coalition_plus = self.masker.mask_groups(x, coalition_plus_target)
            pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
            
            # Marginal contribution
            contribution = pred_coalition - pred_coalition_plus
            marginal_contributions.append(contribution)
            
        return np.mean(marginal_contributions)

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
            
        # Flatten nested groups for marginal computation
        flattened_groups = []
        for group in groups:
            if isinstance(group[0], list):
                # Nested group - flatten it
                flat_group = [idx for subgroup in group for idx in subgroup]
                flattened_groups.append(flat_group)
            else:
                flattened_groups.append(group)
        
        # Compute SHAP values for each group
        for i, group in enumerate(flattened_groups):
            rest_groups = [flattened_groups[j] for j in range(len(flattened_groups)) if j != i]
            phi = self._shap_group(x, group, rest_groups, nsamples=nsamples)
            
            # Distribute SHAP value equally among group members
            for t, f in group:
                attributions[(t, f)] = attributions.get((t, f), 0.0) + phi / len(group)
                
        return attributions

    def shap_values(
        self,
        X,
        nsamples=50,
        check_additivity=True,
        random_seed=42,
        **kwargs
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
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        
        for b in range(B):
            x_orig = X_processed[b]
            
            # Compute hierarchical SHAP attributions
            attr = self._explain_recursive(x_orig, self.hierarchy, nsamples=nsamples)
            
            for (t, f), v in attr.items():
                if 0 <= t < T and 0 <= f < F:
                    shap_vals[b, t, f] = v
            
            # Apply additivity normalization using common algorithm
            all_positions = [(t, f) for t in range(T) for f in range(F)]
            fully_masked = self.masker.mask_groups(x_orig, [all_positions])
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[h-SHAP Additivity] sum(SHAP)={result.sum():.4f}")
        
        return result
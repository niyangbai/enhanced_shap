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

- **Binary Feature Masking**:  
    Individual binary features are masked element-wise while respecting feature dependencies.

- **Additivity Normalization**:  
    Final attributions are scaled such that their total equals the model output difference between  
    the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, optional one-hot group definitions, masking strategy, and device context.

2. **Sparse Coalition Sampling**:
    - For one-hot groups: Sample coalitions of entire groups to mask.
    - For individual features: Sample coalitions of feature-time pairs.
    - Ensure all masked combinations respect sparsity constraints.

3. **SHAP Value Estimation**:
    - For each feature or feature group:
        - Repeatedly sample coalitions of other features/groups.
        - Compute model output after masking the coalition.
        - Compute model output after masking the coalition plus the target feature/group.
        - Record the difference to estimate the marginal contribution.
        - Average these differences across sampled coalitions.

4. **Normalization**:
    - Scale the final attributions so their sum matches the model output difference  
        between the unmasked and fully-masked input.

Use Case
--------

Ideal for:
    - One-hot encoded categorical features.
    - Binary indicators (presence/absence).
    - Sparse high-dimensional data where only valid observed patterns should be used for attribution.
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import BaseMasker, ZeroMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs


class SparseCoalitionMasker(BaseMasker):
    """Custom masker for sparse coalition SHAP with one-hot group support."""
    
    def __init__(self, onehot_groups=None):
        self.onehot_groups = onehot_groups
    
    def mask_features(self, x, mask_positions):
        """Apply sparse coalition masking."""
        x_masked = x.copy()
        
        if self.onehot_groups is not None:
            # Handle one-hot groups - mask_positions contains group indices
            for group_idx in mask_positions:
                if isinstance(group_idx, (list, tuple)):
                    # Direct group indices
                    for feature_idx in group_idx:
                        x_masked[:, feature_idx] = 0
                else:
                    # Group index reference
                    if group_idx < len(self.onehot_groups):
                        for feature_idx in self.onehot_groups[group_idx]:
                            x_masked[:, feature_idx] = 0
        else:
            # Handle individual (t, f) positions
            for position in mask_positions:
                if isinstance(position, tuple) and len(position) == 2:
                    t, f = position
                    if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                        x_masked[t, f] = 0
        
        return x_masked


class SparseCoalitionSHAPExplainer(BaseExplainer):
    r"""
    SparseCoalitionSHAPExplainer: Valid SHAP for Structured Sparse Inputs

    This explainer approximates Shapley values by sampling valid sparse coalitions of features. 
    It ensures that perturbed inputs remain syntactically valid, especially for inputs with 
    structured sparsity such as one-hot encodings or binary indicator features.

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
        self,
        model,
        background,
        onehot_groups=None,
        mask_strategy="zero",
        device=None
    ):
        super().__init__(model, background)
        self.onehot_groups = onehot_groups
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.masker = SparseCoalitionMasker(onehot_groups)
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)

    def _get_coalition_space(self, T, F):
        """Get the appropriate coalition space based on one-hot groups."""
        if self.onehot_groups is not None:
            # For one-hot groups, coalitions are group indices
            return list(range(len(self.onehot_groups)))
        else:
            # For individual features, coalitions are (t, f) positions
            return create_all_positions(T, F)

    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        r"""
        Estimate SHAP values using sparse-valid coalitions.

        For each input sample:
        - Iterates over all features (or one-hot groups).
        - Randomly samples subsets of other features/groups to form coalitions.
        - Computes model output difference when adding the current feature/group to the coalition.
        - Averages these differences to estimate the Shapley value.

        :param X: Input instance(s), shape (T, F) or (B, T, F).
        :param nsamples: Number of coalition samples per feature/group.
        :param check_additivity: Whether to print additivity diagnostic.
        :param random_seed: Random seed for reproducibility.
        :return: SHAP values with same shape as input.
        """
        np.random.seed(random_seed)
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        
        # Initialize SHAP values
        if self.onehot_groups is not None:
            # For one-hot groups, compute group-level SHAP values
            n_groups = len(self.onehot_groups)
            group_shap_vals = np.zeros((B, n_groups), dtype=float)
        
        shap_vals = np.zeros((B, T, F), dtype=float)
        coalition_space = self._get_coalition_space(T, F)
        
        for b in range(B):
            x_orig = X_processed[b]
            
            if self.onehot_groups is not None:
                # Compute SHAP values for one-hot groups
                for group_idx in range(n_groups):
                    marginal_contributions = []
                    
                    for _ in range(nsamples):
                        # Sample coalition of other groups
                        other_groups = [g for g in range(n_groups) if g != group_idx]
                        if len(other_groups) > 0:
                            k = np.random.randint(0, len(other_groups) + 1)
                            coalition = np.random.choice(other_groups, k, replace=False).tolist()
                        else:
                            coalition = []
                        
                        # Evaluate model with coalition
                        x_coalition = self.masker.mask_features(x_orig, coalition)
                        pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                        
                        # Evaluate model with coalition + target group
                        coalition_plus_target = coalition + [group_idx]
                        x_coalition_plus = self.masker.mask_features(x_orig, coalition_plus_target)
                        pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                        
                        # Marginal contribution
                        contribution = pred_coalition_plus - pred_coalition
                        marginal_contributions.append(contribution)
                    
                    group_shap_vals[b, group_idx] = np.mean(marginal_contributions)
                
                # Distribute group SHAP values to individual features
                for group_idx, feature_indices in enumerate(self.onehot_groups):
                    group_value = group_shap_vals[b, group_idx]
                    # Distribute equally among group features (could be enhanced)
                    for feature_idx in feature_indices:
                        if feature_idx < F:
                            shap_vals[b, :, feature_idx] = group_value / len(feature_indices)
            
            else:
                # Standard individual feature SHAP computation
                for t in range(T):
                    for f in range(F):
                        marginal_contributions = []
                        target_position = (t, f)
                        
                        for _ in range(nsamples):
                            # Sample coalition excluding target feature
                            coalition = self.coalition_sampler.sample_coalition(
                                coalition_space, exclude=target_position
                            )
                            
                            # Evaluate model with coalition
                            x_coalition = self.masker.mask_features(x_orig, coalition)
                            pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                            
                            # Evaluate model with coalition + target feature
                            coalition_plus_target = coalition + [target_position]
                            x_coalition_plus = self.masker.mask_features(x_orig, coalition_plus_target)
                            pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                            
                            # Marginal contribution
                            contribution = pred_coalition_plus - pred_coalition
                            marginal_contributions.append(contribution)
                        
                        shap_vals[b, t, f] = np.mean(marginal_contributions)
            
            # Apply additivity normalization using common algorithm
            if self.onehot_groups is not None:
                # For one-hot data, mask all groups
                all_groups = list(range(len(self.onehot_groups)))
                fully_masked = self.masker.mask_features(x_orig, all_groups)
            else:
                # For individual features, mask all positions
                fully_masked = self.masker.mask_features(x_orig, coalition_space)
            
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[SparseCoalitionSHAP] sum(SHAP)={result.sum():.4f}")
        
        return result
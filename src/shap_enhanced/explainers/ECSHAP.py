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
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import BaseMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor


class EmpiricalConditionalMasker(BaseMasker):
    """Custom masker for empirical conditional SHAP with background matching."""
    
    def __init__(self, background_data, skip_unmatched=True, use_closest=False):
        self.background_data = background_data
        self.skip_unmatched = skip_unmatched
        self.use_closest = use_closest
        
        # Compute mean baseline for fallback
        self.mean_baseline = np.mean(background_data, axis=0)
        
        # Simple heuristic to detect continuous data
        F = background_data.shape[-1]
        continuous_features = []
        for f in range(F):
            unique_vals = np.unique(background_data[..., f])
            continuous_features.append(len(unique_vals) > 30)
        
        self.is_continuous = np.mean(continuous_features) > 0.5
        if self.is_continuous:
            print("[EmpCondSHAP] WARNING: Detected continuous/tabular data. Will fallback to mean imputation where needed.")
    
    def _find_conditional_match(self, mask, x):
        """Find background sample matching unmasked features."""
        unmasked_flat = (~mask).reshape(-1)
        x_flat = x.reshape(-1)
        bg_flat = self.background_data.reshape(self.background_data.shape[0], -1)
        
        # Look for exact matches
        match = np.all(bg_flat[:, unmasked_flat] == x_flat[unmasked_flat], axis=1)
        idxs = np.where(match)[0]
        
        if len(idxs) > 0:
            return np.random.choice(idxs)
        elif self.use_closest and len(self.background_data) > 0:
            # Use closest match by Hamming distance
            diffs = np.sum(bg_flat[:, unmasked_flat] != x_flat[unmasked_flat], axis=1)
            idx = np.argmin(diffs)
            return idx
        else:
            return None
    
    def mask_features(self, x, mask_positions):
        """Apply empirical conditional masking."""
        # Create mask from positions
        mask = np.zeros_like(x, dtype=bool)
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                mask[t, f] = True
        
        # Find conditional match
        idx_match = self._find_conditional_match(mask, x)
        
        if idx_match is not None:
            # Use matched background sample
            x_masked = self.background_data[idx_match].copy()
            # Keep unmasked features from original
            x_masked[~mask] = x[~mask]
        else:
            # Fallback to mean imputation
            x_masked = x.copy()
            x_masked[mask] = self.mean_baseline[mask]
        
        return x_masked


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
        self,
        model,
        background,
        skip_unmatched=True,
        use_closest=False,
        device=None
    ):
        super().__init__(model, background)
        
        # Process background data
        self.background = BackgroundProcessor.process_background(background)
        self.skip_unmatched = skip_unmatched
        self.use_closest = use_closest
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.masker = EmpiricalConditionalMasker(
            self.background, skip_unmatched, use_closest
        )
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)

    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
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
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        
        all_positions = create_all_positions(T, F)
        
        for b in range(B):
            x_orig = X_processed[b]
            
            # Compute SHAP values for each position
            for t in range(T):
                for f in range(F):
                    marginal_contributions = []
                    target_position = (t, f)
                    
                    for _ in range(nsamples):
                        # Sample coalition excluding target feature
                        coalition = self.coalition_sampler.sample_coalition(
                            all_positions, exclude=target_position
                        )
                        
                        # Evaluate model with coalition
                        x_coalition = self.masker.mask_features(x_orig, coalition)
                        pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                        
                        # Evaluate model with coalition + target feature
                        coalition_plus_target = coalition + [target_position]
                        x_coalition_plus = self.masker.mask_features(x_orig, coalition_plus_target)
                        pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                        
                        # Marginal contribution (note: EC-SHAP computes different sign)
                        contribution = pred_coalition_plus - pred_coalition
                        marginal_contributions.append(contribution)
                    
                    if len(marginal_contributions) > 0:
                        shap_vals[b, t, f] = np.mean(marginal_contributions)
            
            # Apply additivity normalization using common algorithm
            fully_masked = self.masker.mask_features(x_orig, all_positions)
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[EmpCondSHAP] sum(SHAP)={result.sum():.4f}")
        
        return result
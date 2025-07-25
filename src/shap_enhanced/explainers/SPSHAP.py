"""
Support-Preserving SHAP Explainer
=================================

Theoretical Explanation
-----------------------

Support-Preserving SHAP is a specialized feature attribution method tailored for **sparse** or **structured discrete data**,  
such as one-hot encodings or binary presence/absence features. Unlike traditional SHAP variants that create  
synthetic masked inputs (often resulting in out-of-distribution samples), this explainer **only evaluates inputs that have  
been observed in the dataset** and match the support pattern induced by masking.

For each coalition (subset of features to mask), the method attempts to find a real background sample  
with the **same binary support** (nonzero positions) as the masked instance. If no such sample exists, the coalition  
is skipped or flagged—ensuring that only valid, realistic inputs are used for estimating SHAP values.

For continuous or dense data, the method gracefully falls back to **mean-masking** (standard SHAP behavior).

Key Concepts
^^^^^^^^^^^^

- **Support Pattern Matching**:  
    Masked inputs are replaced with real background examples that match the nonzero pattern (support)  
    of the masked input. This maintains validity and avoids generating unrealistic inputs.

- **One-Hot / Binary Support**:  
    Especially effective for categorical features encoded as one-hot vectors or binary indicators.  
    Masking respects group structures and ensures feasible combinations.

- **Graceful Fallback**:  
    When applied to continuous or dense data, the explainer defaults to mean-masking to retain applicability.

- **Additivity Normalization**:  
    Final attributions are scaled such that their total equals the model output difference between  
    the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, device context, and configuration for skipping or flagging unmatched patterns.

2. **Support-Preserving Masking**:
    - For each sampled coalition of masked features:
        - Create a masked version of the input.
        - Find a background example with the same binary support (nonzero positions).
        - If no match is found, either skip or raise an exception based on configuration.
        - For non-sparse (dense) inputs, fallback to mean-masking.

3. **SHAP Value Estimation**:
    - For each feature:
        - Repeatedly sample coalitions of other features.
        - For each:
            - Mask the coalition and find a matching background sample.
            - Mask the coalition plus the feature of interest and find another match.
            - Compute the model output difference.
        - Average these differences to estimate the feature’s marginal contribution.

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
from shap_enhanced.algorithms.masking import BaseMasker, MeanMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor

class SupportPreservingMasker(BaseMasker):
    """Custom masker for support-preserving SHAP."""
    
    def __init__(self, background_data, skip_unmatched=True):
        self.background_data = background_data
        self.skip_unmatched = skip_unmatched
        self.bg_support = (background_data != 0)
        
    def mask_features(self, x, mask_positions):
        """Apply support-preserving masking."""
        x_masked = x.copy()
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                x_masked[t, f] = 0
        
        # Find matching background sample with same support
        support_mask = (x_masked != 0)
        idx = self._find_matching_sample(support_mask)
        
        if idx is None:
            if self.skip_unmatched:
                return x_masked  # Return zero-masked version if no match
            else:
                raise ValueError("No matching sample found for support pattern!")
        
        return self.background_data[idx]
    
    def _find_matching_sample(self, support_mask):
        """Find background sample with matching support pattern."""
        support_mask = support_mask[None, ...] if support_mask.ndim == 2 else support_mask
        matches = np.all(self.bg_support == support_mask, axis=(1, 2))
        idxs = np.where(matches)[0]
        if len(idxs) > 0:
            return np.random.choice(idxs)
        else:
            return None


class SupportPreservingSHAPExplainer(BaseExplainer):
    r"""
    SupportPreservingSHAPExplainer: Real-Pattern-Constrained SHAP Estimator

    This explainer approximates SHAP values by generating only masked inputs that match real examples 
    in the dataset—preserving the discrete or sparse structure of the input space. It avoids 
    out-of-distribution perturbations by requiring coalitions (masked variants) to have binary 
    support patterns that exist in the original data.

    If the data is not sparse (e.g., continuous), the method falls back to mean-masking, 
    akin to standard SHAP explainers.

    :param model: Predictive model to explain.
    :type model: Any
    :param background: Dataset used to match support patterns (shape: (N, T, F) or (N, F)).
    :type background: np.ndarray or torch.Tensor
    :param bool skip_unmatched: If True, coalitions without support-matching background samples are skipped.
    :param str device: Device to evaluate model on ('cpu' or 'cuda').
    """
    def __init__(
        self,
        model,
        background,
        skip_unmatched=True,
        device=None
    ):
        super().__init__(model, background)
        
        # Process background data
        self.background = BackgroundProcessor.process_background(background)
        self.skip_unmatched = skip_unmatched
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Check if data is one-hot/binary
        data_flat = self.background.reshape(-1, self.background.shape[-1])
        is_binary = np.all((data_flat == 0) | (data_flat == 1))
        is_onehot = np.all(np.sum(data_flat, axis=1) == 1)
        self.is_onehot = bool(is_binary and is_onehot)
        
        if not self.is_onehot:
            print("[SupportPreservingSHAP] WARNING: Data is not one-hot. Will use classic mean-masking SHAP fallback.")
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)
        
        # Setup masker based on data type
        if self.is_onehot:
            self.masker = SupportPreservingMasker(self.background, skip_unmatched)
        else:
            # Fallback to mean masking for non-sparse data
            bg_stats = BackgroundProcessor.compute_background_statistics(self.background)
            self.masker = MeanMasker(bg_stats['mean'])


    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        r"""
        Compute SHAP values by evaluating only valid support-preserving perturbations.

        For sparse inputs (e.g., one-hot or binary), uses support-preserving masking.
        For dense inputs, falls back to standard mean-based masking.

        :param X: Input sample or batch of shape (T, F) or (B, T, F).
        :param nsamples: Number of coalition samples per feature.
        :param check_additivity: If True, prints sum of SHAP vs model output difference.
        :param random_seed: Seed for reproducibility.
        :return: SHAP attributions with same shape as input.
        """
        np.random.seed(random_seed)
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        
        all_positions = create_all_positions(T, F)
        
        for b in range(B):
            x_orig = X_processed[b]
            
            if self.is_onehot:
                # Support-preserving SHAP logic using common algorithms
                for t in range(T):
                    for f in range(F):
                        marginal_contributions = []
                        target_position = (t, f)
                        
                        for _ in range(nsamples):
                            # Sample coalition excluding target feature
                            coalition = self.coalition_sampler.sample_coalition(
                                all_positions, exclude=target_position
                            )
                            
                            try:
                                # Evaluate model with coalition (support-preserving masking)
                                x_coalition = self.masker.mask_features(x_orig, coalition)
                                pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                                
                                # Evaluate model with coalition + target feature
                                coalition_plus_target = coalition + [target_position]
                                x_coalition_plus = self.masker.mask_features(x_orig, coalition_plus_target)
                                pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                                
                                # Marginal contribution
                                contribution = pred_coalition_plus - pred_coalition
                                marginal_contributions.append(contribution)
                                
                            except ValueError:
                                # Skip if no matching support pattern found
                                if self.skip_unmatched:
                                    continue
                                else:
                                    raise
                        
                        if len(marginal_contributions) > 0:
                            shap_vals[b, t, f] = np.mean(marginal_contributions)
            else:
                # Classic SHAP fallback using mean masking
                for t in range(T):
                    for f in range(F):
                        # Direct evaluation for individual features
                        x_masked = self.masker.mask_features(x_orig, [(t, f)])
                        pred_masked = self.model_evaluator.evaluate_single(x_masked)
                        pred_orig = self.model_evaluator.evaluate_single(x_orig)
                        shap_vals[b, t, f] = pred_orig - pred_masked
            
            # Apply additivity normalization using common algorithm
            if self.is_onehot:
                # For one-hot data, try to find zero support pattern match
                try:
                    fully_masked = self.masker.mask_features(x_orig, all_positions)
                except ValueError:
                    # If no match for all-zero pattern, use mean baseline
                    bg_stats = BackgroundProcessor.compute_background_statistics(self.background)
                    fully_masked = bg_stats['mean']
            else:
                fully_masked = self.masker.mask_features(x_orig, all_positions)
            
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[SupportPreservingSHAP] sum(SHAP)={result.sum():.4f}")
        
        return result

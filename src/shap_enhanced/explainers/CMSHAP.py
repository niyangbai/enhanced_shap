"""
Contextual Masking SHAP (CM-SHAP) Explainer for Sequential Models
=================================================================

Theoretical Explanation
-----------------------

CM-SHAP is a SHAP-style feature attribution method designed for sequential (time series) models.
Instead of masking features with zeros or mean values, CM-SHAP uses context-aware imputation:
masked values are replaced by interpolating between their immediate temporal neighbors (forward/backward average).
This preserves the temporal structure and context, leading to more realistic and faithful perturbations for sequential data.

Key Concepts
^^^^^^^^^^^^

- **Contextual Masking**: When masking a feature at time ``t``, its value is replaced by the average of its neighboring time steps (``t-1`` and ``t+1``). For boundary cases (first or last time step), only the available neighbor is used.
- **Coalition Sampling**: For each feature-time position ``(t, f)``, random coalitions (subsets of all other positions) are sampled. The marginal contribution of ``(t, f)`` is estimated by measuring the change in model output when ``(t, f)`` is added to the coalition.
- **Additivity Normalization**: Attributions are normalized so their sum matches the model output difference between the original and fully-masked input.

Algorithm
---------

1. **Initialization**:
    - Accepts a model and device.
    
2. **Contextual Masking**:
    - For each coalition (subset of features to mask), masked positions are replaced by the average of their immediate temporal neighbors.
    
3. **SHAP Value Estimation**:
    - For each feature-time position ``(t, f)``, repeatedly:
        - Sample a coalition of other positions.
        - Mask the coalition using interpolation.
        - Add ``(t, f)`` to the coalition and compute the change in model output.
        - Average over all sampled coalitions.
    
4. **Normalization**:
    - Normalize attributions so their sum equals the difference between the original and fully-masked model outputs.
"""

from typing import Any, Optional, Union
import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import InterpolationMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs


class ContextualMaskingSHAPExplainer(BaseExplainer):
    r"""
    Contextual Masking SHAP (CM-SHAP) Explainer for Sequential Models

    Estimates SHAP values for sequential inputs by replacing masked feature values with
    interpolated values from neighboring time steps. This context-aware masking strategy
    preserves temporal coherence and enables more realistic feature perturbation in time-series data.

    :param model: Model to explain. Must accept NumPy arrays or PyTorch tensors.
    :type model: Any
    :param device: Device to perform computations on ('cpu' or 'cuda'). Defaults to 'cuda' if available.
    :type device: Optional[str]
    """

    def __init__(self, model: Any, device: Optional[str] = None):
        super().__init__(model, background=None)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.masker = InterpolationMasker(method="linear")
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)

    def shap_values(
        self,
        X: Union[np.ndarray, torch.Tensor],
        nsamples: int = 100,
        check_additivity: bool = True,
        random_seed: int = 42,
        **kwargs
    ) -> np.ndarray:
        r"""
        Estimate SHAP values using contextual (interpolated) masking.

        Each feature-time pair (t, f) is evaluated by sampling coalitions of other
        positions, applying context-aware masking, and averaging the difference
        in model outputs when (t, f) is added to the coalition.

        :param X: Input array of shape (T, F) or (B, T, F)
        :param nsamples: Number of sampled coalitions per position.
        :param check_additivity: Whether to normalize SHAP values to match output difference.
        :param random_seed: Random seed for reproducibility.
        :return: SHAP values with same shape as input.
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
                        
                        # Evaluate model with coalition (interpolation masking)
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
            fully_masked = self.masker.mask_features(x_orig, all_positions)
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[CM-SHAP Additivity] sum(SHAP)={result.sum():.4f}")
        
        return result
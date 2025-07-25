r"""
Adaptive Baseline SHAP (Sparse)
===============================

Theoretical Explanation
-----------------------

Adaptive Baseline SHAP (ABSHAP) is a feature attribution method built upon the SHAP framework.  
It is specifically designed to yield valid, interpretable explanations for both dense (e.g., continuous or tabular)  
and sparse (e.g., categorical or one-hot encoded) input data.

Unlike traditional SHAP methods that use static baselines (such as zeros or means), ABSHAP dynamically samples baselines  
for masked features from real observed background samples. This helps avoid out-of-distribution perturbations,  
which is particularly critical for sparse or categorical data where unrealistic combinations can easily arise.

Key Concepts
^^^^^^^^^^^^

- **Adaptive Masking**: Each feature’s masking method is chosen based on its distribution:
    - Dense/continuous features use the mean value from the background dataset.
    - Sparse/categorical features (e.g., those with >90% zeros) are replaced using values from real background examples.
- **Strategy Selection**: The masking approach can be assigned automatically per feature or manually specified by the user.
- **Valid Perturbations**: All masked samples are guaranteed to lie within the original data distribution, preventing unrealistic inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, number of baselines, masking strategy, and device.
    - Automatically determines a masking strategy per feature or uses a user-specified configuration.
    - Computes mean values for dense-feature masking.

2. **Masking**:
    - For each coalition (a selected subset of features to mask), masked values are replaced with:
        - The feature-wise mean (dense features), or
        - A sampled value from a real background example (sparse features).

3. **SHAP Value Estimation**:
    - For each feature:
        - Randomly sample subsets of other features to mask.
        - For each sampled baseline:
            - Compute model outputs on:
                - Input with selected features masked.
                - Input with selected features plus the current feature masked.
            - Calculate the difference in model outputs.
        - Average these differences to estimate the marginal contribution of the feature.
    - Normalize the resulting attributions so their sum equals the difference between the original and fully-masked model outputs.
"""


import numpy as np
import torch
from typing import Any, Union, Sequence

from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import BaseMasker, MeanMasker, BackgroundSamplingMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor

class AdaptiveFeatureMasker(BaseMasker):
    """Adaptive masker that chooses strategy per feature based on sparsity."""
    
    def __init__(self, background_data, feature_strategies, mean_baseline):
        self.background_data = background_data
        self.feature_strategies = feature_strategies
        self.mean_baseline = mean_baseline
        self.N = len(background_data)
        
    def mask_features(self, x, mask_positions):
        """Apply adaptive masking based on feature-specific strategies."""
        x_masked = x.copy()
        
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                strategy = self.feature_strategies[f]
                
                if strategy == "mean":
                    x_masked[t, f] = self.mean_baseline[t, f]
                elif strategy == "adaptive":
                    # Sample from background data
                    bg_idx = np.random.choice(self.N)
                    x_masked[t, f] = self.background_data[bg_idx, t, f]
                else:
                    raise ValueError(f"Unknown strategy: {strategy}")
                    
        return x_masked


class AdaptiveBaselineSHAPExplainer(BaseExplainer):
    r"""
    Adaptive Baseline SHAP (ABSHAP) Explainer for Dense and Sparse Features.

    Implements a SHAP explainer that adaptively masks features based on their data distribution:
    using mean-based masking for continuous features and sample-based masking for sparse or
    categorical features. This ensures valid perturbations and avoids out-of-distribution artifacts.

    .. note::
        Feature masking strategy can be determined automatically or manually specified.

    .. warning::
        Adaptive masking requires background data and introduces computational overhead.

    :param Callable model: Model to be explained. Should accept PyTorch tensors as input.
    :param Union[np.ndarray, torch.Tensor] background: Background dataset for baseline sampling.
        Shape: (N, F) or (N, T, F).
    :param int n_baselines: Number of baselines to sample per explanation. Default is 10.
    :param Union[str, Sequence[str]] mask_strategy: Either "auto" for detection or list per feature.
    :param str device: PyTorch device identifier, e.g., "cpu" or "cuda". Defaults to auto-detection.
    """

    def __init__(
        self,
        model: Any,
        background: Union[np.ndarray, torch.Tensor],
        n_baselines: int = 10,
        mask_strategy: Union[str, Sequence[str]] = "auto",
        device: str = None
    ):
        super().__init__(model, background)
        
        # Process background data
        self.background = BackgroundProcessor.process_background(background)
        self.N, self.T, self.F = self.background.shape
        self.n_baselines = n_baselines
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Determine masking strategy per feature
        if mask_strategy == "auto":
            # For each feature, if >90% zeros in background, use 'adaptive' masking, else 'mean'
            self.feature_strategies = []
            for f in range(self.F):
                bg_feat = self.background[..., f].flatten()
                zero_frac = np.mean(bg_feat == 0)
                self.feature_strategies.append("adaptive" if zero_frac > 0.9 else "mean")
        elif isinstance(mask_strategy, (list, tuple, np.ndarray)):
            assert len(mask_strategy) == self.F
            self.feature_strategies = list(mask_strategy)
        elif isinstance(mask_strategy, str):
            # All features use the same
            self.feature_strategies = [mask_strategy] * self.F
        else:
            raise ValueError(f"Invalid mask_strategy: {mask_strategy}")
        
        # Compute mean baseline
        bg_stats = BackgroundProcessor.compute_background_statistics(self.background)
        self.mean_baseline = bg_stats['mean']
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)
        self.masker = AdaptiveFeatureMasker(self.background, self.feature_strategies, self.mean_baseline)



    def shap_values(
        self,
        X: Union[np.ndarray, torch.Tensor],
        nsamples: int = 100,
        random_seed: int = 42,
        **kwargs
    ) -> np.ndarray:
        r"""
        Estimates SHAP values for the given input `X` using the ABSHAP algorithm.

        For each feature (t, f), estimates its marginal contribution by comparing model
        outputs with and without the feature masked, averaging over sampled coalitions and baselines.

        :param X: Input samples, shape (B, T, F) or (T, F).
        :param nsamples: Number of masking combinations per feature. Default is 100.
        :param random_seed: Seed for reproducibility. Default is 42.
        :return: SHAP values of shape (T, F) or (B, T, F).
        """
        np.random.seed(random_seed)
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        shap_vals = np.zeros((B, T, F), dtype=np.float32)
        
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
                        
                        # For ABSHAP, evaluate across multiple baselines
                        baseline_contributions = []
                        for _ in range(self.n_baselines):  # Sample multiple baselines
                            # Evaluate model with coalition
                            x_coalition = self.masker.mask_features(x_orig, coalition)
                            pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                            
                            # Evaluate model with coalition + target feature
                            coalition_plus_target = coalition + [target_position]
                            x_coalition_plus = self.masker.mask_features(x_orig, coalition_plus_target)
                            pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                            
                            # Marginal contribution for this baseline
                            contribution = pred_coalition_plus - pred_coalition
                            baseline_contributions.append(contribution)
                        
                        # Average across baselines
                        marginal_contributions.append(np.mean(baseline_contributions))
                    
                    shap_vals[b, t, f] = np.mean(marginal_contributions)
            
            # Apply additivity normalization using common algorithm
            fully_masked = self.masker.mask_features(x_orig, all_positions)
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        return shap_vals[0] if is_single else shap_vals

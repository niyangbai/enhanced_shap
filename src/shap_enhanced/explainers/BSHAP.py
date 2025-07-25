r"""
BShapExplainer: Distribution-Free SHAP for Sequential Models
============================================================

Theoretical Explanation
-----------------------

BShap is a distribution-free variant of the SHAP framework, specifically designed for sequential models such as LSTMs.  
Unlike classical SHAP methods that rely on empirical data (e.g., using mean or sample values as feature baselines),  
BShap masks features using uninformative replacements such as uniform noise, Gaussian noise, or zeros.  
This makes it particularly suitable when the underlying data distribution is unknown, unreliable, or intentionally ignored.

By avoiding assumptions about the data, BShap enables a cleaner interpretation of how a model behaves under  
entirely synthetic perturbations—revealing how features contribute even when removed from their contextual correlations.

Key Concepts
^^^^^^^^^^^^

- **Distribution-Free Masking**: Masked features are replaced with independently sampled values that do not rely on the input data distribution.
- **Masking Strategies**:
  - `'random'`: Sample feature values uniformly at random from a defined range (default).
  - `'noise'`: Add Gaussian noise to masked feature values.
  - `'zero'`: Set masked features to zero.
- **No Data Assumptions**: All masking is performed without drawing from empirical feature distributions.
- **Additivity Normalization**: Feature attributions are normalized so that their sum equals the change in model output  
  between the original and fully-masked input.

Algorithm
---------

1. **Initialization**:
   - Accepts a model, a value range for feature sampling, the number of samples, masking strategy (`'random'`, `'noise'`, `'zero'`), and device context.

2. **Masking**:
    - For each coalition (a subset of features to mask), masked values are replaced by:
        - Random values (uniform), or
        - Gaussian noise, or
        - Zeros, depending on the selected strategy.

3. **SHAP Value Estimation**:
    - For each feature:
        - Randomly select a subset of other features to mask.
        - Compute model output for:
            - The input with the coalition masked.
            - The input with the coalition plus the current feature masked.
        - Record and average the difference in outputs as the estimated contribution.

4. **Normalization**:
    - Scale the attributions so that their sum equals the difference between the original and fully-masked model outputs.

References
----------

- Lundberg, S. M., & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions*. Advances in Neural Information Processing Systems.
- `Distribution-Free SHAP Reference <https://www.tandfonline.com/doi/full/10.1080/02331888.2025.2487853>`_
"""


import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import BaseMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs

class BShapDistributionFreeMasker(BaseMasker):
    """Custom masker for BShap distribution-free strategies."""
    
    def __init__(self, strategy="random", input_range=None):
        self.strategy = strategy
        self.input_range = input_range
    
    def mask_features(self, x, mask_positions):
        """Apply distribution-free masking."""
        x_masked = x.copy()
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                if self.strategy == "random":
                    if self.input_range is not None:
                        mn, mx = self.input_range
                        if isinstance(mn, np.ndarray):
                            x_masked[t, f] = np.random.uniform(mn[f], mx[f])
                        else:
                            x_masked[t, f] = np.random.uniform(mn, mx)
                    else:
                        x_masked[t, f] = np.random.uniform(-1, 1)
                elif self.strategy == "noise":
                    x_masked[t, f] = x[t, f] + np.random.normal(0, 0.5)
                elif self.strategy == "zero":
                    x_masked[t, f] = 0.0
                else:
                    raise ValueError(f"Unknown mask_strategy: {self.strategy}")
        return x_masked


class BShapExplainer(BaseExplainer):
    r"""
    BShap: Distribution-Free SHAP Explainer for Sequential Models

    Implements a SHAP-based attribution method that avoids empirical data distribution assumptions
    by applying synthetic masking strategies (e.g., uniform noise, Gaussian noise, or zero).
    This is useful for evaluating model robustness or interpretability in data-agnostic contexts.

    :param model: Sequence model to explain.
    :param input_range: Tuple of (min, max) or arrays defining per-feature value bounds. Used for random masking.
    :type input_range: tuple or (np.ndarray, np.ndarray)
    :param int n_samples: Number of coalitions sampled per feature.
    :param str mask_strategy: Masking strategy: 'random', 'noise', or 'zero'.
    :param str device: Device identifier, e.g., 'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        input_range=None,
        n_samples=50,
        mask_strategy="random",
        device=None
    ):
        super().__init__(model, background=None)
        self.input_range = input_range
        self.n_samples = n_samples
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.masker = BShapDistributionFreeMasker(mask_strategy, input_range)
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)


    def shap_values(
        self,
        X,
        nsamples=None,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        r"""
        Compute SHAP values using distribution-free perturbations.

        Estimates marginal feature contributions by averaging differences between model outputs
        under masked coalitions. Uses synthetic masking based on the configured strategy
        without any reliance on background data statistics.

        Final attributions are normalized to satisfy the SHAP additivity constraint.

        :param X: Input of shape (T, F) or (B, T, F)
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of coalition samples per feature (defaults to self.n_samples).
        :param bool check_additivity: Print diagnostic message for SHAP sum vs. model delta.
        :param int random_seed: Seed for reproducibility.
        :return: SHAP values of shape (T, F) or (B, T, F)
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        if nsamples is None:
            nsamples = self.n_samples
            
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        
        all_positions = create_all_positions(T, F)
        
        for b in range(B):
            x = X_processed[b]
            
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
                        x_coalition = self.masker.mask_features(x, coalition)
                        pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                        
                        # Evaluate model with coalition + target feature
                        coalition_plus_target = coalition + [target_position]
                        x_coalition_plus = self.masker.mask_features(x, coalition_plus_target)
                        pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                        
                        # Marginal contribution
                        contribution = pred_coalition_plus - pred_coalition
                        marginal_contributions.append(contribution)
                    
                    shap_vals[b, t, f] = np.mean(marginal_contributions)
            
            # Apply additivity normalization using common algorithm
            fully_masked = self.masker.mask_features(x, all_positions)
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[BShap] sum(SHAP)={result.sum():.4f} (should match model diff)")
        
        return result

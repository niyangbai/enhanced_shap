"""
ER-SHAP: Ensemble of Random SHAP Explainer
==========================================

Theoretical Explanation
-----------------------

ER-SHAP is a computationally efficient, ensemble-based approximation of Shapley values, designed for  
both sequential and tabular models. Instead of exhaustively enumerating all possible coalitions,  
ER-SHAP repeatedly samples random subsets of feature–timestep positions and estimates their  
marginal contributions to model output.

This stochastic approach significantly accelerates SHAP estimation while maintaining interpretability,  
especially in high-dimensional or temporal settings. ER-SHAP also allows prior knowledge (e.g., feature importance)  
to guide coalition sampling through weighted schemes.

Key Concepts
^^^^^^^^^^^^

- **Random Coalition Sampling**:  
    For each position \((t, f)\), sample coalitions \( C \subseteq (T \times F) \setminus \{(t, f)\} \)  
    and estimate the marginal contribution of \((t, f)\) by measuring its impact on model output.

- **Weighted Sampling**:  
    Coalition sampling can be uniform or weighted based on prior feature importance scores  
    or positional frequency, allowing informed, efficient sampling.

- **Flexible Masking**:  
    Masked features are imputed using:
        - Zeros (hard masking).
        - Feature-wise means from the background dataset (soft masking).

- **Additivity Normalization**:  
    Final attributions are scaled so that their sum matches the model output difference  
    between the original and fully-masked input.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset for imputation, number of sampled coalitions,  
        masking strategy (`'zero'` or `'mean'`), weighting scheme, optional feature importance, and device context.

2. **Coalition Sampling**:
    - For each feature–timestep pair \((t, f)\):
        - Sample coalitions \( C \subseteq (T \times F) \setminus \{(t, f)\} \), either uniformly or using weights.
        - For each coalition:
            - Impute the coalition \( C \) in the input.
            - Impute the coalition \( C \cup \{(t, f)\} \).
            - Compute the model output difference.
        - Average these differences to estimate the marginal contribution of \((t, f)\).

3. **Normalization**:
    - Scale the final attributions so that their total equals the difference in model output  
        between the original input and a fully-masked baseline.
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import WeightedCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import ZeroMasker, MeanMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor


class ERSHAPExplainer(BaseExplainer):
    """
    ER-SHAP: Ensemble of Random SHAP Explainer

    An efficient approximation of Shapley values using random coalition sampling over
    time-feature positions. Supports uniform and weighted sampling strategies and flexible
    masking (zero or mean) to generate perturbed inputs.

    :param model: Model to explain, compatible with PyTorch tensors.
    :type model: Any
    :param background: Background dataset for mean imputation; shape (N, T, F).
    :type background: np.ndarray or torch.Tensor
    :param n_coalitions: Number of coalitions to sample per (t, f) position.
    :type n_coalitions: int
    :param mask_strategy: Masking method: 'zero' or 'mean'.
    :type mask_strategy: str
    :param weighting: Sampling scheme: 'uniform', 'frequency', or 'importance'.
    :type weighting: str
    :param feature_importance: Prior feature importances for weighted sampling; shape (T, F).
    :type feature_importance: Optional[np.ndarray]
    :param device: Device identifier, 'cpu' or 'cuda'.
    :type device: str
    """
    def __init__(
        self,
        model,
        background,
        n_coalitions=100,
        mask_strategy="mean",
        weighting="uniform",
        feature_importance=None,
        device=None
    ):
        super().__init__(model, background)
        self.n_coalitions = n_coalitions
        self.mask_strategy = mask_strategy
        self.weighting = weighting
        self.feature_importance = feature_importance
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process background data
        self.background = BackgroundProcessor.process_background(background)
        
        # Initialize common algorithm components
        if weighting == "uniform":
            from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler
            self.coalition_sampler = RandomCoalitionSampler()
        else:
            self.coalition_sampler = WeightedCoalitionSampler(feature_importance)
        
        if mask_strategy == "zero":
            self.masker = ZeroMasker()
        elif mask_strategy == "mean":
            bg_stats = BackgroundProcessor.compute_background_statistics(self.background)
            self.masker = MeanMasker(bg_stats['mean'])
        else:
            raise ValueError(f"Unknown mask_strategy: {mask_strategy}")
        
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)

    def shap_values(
        self,
        X,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        r"""
        Compute SHAP values via random coalition sampling.

        For each position (t, f), sample coalitions of other positions,
        compute marginal contributions, and average over samples.
        Attributions are normalized to satisfy:

        .. math::
            \sum_{t=1}^T \sum_{f=1}^F \phi_{t,f} \approx f(x) - f(x_{masked})

        :param X: Input array or tensor of shape (T, F) or (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :param check_additivity: Whether to apply normalization for additivity.
        :type check_additivity: bool
        :param random_seed: Seed for reproducibility.
        :type random_seed: int
        :return: SHAP values of shape (T, F) or (B, T, F).
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
                    
                    for _ in range(self.n_coalitions):
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
            print(f"[ERSHAP Additivity] sum(SHAP)={result.sum():.4f}")
        
        return result
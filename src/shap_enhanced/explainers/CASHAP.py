"""
CASHAP: Coalition-Aware SHAP Explainer
======================================

Theoretical Explanation
-----------------------

CASHAP (Coalition-Aware SHAP) is a Shapley value estimation framework tailored for models that process sequential or structured inputs, such as LSTMs.  
Unlike classical SHAP methods that treat features independently, CASHAP considers **feature-time pairs**—enabling attribution of both spatial and temporal components.  

By explicitly sampling coalitions (subsets) of feature-time pairs and measuring marginal contributions, CASHAP provides granular, context-aware explanations.  
It also supports multiple imputation strategies to ensure the perturbed inputs remain valid and interpretable.

Key Concepts
^^^^^^^^^^^^

- **Coalition Sampling**: For every feature-time pair \((t, f)\), random subsets of all other positions are sampled.  
    The contribution of \((t, f)\) is assessed by adding it to each coalition and measuring the change in model output.
- **Masking/Imputation Strategies**:
    - **Zero masking**: Replace masked values with zero.
    - **Mean imputation**: Use feature-wise means from background data.
    - **Custom imputers**: Support for user-defined imputation functions.
- **Model-Agnostic & Domain-General**: While ideal for time-series and sequential models, CASHAP can also be applied to tabular data  
    wherever structured coalition masking is appropriate.
- **Additivity Normalization**: Attribution scores are scaled such that their total sum equals the difference in model output  
    between the original input and a fully-masked version.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background data for imputation, masking strategy, optional custom imputer, and device context.

2. **Coalition Sampling**:
    - For each feature-time pair \((t, f)\):
        - Sample coalitions \( C \subseteq (T \times F) \setminus \{(t, f)\} \).
        - For each coalition \( C \):
            - Impute features in \( C \) using the chosen strategy.
            - Impute features in \( C \cup \{(t, f)\} \).
            - Compute and record the model output difference.

3. **Attribution Estimation**:
    - Average the output differences across coalitions to estimate the marginal contribution of \((t, f)\).

4. **Normalization**:
    - Normalize attributions so that their total matches the difference between the model's prediction  
        on the original and the fully-masked input.
"""


from typing import Any, Optional, Union
from collections.abc import Callable
import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import ZeroMasker, MeanMasker, BaseMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs

class CASHAPCustomMasker(BaseMasker):
    """Custom masker for CASHAP with callable imputer support."""
    
    def __init__(self, imputer_func: Callable):
        self.imputer_func = imputer_func
    
    def mask_features(self, x, mask_positions):
        """Apply custom imputation using provided callable."""
        return self.imputer_func(x, mask_positions)


class CoalitionAwareSHAPExplainer(BaseExplainer):
    """
    Coalition-Aware SHAP (CASHAP) Explainer

    Estimates Shapley values for models processing structured inputs (e.g., time-series, sequences)
    by sampling coalitions of feature-time pairs and computing their marginal contributions
    using various imputation strategies.

    :param model: Model to be explained.
    :type model: Any
    :param background: Background data used for mean imputation strategy.
    :type background: Optional[np.ndarray or torch.Tensor]
    :param str mask_strategy: Strategy for imputing/masking feature-time pairs.
                              Options: 'zero', 'mean', or 'custom'.
    :param imputer: Custom callable for imputation. Required if `mask_strategy` is 'custom'.
    :type imputer: Optional[Callable]
    :param device: Device on which computation runs. Defaults to 'cuda' if available.
    :type device: Optional[str]
    """

    def __init__(
        self,
        model: Any,
        background: Optional[Union[np.ndarray, torch.Tensor]] = None,
        mask_strategy: str = "zero",
        imputer: Optional[Callable] = None,
        device: Optional[str] = None
    ):
        super().__init__(model, background)
        self.mask_strategy = mask_strategy
        self.imputer = imputer
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize common algorithm components
        self.coalition_sampler = RandomCoalitionSampler()
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)
        
        # Setup masker based on strategy
        if mask_strategy == "zero":
            self.masker = ZeroMasker()
        elif mask_strategy == "mean":
            if background is None:
                raise ValueError("Mean imputation requires background data.")
            # Process background data
            from shap_enhanced.algorithms.data_processing import BackgroundProcessor
            bg_processed = BackgroundProcessor.process_background(background)
            bg_stats = BackgroundProcessor.compute_background_statistics(bg_processed)
            self.masker = MeanMasker(bg_stats['mean'])
        elif mask_strategy == "custom":
            if imputer is None:
                raise ValueError("Custom imputer must be provided for custom mask strategy.")
            self.masker = CASHAPCustomMasker(imputer)
        else:
            raise ValueError(f"Unknown mask_strategy: {mask_strategy}")


    def shap_values(
        self, 
        X: Union[np.ndarray, torch.Tensor], 
        nsamples: int = 100,
        coalition_size: Optional[int] = None,
        mask_strategy: Optional[str] = None,
        check_additivity: bool = True,
        random_seed: int = 42,
        **kwargs
    ) -> np.ndarray:
        """
        Compute CASHAP Shapley values for structured inputs via coalition-aware sampling.

        For each feature-time pair \((t, f)\), randomly sample coalitions excluding \((t, f)\),
        compute model outputs with and without the pair added, and average the marginal contributions.
        Attribution values are normalized so their total matches the model output difference
        between the original and fully-masked input.

        :param X: Input sample of shape (T, F) or batch (B, T, F).
        :param nsamples: Number of coalitions sampled per (t, f).
        :param coalition_size: Fixed size of sampled coalitions. If None, varies randomly.
        :param mask_strategy: Override default masking strategy (not implemented in refactor).
        :param check_additivity: Print diagnostic SHAP sum vs. model delta.
        :param random_seed: Seed for reproducibility.
        :return: SHAP values of shape (T, F) or (B, T, F).
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
                        if coalition_size is not None:
                            # Fixed coalition size
                            size_range = (coalition_size, coalition_size)
                        else:
                            # Variable coalition size
                            size_range = None
                            
                        coalition = self.coalition_sampler.sample_coalition(
                            all_positions, exclude=target_position, size_range=size_range
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
            print(f"[CASHAP Additivity] sum(SHAP)={result.sum():.4f}")
        
        return result

if __name__ == "__main__":
    import torch
    import torch.nn as nn
    import numpy as np

    # --- Dummy LSTM model for demo ---
    class DummyLSTM(nn.Module):
        def __init__(self, input_dim=3, hidden_dim=8, output_dim=1):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
            self.fc = nn.Linear(hidden_dim, output_dim)

        def forward(self, x):
            # Ensure input is float tensor
            if not torch.is_tensor(x):
                x = torch.tensor(x, dtype=torch.float32)
            x = x.float()
            # x: (B, T, F)
            out, _ = self.lstm(x)
            # Use last time step's output
            out = self.fc(out[:, -1, :])
            return out.squeeze(-1)  # (B,)

    # --- Generate synthetic data ---
    np.random.seed(0)
    torch.manual_seed(0)
    B, T, F = 2, 5, 3
    train_X = np.random.normal(0, 1, (20, T, F)).astype(np.float32)
    test_X = np.random.normal(0, 1, (B, T, F)).astype(np.float32)

    # --- Initialize model and explainer ---
    model = DummyLSTM(input_dim=F, hidden_dim=8, output_dim=1)
    model.eval()

    explainer = CoalitionAwareSHAPExplainer(
        model=model,
        background=train_X,
        mask_strategy="mean"
    )

    # --- Compute SHAP values ---
    shap_vals = explainer.shap_values(
        test_X,           # (B, T, F)
        nsamples=10,      # small for demo, increase for quality
        coalition_size=4, # mask 4 pairs at a time
        check_additivity=True
    )

    print("SHAP values shape:", shap_vals.shape)
    print("First sample SHAP values:\n", shap_vals[0])
# src/shap_enhanced/datasets/synthetic_tabular.py

import numpy as np
import torch
from typing import Callable, Optional, Tuple

def generate_tabular_data(
    n_samples: int,
    n_features: int,
    target_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    noise_std: float = 0.1,
    feature_distribution: Callable[[int, int], np.ndarray] = np.random.randn
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic tabular data with customizable relationships and distributions.

    Args:
        n_samples (int): Number of samples to generate.
        n_features (int): Number of features to generate.
        target_function (Callable[[np.ndarray], np.ndarray], optional): 
            A function to compute the target variable `y` from the features `X`.
            If None, a default non-linear function is used.
        noise_std (float): Standard deviation of Gaussian noise added to the target.
        feature_distribution (Callable[[int, int], np.ndarray]): 
            A function to generate the feature matrix `X`. Defaults to standard normal distribution.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature matrix `X` 
        and the target vector `y` as PyTorch tensors.
    """
    # Generate feature matrix
    X = feature_distribution(n_samples, n_features)
    
    # Default target function if none is provided
    if target_function is None:
        target_function = lambda X: np.sum(X**2, axis=1)
    
    # Compute target variable
    y = target_function(X) + np.random.randn(n_samples) * noise_std
    
    # Convert to PyTorch tensors
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

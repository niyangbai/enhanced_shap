# src/shap_enhanced/datasets/synthetic_sparse.py

import numpy as np
import torch
from typing import Callable, Optional, Tuple

def generate_sparse_data(
    n_samples: int,
    n_features: int,
    sparsity: float = 0.8,
    noise_std: float = 0.0,
    target_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    random_seed: Optional[int] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Generate synthetic sparse data.

    Args:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        sparsity (float): Fraction of elements to set to zero (0 <= sparsity <= 1).
        noise_std (float): Standard deviation of Gaussian noise to add to the target.
        target_function (Callable[[np.ndarray], np.ndarray], optional): 
            Function to generate target values from features. Defaults to sum of features.
        random_seed (int, optional): Random seed for reproducibility.

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Feature matrix (X) and target vector (y).
    """
    if not (0 <= sparsity <= 1):
        raise ValueError("Sparsity must be between 0 and 1.")

    if random_seed is not None:
        np.random.seed(random_seed)

    # Generate random data
    X = np.random.randn(n_samples, n_features)

    # Introduce sparsity
    mask = np.random.rand(n_samples, n_features) < sparsity
    X[mask] = 0

    # Generate target values
    if target_function is None:
        target_function = lambda x: np.sum(x, axis=1)  # Default to sum of features
    y = target_function(X)

    # Add noise to the target
    if noise_std > 0:
        y += np.random.normal(0, noise_std, size=y.shape)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X, dtype=torch.float32)
    y_tensor = torch.tensor(y, dtype=torch.float32)

    return X_tensor, y_tensor

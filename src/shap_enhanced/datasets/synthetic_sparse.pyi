"""
This module provides functionality to generate synthetic sparse datasets for testing and experimentation.
It includes a function to create feature matrices with a specified level of sparsity and optional noise,
along with corresponding target values generated using a customizable target function.
"""

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
    """Generate synthetic sparse data.

    :param int n_samples: Number of samples.
    :param int n_features: Number of features.
    :param float sparsity: Fraction of elements to set to zero (0 <= sparsity <= 1), defaults to 0.8
    :param float noise_std: Standard deviation of Gaussian noise to add to the target, defaults to 0.0
    :param Optional[Callable[[np.ndarray], np.ndarray]] target_function: unction to generate target values from features. Defaults to sum of features, defaults to None
    :param Optional[int] random_seed: Random seed for reproducibility, defaults to None
    :return Tuple[torch.Tensor, torch.Tensor]: Feature matrix (X) and target vector (y)
    """

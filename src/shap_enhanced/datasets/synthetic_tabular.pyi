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
    """Generate synthetic tabular data with customizable relationships and distributions.

    :param int n_samples: Number of samples to generate.
    :param int n_features: Number of features to generate.
    :param Optional[Callable[[np.ndarray], np.ndarray]] target_function: A function to compute the target variable `y` from the features `X`.
            If None, a default non-linear function is used, defaults to None
    :param float noise_std: Standard deviation of Gaussian noise added to the target, defaults to 0.1
    :param Callable[[int, int], np.ndarray] feature_distribution: A function to generate the feature matrix `X`. Defaults to standard normal distribution, defaults to np.random.randn
    :return Tuple[torch.Tensor, torch.Tensor]: A tuple containing the feature matrix `X` 
        and the target vector `y` as PyTorch tensors.
    """

import numpy as np
import torch
from typing import Callable, Optional, Tuple

def generate_sequential_data(
    n_samples: int,
    n_timesteps: int,
    n_features: int,
    pattern_function: Optional[Callable[[int, int, int], np.ndarray]] = None,
    noise_std: float = 0.1,
    target_function: Optional[Callable[[np.ndarray], np.ndarray]] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate synthetic sequential data with customizable patterns and targets.

    :param int n_samples: Number of samples to generate.
    :param int n_timesteps: Number of timesteps per sample.
    :param int n_features: Number of features per sample.
    :param Optional[Callable[[int, int, int], np.ndarray]] pattern_function: Function to generate the base pattern for the data.
            Should accept (n_timesteps, n_features, sample_index) and return a 2D array of shape (n_timesteps, n_features).
            Defaults to a sinusoidal pattern, defaults to None
    :param float noise_std: Standard deviation of Gaussian noise to add to the data, defaults to 0.1
    :param Optional[Callable[[np.ndarray], np.ndarray]] target_function: Function to generate target values from the data.
            Should accept the full data array (n_samples, n_timesteps, n_features) and return a 1D array of shape (n_samples,).
            Defaults to summing over timesteps and features, defaults to None
    :return Tuple[torch.Tensor, torch.Tensor]: A tuple containing the generated data (X) and targets (y).
    """
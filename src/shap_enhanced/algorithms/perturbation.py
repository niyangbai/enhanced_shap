import torch
import random
from typing import List

def mask_features(X: torch.Tensor, features: List[int], mask_value: float = 0.0) -> torch.Tensor:
    """Mask specified features."""
    X_masked = X.clone()
    X_masked[:, features] = mask_value
    return X_masked

def mask_timesteps(X: torch.Tensor, timesteps: List[int], mask_value: float = 0.0) -> torch.Tensor:
    """Mask specified timesteps."""
    X_masked = X.clone()
    X_masked[:, timesteps, :] = mask_value
    return X_masked

def random_mask(X: torch.Tensor, prob: float = 0.5, mask_value: float = 0.0) -> torch.Tensor:
    """Randomly mask features."""
    mask = (torch.rand_like(X) > prob).float()
    return X * mask + (1 - mask) * mask_value

def mask_time_window(X: torch.Tensor, window_size: int, mask_value: float = 0.0) -> torch.Tensor:
    """Mask a continuous window of timesteps.

    :param torch.Tensor X: Input tensor (batch, time, features)
    :param int window_size: Size of window to mask
    :param float mask_value: Mask value
    :return torch.Tensor: Masked tensor
    """
    batch_size, time_steps, _ = X.shape
    X_masked = X.clone()

    for i in range(batch_size):
        start = torch.randint(0, time_steps - window_size + 1, (1,)).item()
        X_masked[i, start:start+window_size, :] = mask_value

    return X_masked

def perturb_sequence_with_noise(X: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """Add structured noise to a sequence.

    :param torch.Tensor X: Input tensor (batch, time, features)
    :param float noise_level: Std deviation of noise
    :return torch.Tensor: Noisy tensor
    """
    noise = torch.randn_like(X) * noise_level
    return X + noise

def mask_random_features(X: torch.Tensor, n_features_to_mask: int, mask_value: float = 0.0) -> torch.Tensor:
    """Randomly mask n features for each sample.

    :param torch.Tensor X: Input tensor (batch, features).
    :param int n_features_to_mask: Number of features to mask per sample.
    :param float mask_value: Mask value.
    :return torch.Tensor: Masked tensor.
    """
    X_masked = X.clone()
    batch_size, n_features = X.shape
    for i in range(batch_size):
        features = random.sample(range(n_features), n_features_to_mask)
        X_masked[i, features] = mask_value
    return X_masked

def mask_feature_groups(X: torch.Tensor, groups: List[List[int]], group_indices: List[int], mask_value: float = 0.0) -> torch.Tensor:
    """Mask specific groups of features.

    :param torch.Tensor X: Input tensor (batch, features).
    :param List[List[int]] groups: Feature groups.
    :param List[int] group_indices: Indices of groups to mask.
    :param float mask_value: Mask value.
    :return torch.Tensor: Masked tensor.
    """
    features_to_mask = [idx for g in group_indices for idx in groups[g]]
    return mask_features(X, features_to_mask, mask_value)
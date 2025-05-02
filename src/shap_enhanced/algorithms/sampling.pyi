"""
This module provides various sampling methods for subsets, timesteps, and feature groups.
It is designed to support tasks that require random or balanced sampling of indices or groups,
such as feature perturbation or time series analysis.

Functions:
- sample_subsets: Samples random subsets of feature indices.
- sample_balanced_subsets: Samples subsets with balanced sizes (small, medium, large).
- sample_timesteps: Samples random timesteps from a sequence.
- sample_feature_subsets: Samples random subsets for feature perturbations.
- sample_feature_groups: Samples random groups of features based on predefined groups.
"""

from typing import List

def sample_subsets(n_features: int, nsamples: int) -> list:
    """Sample random subsets of feature indices.

    :param int n_features: Total number of features.
    :param int nsamples: Number of subsets to sample.
    :return list: List of subsets (as sets).
    """
    ...
    
def sample_balanced_subsets(n_features: int, nsamples: int) -> list:
    """Sample subsets with size balanced across small, medium, large.

    :param int n_features: Total number of features.
    :param int nsamples: Number of samples.
    :return list: List of subsets.
    """
    ...
    
def sample_timesteps(total_timesteps: int, nsamples: int) -> list:
    """Sample random timesteps.

    :param int total_timesteps: Number of total timesteps
    :param int nsamples: Number of samples
    :return list: List of sampled timestep indices
    """
    ...
    
def sample_feature_subsets(n_features: int, nsamples: int) -> list:
    """Sample random subsets for feature perturbations.

    :param int n_features: Total features.
    :param int nsamples: Number of samples.
    :return list: List of feature indices.
    """
    ...
    
def sample_feature_groups(groups: List[List[int]], nsamples: int) -> list:
    """Sample random groups of features.

    :param List[List[int]] groups: Feature groups.
    :param int nsamples: Number of samples.
    :return list: List of group indices.
    """
    ...
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
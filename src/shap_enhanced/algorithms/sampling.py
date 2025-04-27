import random
from typing import List

def sample_subsets(n_features: int, nsamples: int) -> list:
    """Sample random subsets of feature indices.

    :param int n_features: Total number of features.
    :param int nsamples: Number of subsets to sample.
    :return list: List of subsets (as sets).
    """
    all_features = list(range(n_features))
    subsets = []
    for _ in range(nsamples):
        subset_size = random.randint(0, n_features - 1)
        subset = set(random.sample(all_features, subset_size))
        subsets.append(subset)
    return subsets

def sample_balanced_subsets(n_features: int, nsamples: int) -> list:
    """Sample subsets with size balanced across small, medium, large.

    :param int n_features: Total number of features.
    :param int nsamples: Number of samples.
    :return list: List of subsets.
    """
    all_features = list(range(n_features))
    subsets = []
    for _ in range(nsamples):
        subset_size = random.choice([1, n_features//2, n_features-1])
        subset = set(random.sample(all_features, subset_size))
        subsets.append(subset)
    return subsets

def sample_timesteps(total_timesteps: int, nsamples: int) -> list:
    """Sample random timesteps.

    :param int total_timesteps: Number of total timesteps
    :param int nsamples: Number of samples
    :return list: List of sampled timestep indices
    """
    return [random.randint(0, total_timesteps - 1) for _ in range(nsamples)]

def sample_feature_subsets(n_features: int, nsamples: int) -> list:
    """Sample random subsets for feature perturbations.

    :param int n_features: Total features.
    :param int nsamples: Number of samples.
    :return list: List of feature indices.
    """
    return [random.sample(range(n_features), random.randint(1, n_features)) for _ in range(nsamples)]

def sample_feature_groups(groups: List[List[int]], nsamples: int) -> list:
    """Sample random groups of features.

    :param List[List[int]] groups: Feature groups.
    :param int nsamples: Number of samples.
    :return list: List of group indices.
    """
    n_groups = len(groups)
    return [random.sample(range(n_groups), random.randint(1, n_groups)) for _ in range(nsamples)]
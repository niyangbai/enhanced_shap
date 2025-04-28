def shapley_kernel_weights(n: int, s: int) -> float:
    """Compute KernelSHAP standard weight for subset size s.

    :param int n: Total features.
    :param int s: Size of subset.
    :return float: Kernel weight.
    """

def entropy_kernel_weights(n: int, s: int) -> float:
    """Compute entropy-weighted kernel for subset size s.

    Penalizes highly unbalanced coalitions.

    :param int n: Total features.
    :param int s: Size of subset.
    :return float: Entropy kernel weight.
    """

def uniform_kernel_weights(n_features: int, subset_size: int) -> float:
    """Compute the uniform kernel weight for a subset.

    In uniform kernel weighting, all subsets are considered to have the same importance, 
    meaning that each subset of features has an equal weight.

    :param int n_features: Total number of features in the dataset.
    :param int subset_size: Size of the subset being considered.
    :return float: Weight for this subset.
    """
    # In the case of uniform kernel, every subset has the same weight
    return 1.0
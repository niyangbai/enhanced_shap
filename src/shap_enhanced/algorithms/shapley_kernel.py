import math

def shapley_kernel_weights(n: int, s: int) -> float:
    """Compute KernelSHAP standard weight for subset size s.

    :param int n: Total features.
    :param int s: Size of subset.
    :return float: Kernel weight.
    """
    if s == 0 or s == n:
        return 1000000.0  # Large weight for full/empty sets
    return (n - 1) / (math.comb(n, s) * s * (n - s))

def entropy_kernel_weights(n: int, s: int) -> float:
    """Compute entropy-weighted kernel for subset size s.

    Penalizes highly unbalanced coalitions.

    :param int n: Total features.
    :param int s: Size of subset.
    :return float: Entropy kernel weight.
    """
    if s == 0 or s == n:
        return 1000000.0
    p = s / n
    entropy = - (p * math.log(p + 1e-8) + (1 - p) * math.log(1 - p + 1e-8))
    return (1 / entropy) if entropy > 0 else 1000000.0

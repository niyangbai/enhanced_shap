import numpy as np

def generate_synthetic_seqregression(seq_len=10, n_features=3, n_samples=200, seed=0):
    """Generate synthetic data for sequence regression."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, seq_len, n_features))
    y = np.sin(X[:, :, 0].sum(axis=1)) + 0.1 * rng.standard_normal(n_samples)
    return X, y

def generate_synthetic_tabular(
    n_samples=500,
    n_features=5,
    sparse=True,
    model_type="nonlinear",  # "linear" or "nonlinear"
    sparsity=0.85,           # Default to 85% zeros if sparse
    random_seed=42
):
    """
    Generate synthetic tabular data with controllable sparsity.

    Args:
        n_samples (int): Number of samples.
        n_features (int): Number of features.
        sparse (bool): Whether to zero out random entries.
        model_type (str): "linear" or "nonlinear".
        sparsity (float): Fraction of zeros in X if sparse.
        random_seed (int): For reproducibility.

    Returns:
        X (np.ndarray): Feature matrix.
        y (np.ndarray): Target vector.
        true_w (np.ndarray): True weights used.
    """
    rng = np.random.default_rng(random_seed)
    X = rng.standard_normal((n_samples, n_features))
    if sparse:
        mask = rng.uniform(0, 1, size=X.shape) < sparsity
        X[mask] = 0.0
        # Optional: print actual sparsity
        # print(f"Sparsity: {(X == 0).mean():.2%} of elements are zero.")
    true_w = rng.uniform(-2, 3, size=n_features)
    if model_type == "linear":
        y = X.dot(true_w)
    else:
        y = X.dot(true_w)
        y = np.tanh(y) + 0.1 * (y ** 2) + 0.1 * rng.normal(size=n_samples)
    return X, y, true_w

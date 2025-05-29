import numpy as np

def generate_synthetic_seqregression(seq_len=10, n_features=3, n_samples=200, seed=0):
    """Generate synthetic data for sequence regression."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, seq_len, n_features))
    y = np.sin(X[:, :, 0].sum(axis=1)) + 0.1 * rng.standard_normal(n_samples)
    return X, y

"""
Synthetic Data Generators for Regression Benchmarks
===================================================

Overview
--------

This module provides utility functions for generating synthetic datasets
for benchmarking SHAP and other model explainability techniques in both sequential
and tabular regression settings.

Two types of data generators are included:

- **Sequential Regression Generator**: Produces multivariate time-series inputs with
  sinusoidal targets based on cumulative temporal signals.

- **Tabular Regression Generator**: Supports sparse or dense tabular inputs, with configurable linear
  or nonlinear target mappings. Also outputs ground-truth feature weights.

Key Functions
^^^^^^^^^^^^^

- **generate_synthetic_seqregression**:
  Creates a synthetic sequence-to-scalar regression dataset using sinusoidal logic.

- **generate_synthetic_tabular**:
  Produces sparse or dense tabular data with a tunable regression function.
  Returns both the dataset and the true underlying feature weights used to generate targets.

Use Case
--------

These generators are useful for:
- Testing SHAP explainers on known attribution structures.
- Evaluating sensitivity to sparsity, nonlinearity, or sequence length.
- Building reproducible benchmarks for model interpretability.

Example
-------

.. code-block:: python

    # Sequence data
    X_seq, y_seq = generate_synthetic_seqregression(seq_len=12, n_features=4, n_samples=100)

    # Tabular data with sparsity and nonlinearity
    X_tab, y_tab, w = generate_synthetic_tabular(n_samples=200, n_features=6, sparse=True, model_type="nonlinear")
"""

import numpy as np

__all__ = ["generate_synthetic_seqregression", "generate_synthetic_tabular"]


def generate_synthetic_seqregression(seq_len=10, n_features=3, n_samples=200, seed=0):
    r"""
    Generate synthetic multivariate time-series data for sequence-to-scalar regression.

    Each target is constructed using a sinusoidal function over the cumulative sum
    of the first feature across timesteps.

    .. math::
        y_i = \sin\left(\sum_{t=1}^T x_{it1}\right) + \epsilon_i,\quad \epsilon_i \sim \mathcal{N}(0, 0.1^2)

    :param int seq_len: Length of the time series (T).
    :param int n_features: Number of features per timestep (F).
    :param int n_samples: Number of sequences to generate (N).
    :param int seed: Random seed for reproducibility.
    :return: Tuple of input sequences and target values.
    :rtype: (np.ndarray, np.ndarray)
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, seq_len, n_features))
    y = np.sin(X[:, :, 0].sum(axis=1)) + 0.1 * rng.standard_normal(n_samples)
    return X, y


def generate_synthetic_tabular(
    n_samples=500,
    n_features=5,
    sparse=True,
    model_type="nonlinear",  # "linear" or "nonlinear"
    sparsity=0.85,  # Default to 85% zeros if sparse
    random_seed=42,
):
    r"""
    Generate synthetic tabular data with optional sparsity and a configurable regression function.

    Supports both linear and nonlinear target mappings. Optionally zeroes out random entries to simulate sparsity.

    .. math::
        y = \begin{cases}
        X w & \text{(linear)} \\
        \tanh(X w) + 0.1 (X w)^2 + \mathcal{N}(0, 0.1^2) & \text{(nonlinear)}
        \end{cases}

    :param int n_samples: Number of data samples.
    :param int n_features: Number of features.
    :param bool sparse: Whether to randomly zero entries to simulate sparsity.
    :param str model_type: Type of regression model ("linear" or "nonlinear").
    :param float sparsity: Proportion of elements to set to zero if sparse.
    :param int random_seed: Seed for reproducibility.
    :return: Feature matrix, target vector, and true coefficient weights.
    :rtype: (np.ndarray, np.ndarray, np.ndarray)
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
        y = np.tanh(y) + 0.1 * (y**2) + 0.1 * rng.normal(size=n_samples)
    return X, y, true_w

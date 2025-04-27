# src/shap_enhanced/algorithms/approximation.py

import torch
from typing import Callable, List, Optional

def monte_carlo_expectation(f: Callable, X: torch.Tensor, nsamples: int = 100) -> torch.Tensor:
    """Monte Carlo estimate of E[f(X)].

    :param Callable f: Function to evaluate.
    :param torch.Tensor X: Input samples (batch, features).
    :param int nsamples: Number of samples.
    :return torch.Tensor: Expectation estimate.
    """
    outputs = []
    for _ in range(nsamples):
        idx = torch.randint(0, X.shape[0], (1,))
        x_sample = X[idx]
        outputs.append(f(x_sample))
    return torch.stack(outputs).mean(dim=0)

def joint_marginal_expectation(model: Callable, x: torch.Tensor, S: List[int],
                                background: torch.Tensor, nsamples: int = 50,
                                target_index: Optional[int] = 0) -> torch.Tensor:
    """Estimate E[f(X) | X_S = x_S] where S is subset of features.

    :param Callable model: Model to evaluate.
    :param torch.Tensor x: Single input point (1, features).
    :param List[int] S: Indices of features to fix.
    :param torch.Tensor background: Background dataset.
    :param int nsamples: Number of samples.
    :param Optional[int] target_index: Output index.
    :return torch.Tensor: Estimated expectation (single value).
    """
    device = x.device
    n_features = x.shape[1]

    idx = torch.randint(0, background.shape[0], (nsamples,), device=device)
    sampled = background[idx].clone()

    # Overwrite features S
    for s in S:
        sampled[:, s] = x[:, s].expand(nsamples)

    outputs = model(sampled)

    if outputs.ndim > 1 and outputs.shape[1] > 1 and target_index is not None:
        outputs = outputs[:, target_index]
    elif outputs.ndim > 1 and outputs.shape[1] > 1:
        outputs = outputs[:, 0]

    return outputs.mean()

def conditional_marginal_expectation(model: Callable, x: torch.Tensor, fixed_features: List[int],
                                     background: torch.Tensor, nsamples: int = 50,
                                     target_index: Optional[int] = 0) -> torch.Tensor:
    """Estimate conditional E[f(X) | X_fixed = x_fixed].

    (Alias for joint marginal but clearer when fixing several features.)

    :param Callable model: Model to evaluate.
    :param torch.Tensor x: Single input point (1, features).
    :param List[int] fixed_features: Indices to fix.
    :param torch.Tensor background: Background dataset.
    :param int nsamples: Number of samples.
    :param Optional[int] target_index: Output index.
    :return torch.Tensor: Estimated expectation (single value).
    """
    return joint_marginal_expectation(model, x, fixed_features, background, nsamples, target_index)

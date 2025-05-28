"""
TimeExplainer: Attribution for Sequential Models via Temporal Perturbation

This explainer perturbs timesteps to estimate their influence on the model output:

.. math::

    \mathrm{Attribution}_t = f(x) - f(x_{\setminus t})

Supports single timestep masking, coalition masking, mean imputation,
and perturbation with Gaussian noise.

No normalization or post-processing is applied by default.
"""

import torch
import random
import numpy as np
from typing import Any, Optional, List

# -------------------------------
# Perturbation Utilities
# -------------------------------

def sample_coalition(T: int, exclude: List[int] = [], p: float = 0.3) -> List[int]:
    """
    Sample a random subset of timesteps as a coalition.

    :param T: Total number of timesteps
    :param exclude: Timesteps to exclude
    :param p: Proportion of timesteps to sample
    :return: List of timestep indices
    """
    candidates = list(set(range(T)) - set(exclude))
    k = max(1, int(p * len(candidates)))
    return random.sample(candidates, k)

def mask_timesteps(X: torch.Tensor, timesteps: List[int], mask_value: float = 0.0) -> torch.Tensor:
    """
    Mask specified timesteps in a sequence.

    :param X: Input tensor [B, T, F]
    :param timesteps: List of indices to mask
    :param mask_value: Value to use
    :return: Masked tensor
    """
    X_masked = X.clone()
    for t in timesteps:
        if 0 <= t < X.shape[1]:
            X_masked[:, t, :] = mask_value
    return X_masked

def impute_with_context(X: torch.Tensor, timesteps: List[int]) -> torch.Tensor:
    """
    Impute masked timesteps with mean of neighbors.

    :param X: Input [B, T, F]
    :param timesteps: Timesteps to impute
    :return: Modified tensor
    """
    X_copy = X.clone()
    B, T, F = X.shape
    for t in timesteps:
        if 0 < t < T - 1:
            X_copy[:, t] = 0.5 * (X[:, t - 1] + X[:, t + 1])
        else:
            X_copy[:, t] = 0.0
    return X_copy

def perturb_sequence_with_noise(X: torch.Tensor, noise_level: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise.

    :param X: Input [B, T, F]
    :param noise_level: Std dev
    :return: Noisy tensor
    """
    return X + torch.randn_like(X) * noise_level

# -------------------------------
# TimeExplainer
# -------------------------------

class TimeExplainer:
    """
    Time Explainer for sequential models.

    Perturbs specific timesteps (or coalitions) and computes attribution:

    .. math::

        \mathrm{Attribution}_t = |f(x) - f(x_{\setminus t})|

    :param model: Model to explain
    :param mode: 'mask', 'noise', or 'ensemble'
    :param mask_value: Value for masking
    :param noise_level: Std for Gaussian noise
    :param nsamples: Number of perturbation rounds
    :param strategy: 'sequential' or 'coalition'
    :param use_imputation: If True, impute instead of hard mask
    :param coalition_p: Proportion of time steps to perturb in coalition
    :param target_index: Output index to explain (if multi-output)
    """

    def __init__(self, model: Any, mode: str = "mask", mask_value: float = 0.0,
                 noise_level: float = 0.1, nsamples: int = 10, strategy: str = "sequential",
                 use_imputation: bool = False, coalition_p: float = 0.3,
                 target_index: Optional[int] = 0):
        self.model = model
        self.mode = mode
        self.mask_value = mask_value
        self.noise_level = noise_level
        self.nsamples = nsamples
        self.strategy = strategy
        self.use_imputation = use_imputation
        self.coalition_p = coalition_p
        self.target_index = target_index

    def explain(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute temporal attribution.

        :param X: Input [B, T, F]
        :return: Attribution [B, T]
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        B, T, F = X.shape
        attributions = torch.zeros((B, T), device=X.device)

        base_outputs = self.model(X)
        base_outputs = self._extract_output(base_outputs)

        for _ in range(self.nsamples):
            if self.strategy == "sequential":
                timestep_sets = [[t] for t in range(T)]
            elif self.strategy == "coalition":
                timestep_sets = [sample_coalition(T, p=self.coalition_p) for _ in range(T)]
            else:
                raise ValueError(f"Unknown strategy: {self.strategy}")

            for idx, timesteps in enumerate(timestep_sets):
                # Select perturbation type
                if self.mode == "ensemble":
                    current_mode = random.choice(["mask", "noise"])
                else:
                    current_mode = self.mode

                # Apply perturbation
                if current_mode == "mask":
                    if self.use_imputation:
                        perturbed_X = impute_with_context(X, timesteps)
                    else:
                        perturbed_X = mask_timesteps(X, timesteps, self.mask_value)
                elif current_mode == "noise":
                    perturbed_X = X.clone()
                    perturbed_X[:, timesteps, :] = perturb_sequence_with_noise(
                        perturbed_X[:, timesteps, :], self.noise_level
                    )
                else:
                    raise ValueError(f"Unsupported mode: {current_mode}")

                # Forward pass
                perturbed_outputs = self.model(perturbed_X)
                perturbed_outputs = self._extract_output(perturbed_outputs)

                # Absolute delta
                delta = (base_outputs - perturbed_outputs).abs()
                for t in timesteps:
                    attributions[:, t] += delta

        return attributions / self.nsamples

    def _extract_output(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Extracts scalar outputs for attribution.

        :param outputs: Tensor [B] or [B, C] or [B, 1]
        :return: Tensor [B]
        """
        if outputs.ndim == 2 and outputs.shape[1] > 1 and self.target_index is not None:
            return outputs[:, self.target_index]
        elif outputs.ndim == 2:
            return outputs.squeeze(1)
        return outputs

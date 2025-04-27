# src/shap_enhanced/explainers/gradient_based/smoothgrad_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class SmoothGradExplainer(BaseExplainer):
    """SmoothGrad Explainer for feature attribution.

    Adds noise to input and averages multiple gradients to smooth attributions:

    .. math::

        \\text{Attribution}_i = \\mathbb{E}_{\\epsilon \\sim \\mathcal{D}} \\left[ \\frac{\\partial f(x + \\epsilon)}{\\partial x_i} \\right]
    """

    def __init__(self, model: Any, noise_level: float = 0.1, samples: int = 50, noise_type: str = "normal", target_index: Optional[int] = 0) -> None:
        """Initialize the SmoothGrad Explainer.

        :param Any model: A differentiable model.
        :param float noise_level: Scale of noise added to input.
        :param int samples: Number of noisy samples to average.
        :param str noise_type: Type of noise ('normal', 'uniform', 'laplace').
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self.noise_level = noise_level
        self.samples = samples
        self.noise_type = noise_type
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using SmoothGrad.

        :param Any X: Input data (torch.Tensor).
        :return Any: Smoothed feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for SmoothGradExplainer.")

        X = X.clone().detach()
        all_gradients = []

        for _ in range(self.samples):
            noisy_X = self._add_noise(X)
            noisy_X.requires_grad_(True)

            outputs = self.model(noisy_X)

            if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                outputs = outputs[:, self.target_index]
            elif outputs.ndim > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 0]

            outputs = outputs.sum()
            outputs.backward()

            gradients = noisy_X.grad.detach()
            all_gradients.append(gradients)

        all_gradients = torch.stack(all_gradients, dim=0)  # (samples, batch, features...)
        attributions = monte_carlo_expectation(all_gradients)

        return self._format_output(attributions)

    def _add_noise(self, X: torch.Tensor) -> torch.Tensor:
        """Add noise to input tensor based on noise type.

        :param torch.Tensor X: Input tensor.
        :return torch.Tensor: Noisy tensor.
        """
        if self.noise_type == "normal":
            noise = torch.randn_like(X) * self.noise_level
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(X) - 0.5) * 2 * self.noise_level
        elif self.noise_type == "laplace":
            noise = torch.distributions.Laplace(0, self.noise_level).sample(X.shape).to(X.device)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return X + noise
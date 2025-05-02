"""Gradient Explainer for feature attribution using gradients."""


from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class NoiseGradExplainer(BaseExplainer):
    """NoiseGrad Explainer for feature attribution.

    Adds noise to both inputs and gradients to smooth attributions:

    .. math::

        \\text{Attribution}_i = \\mathbb{E}_{\\epsilon \\sim \\mathcal{D}} \\left[ \\frac{\\partial f(x + \\epsilon)}{\\partial (x + \\epsilon)} \\right]
    """

    def __init__(self, model: Any, input_noise_level: float = 0.1, gradient_noise_level: float = 0.1,
                 samples: int = 50, noise_type: str = "normal", target_index: Optional[int] = 0) -> None:
        """Initialize the NoiseGrad Explainer.

        :param Any model: A differentiable model.
        :param float input_noise_level: Noise std added to input.
        :param float gradient_noise_level: Noise std added to gradients.
        :param int samples: Number of noisy samples.
        :param str noise_type: Type of noise ('normal', 'uniform', 'laplace').
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self.input_noise_level = input_noise_level
        self.gradient_noise_level = gradient_noise_level
        self.samples = samples
        self.noise_type = noise_type
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using NoiseGrad.

        :param Any X: Input data (torch.Tensor).
        :return Any: Smoothed feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for NoiseGradExplainer.")

        X = X.clone().detach()
        all_gradients = []

        for _ in range(self.samples):
            noisy_X = self._add_noise(X, noise_level=self.input_noise_level)
            noisy_X.requires_grad_(True)

            outputs = self.model(noisy_X)

            if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                outputs = outputs[:, self.target_index]
            elif outputs.ndim > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 0]

            outputs = outputs.sum()
            outputs.backward()

            gradients = noisy_X.grad.detach()
            noisy_gradients = self._add_noise(gradients, noise_level=self.gradient_noise_level)

            all_gradients.append(noisy_gradients)

        all_gradients = torch.stack(all_gradients, dim=0)  # (samples, batch, features...)
        attributions = monte_carlo_expectation(all_gradients)

        return self._format_output(attributions)

    def _add_noise(self, X: torch.Tensor, noise_level: float) -> torch.Tensor:
        """Add noise to a tensor.

        :param torch.Tensor X: Input tensor.
        :param float noise_level: Noise level.
        :return torch.Tensor: Noisy tensor.
        """
        if self.noise_type == "normal":
            noise = torch.randn_like(X) * noise_level
        elif self.noise_type == "uniform":
            noise = (torch.rand_like(X) - 0.5) * 2 * noise_level
        elif self.noise_type == "laplace":
            noise = torch.distributions.Laplace(0, noise_level).sample(X.shape).to(X.device)
        else:
            raise ValueError(f"Unsupported noise type: {self.noise_type}")
        return X + noise
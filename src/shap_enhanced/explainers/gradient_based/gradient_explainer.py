# src/shap_enhanced/explainers/gradient_based/gradient_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class GradientExplainer(BaseExplainer):
    """Gradient Explainer for feature attribution.

    Computes the gradient of the model output with respect to the input:

    .. math::

        \\text{Attribution}_i = \\frac{\\partial f(x)}{\\partial x_i}
    """

    def __init__(self, model: Any, target_index: Optional[int] = 0) -> None:
        """Initialize the Gradient Explainer.

        :param Any model: A differentiable model (e.g., PyTorch).
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using gradients.

        :param Any X: Input data (torch.Tensor).
        :return Any: Feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for GradientExplainer.")

        X = X.clone().detach().requires_grad_(True)
        outputs = self.model(X)

        if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
            outputs = outputs[:, self.target_index]
        elif outputs.ndim > 1 and outputs.shape[1] > 1:
            outputs = outputs[:, 0]

        outputs = outputs.sum()  # Sum over batch to get a scalar
        outputs.backward()

        gradients = X.grad.detach()

        return self._format_output(gradients)

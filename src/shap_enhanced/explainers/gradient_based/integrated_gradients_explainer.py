# src/shap_enhanced/explainers/gradient_based/integrated_gradients_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.interpolation import linear_interpolation
from shap_enhanced.algorithms.integration import trapezoidal_integrate

class IntegratedGradientsExplainer(BaseExplainer):
    """Integrated Gradients Explainer for feature attribution.

    Computes path-integrated gradients from a baseline input to the actual input:

    .. math::

        \\text{Attribution}_i = (x_i - x'_i) \\times \\int_{\\alpha=0}^1 \\frac{\\partial f(x' + \\alpha (x - x'))}{\\partial x_i} d\\alpha
    """

    def __init__(self, model: Any, baseline: Optional[torch.Tensor] = None, steps: int = 50, target_index: Optional[int] = 0) -> None:
        """Initialize the Integrated Gradients Explainer.

        :param Any model: A differentiable model.
        :param Optional[torch.Tensor] baseline: Reference input. If None, uses zeros.
        :param int steps: Number of interpolation steps.
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self.baseline = baseline
        self.steps = steps
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using Integrated Gradients.

        :param Any X: Input data (torch.Tensor).
        :return Any: Integrated Gradients feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for IntegratedGradientsExplainer.")

        if self.baseline is None:
            baseline = torch.zeros_like(X)
        else:
            baseline = self.baseline.to(X.device).expand_as(X)

        interpolated_inputs = linear_interpolation(baseline, X, self.steps)  # (steps+1, batch, features...)

        gradients = []

        for step_input in interpolated_inputs:
            step_input = step_input.clone().detach().requires_grad_(True)
            outputs = self.model(step_input)

            if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                outputs = outputs[:, self.target_index]
            elif outputs.ndim > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 0]

            outputs = outputs.sum()
            outputs.backward()

            gradients.append(step_input.grad.detach())

        gradients = torch.stack(gradients, dim=0)  # (steps+1, batch, features...)
        avg_gradients = trapezoidal_integrate(gradients)

        attributions = (X - baseline) * avg_gradients

        return self._format_output(attributions)
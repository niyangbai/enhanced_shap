"""DeepLIFT Explainer for feature attribution."""

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer

class DeepLIFTExplainer(BaseExplainer):
    """DeepLIFT Explainer for feature attribution.

    Computes contribution scores relative to a baseline input:

    .. math::

        C_i = (x_i - x'_i) \\times \\frac{f(x) - f(x')}{x_i - x'_i}
    """

    def __init__(self, model: Any, baseline: Optional[torch.Tensor] = None, target_index: Optional[int] = 0) -> None:
        """Initialize the DeepLIFT Explainer.

        :param Any model: A differentiable model.
        :param Optional[torch.Tensor] baseline: Reference input. If None, uses zeros.
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self.baseline = baseline
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using DeepLIFT.

        :param Any X: Input data (torch.Tensor).
        :return Any: DeepLIFT feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for DeepLIFTExplainer.")

        if self.baseline is None:
            baseline = torch.zeros_like(X)
        else:
            baseline = self.baseline.to(X.device)
            if baseline.shape[0] == 1:
                baseline = baseline.expand_as(X)
            elif baseline.shape[0] != X.shape[0]:
                raise ValueError(f"Baseline batch size {baseline.shape[0]} must match input batch size {X.shape[0]} or be broadcastable.")

        X = X.clone().detach().requires_grad_(True)
        baseline = baseline.clone().detach().requires_grad_(True)

        output_X = self.model(X)
        output_baseline = self.model(baseline)

        if output_X.ndim > 1 and output_X.shape[1] > 1 and self.target_index is not None:
            output_X = output_X[:, self.target_index]
            output_baseline = output_baseline[:, self.target_index]
        elif output_X.ndim > 1 and output_X.shape[1] > 1:
            output_X = output_X[:, 0]
            output_baseline = output_baseline[:, 0]

        delta_output = (output_X - output_baseline).view(-1, *([1] * (X.ndim - 1)))  # (batch_size, 1, ..., 1)

        delta_input = X - baseline  # (batch_size, features...)

        outputs = output_X.sum()
        outputs.backward()

        gradients = X.grad.detach()  # (batch_size, features...)

        with torch.no_grad():
            contributions = gradients * delta_input

        return self._format_output(contributions)
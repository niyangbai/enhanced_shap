"""Guided Backpropagation Explainer for feature attribution."""

from __future__ import annotations

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.attention import guided_relu_backward_hook

class GuidedBackpropExplainer(BaseExplainer):
    """Guided Backpropagation Explainer for feature attribution.

    Modifies the backward pass through ReLU layers to allow only positive gradients:

    .. math::

        \\text{GuidedGrad}_i = \\max(0, \\frac{\\partial f(x)}{\\partial x_i})
    """

    def __init__(self, model: Any, target_index: Optional[int] = 0) -> None:
        """Initialize the Guided Backprop Explainer.

        :param Any model: A differentiable model with ReLU activations.
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self._hooks = []
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using Guided Backpropagation.

        :param Any X: Input data (torch.Tensor).
        :return Any: Guided backpropagation attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for GuidedBackpropExplainer.")

        self._register_hooks()

        X = X.clone().detach().requires_grad_(True)
        outputs = self.model(X)

        if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
            outputs = outputs[:, self.target_index]
        elif outputs.ndim > 1 and outputs.shape[1] > 1:
            outputs = outputs[:, 0]

        outputs = outputs.sum()
        outputs.backward()

        attributions = X.grad.detach()

        self._remove_hooks()

        return self._format_output(attributions)

    def _register_hooks(self) -> None:
        """Register backward hooks to modify ReLU layers."""
        self._hooks.clear()
        for module in self.model.modules():
            if isinstance(module, torch.nn.ReLU):
                hook = module.register_full_backward_hook(guided_relu_backward_hook())
                self._hooks.append(hook)

    def _remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
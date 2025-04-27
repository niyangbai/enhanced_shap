# src/shap_enhanced/explainers/gradient_based/expected_gradients_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.interpolation import linear_interpolation
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class ExpectedGradientsExplainer(BaseExplainer):
    """Expected Gradients Explainer for feature attribution.

    Approximates Shapley values by integrating gradients between random baselines and the input:

    .. math::

        \\text{Attribution}_i = \\mathbb{E}_{x' \\sim \\mathcal{D}} \\left[ (x_i - x'_i) \\times \\int_{0}^1 \\frac{\\partial f(x' + \\alpha (x - x'))}{\\partial x_i} d\\alpha \\right]
    """

    def __init__(self, model: Any, baselines: Optional[torch.Tensor] = None, steps: int = 50, samples: int = 10, target_index: Optional[int] = 0) -> None:
        """Initialize the Expected Gradients Explainer.

        :param Any model: A differentiable model.
        :param Optional[torch.Tensor] baselines: Set of reference inputs. If None, uses zeros.
        :param int steps: Number of interpolation steps.
        :param int samples: Number of baseline samples.
        :param Optional[int] target_index: Output index to explain (for multi-output models).
        """
        super().__init__(model)
        self.baselines = baselines
        self.steps = steps
        self.samples = samples
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using Expected Gradients.

        :param Any X: Input data (torch.Tensor).
        :return Any: Expected Gradients feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for ExpectedGradientsExplainer.")

        X = X.clone().detach()
        batch_size = X.shape[0]

        if self.baselines is None:
            baselines = torch.zeros_like(X)
        else:
            baselines = self.baselines.to(X.device)

        all_attributions = []

        for i in range(batch_size):
            x = X[i:i+1]  # (1, features...)

            attribution_samples = []

            for _ in range(self.samples):
                baseline_idx = torch.randint(0, baselines.shape[0], (1,))
                baseline_sample = baselines[baseline_idx].clone()

                interpolated_inputs = linear_interpolation(baseline_sample, x, self.steps)  # (steps+1, ...)

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

                gradients = torch.stack(gradients, dim=0)
                avg_gradients = gradients[:-1].mean(dim=0)  # Approximate integral (no trapezoidal rule here for efficiency)

                delta_input = (x - baseline_sample).squeeze(0)  # (features...)

                attribution = avg_gradients * delta_input
                attribution_samples.append(attribution)

            attribution_samples = torch.stack(attribution_samples, dim=0)
            expected_attribution = monte_carlo_expectation(attribution_samples)

            all_attributions.append(expected_attribution)

        final_attributions = torch.stack(all_attributions, dim=0)  # (batch_size, features...)

        return self._format_output(final_attributions)
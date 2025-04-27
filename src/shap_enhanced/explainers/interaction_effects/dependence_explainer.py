# src/shap_enhanced/explainers/interaction_effects/dependence_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.approximation import joint_marginal_expectation

class DependenceExplainer(BaseExplainer):
    """Dependence Explainer for marginal feature attribution.

    Estimates partial dependence by marginalizing other features:

    .. math::

        PD(x_i) = \\mathbb{E}_{x_{-i}} [ f(x_i, x_{-i}) ]
    """

    def __init__(self, model: Any, background: torch.Tensor, samples: int = 50,
                 target_index: Optional[int] = 0) -> None:
        """Initialize the Dependence Explainer.

        :param Any model: Black-box model.
        :param torch.Tensor background: Background dataset to marginalize over.
        :param int samples: Number of marginal samples.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.background = background
        self.samples = samples
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate partial dependence feature attributions.

        :param Any X: Input tensor (torch.Tensor) [batch, features].
        :return Any: Partial dependence attributions (torch.Tensor) [batch, features].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for DependenceExplainer.")

        if self.background is None:
            raise ValueError("Background dataset must be provided for DependenceExplainer.")

        batch_size, num_features = X.shape
        attributions = torch.zeros((batch_size, num_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]

            for j in range(num_features):
                pd_estimate = joint_marginal_expectation(
                    model=self.model,
                    x=x,
                    S=[j],
                    background=self.background,
                    nsamples=self.samples,
                    target_index=self.target_index,
                )
                attributions[i, j] = pd_estimate

        return self._format_output(attributions)
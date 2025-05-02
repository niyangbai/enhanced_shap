"""Ablation Explainer for Grouped Feature Attribution."""

from typing import Any, List
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.perturbation import mask_features

class AblationExplainer(BaseExplainer):
    """Ablation explainer for grouped feature attribution.

    Measures importance by ablating (masking) groups of features and
    observing the change in model output:

    .. math::

        \\text{Attribution}_G = f(x) - f(x_{\\setminus G})
    """

    def __init__(self, model: Any, groups: List[List[int]], mask_value: float = 0.0) -> None:
        """Initialize the Ablation Explainer.

        :param Any model: A model object (black-box model).
        :param List[List[int]] groups: List of feature index groups to ablate together.
        :param float mask_value: Value used to mask the features during ablation.
        """
        super().__init__(model)
        self.groups = groups
        self.mask_value = mask_value

    def explain(self, X: Any) -> Any:
        """Generate group-level attributions using ablation.

        :param Any X: Input data to explain (torch.Tensor).
        :return Any: Group-level attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for AblationExplainer.")

        batch_size = X.shape[0]
        num_groups = len(self.groups)
        attributions = torch.zeros((batch_size, num_groups), device=X.device)

        base_outputs = self.model(X)

        if base_outputs.ndim > 1 and base_outputs.shape[1] > 1:
            base_outputs = base_outputs[:, 0]

        for idx, group in enumerate(self.groups):
            X_masked = mask_features(X, group, mask_value=self.mask_value)
            perturbed_outputs = self.model(X_masked)

            if perturbed_outputs.ndim > 1 and perturbed_outputs.shape[1] > 1:
                perturbed_outputs = perturbed_outputs[:, 0]

            delta = base_outputs - perturbed_outputs
            attributions[:, idx] = delta

        return self._format_output(attributions)

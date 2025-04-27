from typing import Any
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.perturbation import mask_features

class OcclusionExplainer(BaseExplainer):
    """Occlusion explainer for feature attribution.

    Measures importance by masking features and observing prediction drop:

    .. math::

        \\text{Attribution}_i = f(x) - f(x_{\\setminus i})
    """

    def __init__(self, model: Any, mask_value: float = 0.0) -> None:
        """Initialize the Occlusion Explainer.

        :param Any model: A model object (black-box or differentiable).
        :param float mask_value: Value used to mask features.
        """
        super().__init__(model)
        self.mask_value = mask_value

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using occlusion sensitivity.

        :param Any X: Input data to explain (torch.Tensor).
        :return Any: Occlusion-based feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for OcclusionExplainer.")

        batch_size, num_features = X.shape
        attributions = torch.zeros((batch_size, num_features), device=X.device)

        base_outputs = self.model(X)

        if base_outputs.ndim > 1 and base_outputs.shape[1] > 1:
            base_outputs = base_outputs[:, 0]

        for i in range(num_features):
            X_masked = mask_features(X, [i], mask_value=self.mask_value)
            perturbed_outputs = self.model(X_masked)

            if perturbed_outputs.ndim > 1 and perturbed_outputs.shape[1] > 1:
                perturbed_outputs = perturbed_outputs[:, 0]

            delta = base_outputs - perturbed_outputs
            attributions[:, i] = delta

        return self._format_output(attributions)
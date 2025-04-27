# src/shap_enhanced/explainers/kernel_sampling/kernel_explainer.py

from typing import Any, Optional
import torch
import random
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_subsets
from shap_enhanced.algorithms.shapley_kernel import shapley_kernel_weights, entropy_kernel_weights, uniform_kernel_weights

class KernelExplainer(BaseExplainer):
    """KernelSHAP-style Explainer for model-agnostic feature attribution.

    Approximates Shapley values via sampling and kernel-based weighting:

    .. math::

        \\phi_i \\approx \\sum_{S \\subseteq N \\setminus \\{i\\}} \\omega(S) \\left( f(S \\cup \\{i\\}) - f(S) \\right)
    """

    def __init__(self, model: Any, background: torch.Tensor, nsamples: int = 100,
                 mask_value: float = 0.0, kernel_type: str = "shapley",
                 random_seed: Optional[int] = None, target_index: Optional[int] = 0) -> None:
        """Initialize the Kernel Explainer.

        :param Any model: Black-box model.
        :param torch.Tensor background: Background sample(s) for masked features.
        :param int nsamples: Number of subset samples.
        :param float mask_value: Used if background not provided.
        :param str kernel_type: 'shapley', 'entropy', 'uniform', or 'linear'.
        :param Optional[int] random_seed: Random seed for reproducibility.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.background = background
        self.nsamples = nsamples
        self.mask_value = mask_value
        self.kernel_type = kernel_type
        self.target_index = target_index
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using KernelSHAP approximation.

        :param Any X: Input data to explain (torch.Tensor).
        :return Any: Feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for KernelExplainer.")

        batch_size, num_features = X.shape
        attributions = torch.zeros((batch_size, num_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]
            phi = torch.zeros(num_features, device=X.device)

            subset_list = sample_subsets(num_features, self.nsamples)

            perturbed_inputs = []
            weights = []
            feature_maps = []

            for subset in subset_list:
                subset_features = list(subset)

                x_masked = self._mask_features_with_background(x, subset_features)
                perturbed_inputs.append(x_masked)
                weights.append(self._compute_kernel_weight(num_features, len(subset_features)))
                feature_maps.append(subset_features)

            perturbed_inputs = torch.cat(perturbed_inputs, dim=0)  # (nsamples, features)

            outputs = self.model(perturbed_inputs)

            if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                outputs = outputs[:, self.target_index]
            elif outputs.ndim > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 0]

            base_output = self.model(x)
            if base_output.ndim > 1 and base_output.shape[1] > 1 and self.target_index is not None:
                base_output = base_output[:, self.target_index]
            elif base_output.ndim > 1 and base_output.shape[1] > 1:
                base_output = base_output[:, 0]
            base_output = base_output.squeeze(0)

            for j, subset_features in enumerate(feature_maps):
                marginal_contribution = (base_output - outputs[j]).squeeze(0)
                for f in subset_features:
                    phi[f] += weights[j] * marginal_contribution

            attributions[i] = phi

        return self._format_output(attributions)

    def _mask_features_with_background(self, x: torch.Tensor, present_features: list) -> torch.Tensor:
        """Mask features not in present set using background.

        :param torch.Tensor x: Single input.
        :param list present_features: Indices of features to keep.
        :return torch.Tensor: Masked input.
        """
        if self.background is None:
            masked_x = torch.full_like(x, fill_value=self.mask_value)
        else:
            masked_x = self.background.clone()
            if masked_x.shape != x.shape:
                masked_x = masked_x.expand_as(x)

        masked_x[:, present_features] = x[:, present_features]
        return masked_x

    def _compute_kernel_weight(self, n_features: int, subset_size: int) -> float:
        """Compute kernel weight for a subset size.

        :param int n_features: Total number of features.
        :param int subset_size: Size of subset.
        :return float: Weight.
        """
        if self.kernel_type == "shapley":
            return shapley_kernel_weights(n_features, subset_size)
        elif self.kernel_type == "entropy":
            return entropy_kernel_weights(n_features, subset_size)
        elif self.kernel_type == "uniform":
            return uniform_kernel_weights(n_features, subset_size)
        elif self.kernel_type == "linear":
            return 1.0 / (subset_size + 1e-8)
        else:
            raise ValueError(f"Unknown kernel type: {self.kernel_type}")
# src/shap_enhanced/explainers/kernel_sampling/random_sampling_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_subsets

class RandomSamplingExplainer(BaseExplainer):
    """Random Sampling Explainer for feature attribution.

    Approximates marginal contributions by random feature masking:

    .. math::

        \\phi_i \\approx \\mathbb{E}_{S \\sim \\mathcal{U}} \\left[ f(x_S \\cup \\{i\\}) - f(x_S) \\right]
    """

    def __init__(self, model: Any, nsamples: int = 100, mask_value: float = 0.0,
                 subset_strategy: str = "uniform", target_index: Optional[int] = 0) -> None:
        """Initialize the Random Sampling Explainer.

        :param Any model: Black-box model.
        :param int nsamples: Number of random subsets.
        :param float mask_value: Masking value for features.
        :param str subset_strategy: 'uniform', 'sparse', or 'dense' subset sampling.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.nsamples = nsamples
        self.mask_value = mask_value
        self.subset_strategy = subset_strategy
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions via random sampling.

        :param Any X: Input tensor (torch.Tensor) [batch, features].
        :return Any: Feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for RandomSamplingExplainer.")

        batch_size, num_features = X.shape
        attributions = torch.zeros((batch_size, num_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]
            feature_contributions = torch.zeros(num_features, device=X.device)

            subsets = self._sample_feature_subsets(num_features)

            perturbed_inputs = []
            feature_maps = []

            base_output = self.model(x)
            if base_output.ndim > 1 and base_output.shape[1] > 1 and self.target_index is not None:
                base_output = base_output[:, self.target_index]
            elif base_output.ndim > 1 and base_output.shape[1] > 1:
                base_output = base_output[:, 0]
            base_output = base_output.squeeze(0)

            for subset in subsets:
                S = list(subset)
                x_S = self._mask_features(x, [j for j in range(num_features) if j not in S])
                perturbed_inputs.append(x_S)
                feature_maps.append(S)

            perturbed_inputs = torch.cat(perturbed_inputs, dim=0)

            outputs = self.model(perturbed_inputs)
            if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                outputs = outputs[:, self.target_index]
            elif outputs.ndim > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 0]

            for j, subset in enumerate(feature_maps):
                marginal_contribution = (base_output - outputs[j]).squeeze(0)
                for f in subset:
                    feature_contributions[f] += marginal_contribution

            feature_contributions /= self.nsamples
            attributions[i] = feature_contributions

        return self._format_output(attributions)

    def _sample_feature_subsets(self, n_features: int) -> list:
        """Sample feature subsets based on strategy.

        :param int n_features: Total number of features.
        :return list: List of feature subsets.
        """
        subsets = []
        for _ in range(self.nsamples):
            if self.subset_strategy == "uniform":
                subset_size = torch.randint(1, n_features, (1,)).item()
            elif self.subset_strategy == "sparse":
                subset_size = torch.randint(1, n_features // 2 + 1, (1,)).item()
            elif self.subset_strategy == "dense":
                subset_size = torch.randint(n_features // 2, n_features, (1,)).item()
            else:
                raise ValueError(f"Unknown subset strategy: {self.subset_strategy}")

            subset = torch.randperm(n_features)[:subset_size].tolist()
            subsets.append(subset)
        return subsets

    def _mask_features(self, x: torch.Tensor, features_to_mask: list) -> torch.Tensor:
        """Mask specified features.

        :param torch.Tensor x: Single input.
        :param list features_to_mask: Feature indices to mask.
        :return torch.Tensor: Masked input.
        """
        masked_x = x.clone()
        masked_x[:, features_to_mask] = self.mask_value
        return masked_x
# src/shap_enhanced/explainers/sparse_tabular/sparse_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_feature_subsets
from shap_enhanced.algorithms.perturbation import mask_features
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class SparseExplainer(BaseExplainer):
    """Sparse Feature Perturbation Explainer.

    Approximates feature importance by perturbing sparse subsets:

    .. math::

        \\phi_i = \\mathbb{E}_{S \\ni i} \\left[ f(x_S) - f(x_{S \\setminus \\{i\\}}) \\right]
    """

    def __init__(self, model: Any, nsamples: int = 100, mask_value: float = 0.0,
                 perturbation_level: float = 0.5, target_index: Optional[int] = 0,
                 random_seed: Optional[int] = None) -> None:
        """Initialize the Sparse Explainer.

        :param Any model: A tabular model.
        :param int nsamples: Number of sparse subsets sampled.
        :param float mask_value: Value to replace masked features.
        :param float perturbation_level: Fraction of features to perturb.
        :param Optional[int] target_index: Output index to explain.
        :param Optional[int] random_seed: Random seed for reproducibility.
        """
        super().__init__(model)
        self.nsamples = nsamples
        self.mask_value = mask_value
        self.perturbation_level = perturbation_level
        self.target_index = target_index
        if random_seed is not None:
            torch.manual_seed(random_seed)

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using sparse perturbations.

        :param Any X: Input tensor (batch, features).
        :return Any: Feature attributions (torch.Tensor) [batch, features].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for SparseExplainer.")

        batch_size, n_features = X.shape
        attributions = torch.zeros((batch_size, n_features), device=X.device)

        base_outputs = self.model(X)

        if base_outputs.ndim > 1 and base_outputs.shape[1] > 1 and self.target_index is not None:
            base_outputs = base_outputs[:, self.target_index]
        elif base_outputs.ndim > 1 and base_outputs.shape[1] > 1:
            base_outputs = base_outputs[:, 0]

        for i in range(batch_size):
            x = X[i:i+1]
            base_out = base_outputs[i:i+1]

            contribs = torch.zeros((n_features,), device=X.device)

            feature_subsets = sample_feature_subsets(n_features, self.nsamples)

            for subset in feature_subsets:
                x_subset_masked = self._mask_except(x, subset)
                output_subset = self.model(x_subset_masked)

                if output_subset.ndim > 1 and output_subset.shape[1] > 1 and self.target_index is not None:
                    output_subset = output_subset[:, self.target_index]
                elif output_subset.ndim > 1 and output_subset.shape[1] > 1:
                    output_subset = output_subset[:, 0]

                marginal_contribution = (output_subset - base_out).squeeze(0)

                for f in subset:
                    contribs[f] += marginal_contribution

            contribs /= self.nsamples
            attributions[i] = contribs

        return self._format_output(attributions)

    def _mask_except(self, x: torch.Tensor, feature_indices: list) -> torch.Tensor:
        """Mask all features except selected ones.

        :param torch.Tensor x: Single input tensor (1, features).
        :param list feature_indices: List of feature indices to keep.
        :return torch.Tensor: Masked tensor.
        """
        n_features = x.shape[1]
        mask = torch.zeros((n_features,), device=x.device)
        mask[feature_indices] = 1
        mask = mask.unsqueeze(0)

        x_masked = x.clone()
        x_masked = x_masked * mask + (1 - mask) * self.mask_value
        return x_masked
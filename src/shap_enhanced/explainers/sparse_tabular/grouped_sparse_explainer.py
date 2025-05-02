"""Grouped Sparse Feature Perturbation Explainer."""


from typing import Any, Optional, List
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_feature_groups
from shap_enhanced.algorithms.perturbation import mask_feature_groups
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class GroupedSparseExplainer(BaseExplainer):
    """Grouped Sparse Feature Perturbation Explainer.

    Approximates group-level feature attributions:

    .. math::

        \\phi_{G_j} = \\mathbb{E}_{S \\subseteq G \\setminus \\{G_j\\}} \\left[ f(x_{S \\cup G_j}) - f(x_S) \right]
    """

    def __init__(self, model: Any, groups: List[List[int]], nsamples: int = 100,
                 mask_value: float = 0.0, target_index: Optional[int] = 0) -> None:
        """Initialize the Grouped Sparse Explainer.

        :param Any model: Model to explain.
        :param List[List[int]] groups: List of feature groups.
        :param int nsamples: Number of random group subset samples.
        :param float mask_value: Value for masking.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.groups = groups
        self.nsamples = nsamples
        self.mask_value = mask_value
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate group-level feature attributions.

        :param Any X: Input tensor (batch, features).
        :return Any: Group attributions (torch.Tensor) [batch, n_groups].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for GroupedSparseExplainer.")

        batch_size, n_features = X.shape
        n_groups = len(self.groups)
        attributions = torch.zeros((batch_size, n_groups), device=X.device)

        base_outputs = self.model(X)

        if base_outputs.ndim > 1 and base_outputs.shape[1] > 1 and self.target_index is not None:
            base_outputs = base_outputs[:, self.target_index]
        elif base_outputs.ndim > 1 and base_outputs.shape[1] > 1:
            base_outputs = base_outputs[:, 0]

        for i in range(batch_size):
            x = X[i:i+1]
            base_out = base_outputs[i:i+1]

            group_contribs = torch.zeros((n_groups,), device=X.device)

            sampled_subsets = sample_feature_groups(self.groups, self.nsamples)

            for subset in sampled_subsets:
                x_subset = self._mask_except_groups(x, subset)
                output_subset = self.model(x_subset)

                if output_subset.ndim > 1 and output_subset.shape[1] > 1 and self.target_index is not None:
                    output_subset = output_subset[:, self.target_index]
                elif output_subset.ndim > 1 and output_subset.shape[1] > 1:
                    output_subset = output_subset[:, 0]

                for g_idx in range(n_groups):
                    if g_idx in subset:
                        continue
                    subset_with_g = subset + [g_idx]
                    x_subset_g = self._mask_except_groups(x, subset_with_g)
                    output_subset_g = self.model(x_subset_g)

                    if output_subset_g.ndim > 1 and output_subset_g.shape[1] > 1 and self.target_index is not None:
                        output_subset_g = output_subset_g[:, self.target_index]
                    elif output_subset_g.ndim > 1 and output_subset_g.shape[1] > 1:
                        output_subset_g = output_subset_g[:, 0]

                    marginal_contribution = (output_subset_g - output_subset).squeeze(0)
                    group_contribs[g_idx] += marginal_contribution

            group_contribs /= self.nsamples
            attributions[i] = group_contribs

        return self._format_output(attributions)

    def _mask_except_groups(self, x: torch.Tensor, keep_groups: List[int]) -> torch.Tensor:
        """Mask all groups except specified ones.

        :param torch.Tensor x: Input tensor (1, features).
        :param List[int] keep_groups: Indices of groups to keep.
        :return torch.Tensor: Masked tensor.
        """
        selected_features = [idx for g_idx in keep_groups for idx in self.groups[g_idx]]
        n_features = x.shape[1]

        mask = torch.zeros((n_features,), device=x.device)
        mask[selected_features] = 1
        mask = mask.unsqueeze(0)

        x_masked = x.clone()
        x_masked = x_masked * mask + (1 - mask) * self.mask_value
        return x_masked
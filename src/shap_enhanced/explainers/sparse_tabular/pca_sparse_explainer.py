# src/shap_enhanced/explainers/sparse_tabular/pca_sparse_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.pca_tools import fit_pca, transform_pca, inverse_pca, select_principal_components
from shap_enhanced.algorithms.sampling import sample_feature_subsets
from shap_enhanced.algorithms.masking import apply_binary_mask

class PCASparseExplainer(BaseExplainer):
    """PCA-Sparse Feature Perturbation Explainer.

    Perturbs principal components instead of raw features:

    .. math::

        \\phi_i = f(x) - f(\\text{PCA}^{-1}(\\text{masked PCA}(x)))
    """

    def __init__(self, model: Any, variance_threshold: float = 0.95,
                 nsamples: int = 100, mask_value: float = 0.0,
                 target_index: Optional[int] = 0) -> None:
        """Initialize the PCA Sparse Explainer.

        :param Any model: A model to explain.
        :param float variance_threshold: Cumulative variance explained by selected components.
        :param int nsamples: Number of sparse perturbation samples.
        :param float mask_value: Value to mask principal components.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.variance_threshold = variance_threshold
        self.nsamples = nsamples
        self.mask_value = mask_value
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using PCA-sparse perturbations.

        :param Any X: Input tensor (batch, features).
        :return Any: Feature attributions (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for PCASparseExplainer.")

        batch_size, n_features = X.shape
        attributions = torch.zeros((batch_size, n_features), device=X.device)

        pca = fit_pca(X, n_components=None)
        n_components = select_principal_components(pca, variance_threshold=self.variance_threshold)

        Z = transform_pca(X, pca)[:, :n_components]  # (batch, n_components)

        base_outputs = self.model(X)

        if base_outputs.ndim > 1 and base_outputs.shape[1] > 1 and self.target_index is not None:
            base_outputs = base_outputs[:, self.target_index]
        elif base_outputs.ndim > 1 and base_outputs.shape[1] > 1:
            base_outputs = base_outputs[:, 0]

        for i in range(batch_size):
            z = Z[i:i+1]
            base_out = base_outputs[i:i+1]

            contribs = torch.zeros((n_features,), device=X.device)

            component_subsets = sample_feature_subsets(n_components, self.nsamples)

            for subset in component_subsets:
                mask = torch.zeros_like(z)
                mask[:, subset] = 1

                z_masked = apply_binary_mask(z, mask, mask_value=self.mask_value)
                x_recon = inverse_pca(torch.cat([z_masked, torch.zeros(1, Z.shape[1] - n_components, device=X.device)], dim=1), pca)

                output_masked = self.model(x_recon)

                if output_masked.ndim > 1 and output_masked.shape[1] > 1 and self.target_index is not None:
                    output_masked = output_masked[:, self.target_index]
                elif output_masked.ndim > 1 and output_masked.shape[1] > 1:
                    output_masked = output_masked[:, 0]

                marginal_contribution = (base_out - output_masked).squeeze(0)

                recon_diff = (x_recon - X[i:i+1]).squeeze(0)

                contribs += marginal_contribution * recon_diff.abs()

            contribs /= self.nsamples
            attributions[i] = contribs

        return self._format_output(attributions)
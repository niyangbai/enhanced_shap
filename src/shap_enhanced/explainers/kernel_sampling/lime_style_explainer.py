"""LIME++ Style Explainer using local surrogate models with weighted fitting."""

from typing import Any, Optional
import torch
import numpy as np
from sklearn.linear_model import Ridge
from shap_enhanced.explainers.base import BaseExplainer

class LimeStyleExplainer(BaseExplainer):
    """LIME++ Style Explainer using local surrogate models with weighted fitting.

    Fits a weighted linear model to local perturbations around the input:

    .. math::

        f(x) \\approx g(x), \\quad g(z) = w(z) \\cdot \\text{Ridge}(z)
    """

    def __init__(self, model: Any, nsamples: int = 100, noise_level: float = 0.02,
                 alpha: float = 1.0, distance_metric: str = "l2",
                 target_index: Optional[int] = 0) -> None:
        """Initialize the LIME++ Style Explainer.

        :param Any model: Black-box model.
        :param int nsamples: Number of perturbations.
        :param float noise_level: Std deviation of Gaussian noise for perturbations.
        :param float alpha: Regularization strength for Ridge regression.
        :param str distance_metric: Distance metric ('l2' supported).
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.nsamples = nsamples
        self.noise_level = noise_level
        self.alpha = alpha
        self.distance_metric = distance_metric
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions using LIME++.

        :param Any X: Input tensor (torch.Tensor) [batch, features].
        :return Any: Feature attributions (torch.Tensor) [batch, features].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for LimeStyleExplainer.")

        batch_size, n_features = X.shape
        attributions = torch.zeros((batch_size, n_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]

            perturbations = self._generate_local_perturbations(x)
            perturbed_inputs = x + perturbations  # (nsamples, features)

            outputs = self.model(perturbed_inputs)

            if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                outputs = outputs[:, self.target_index]
            elif outputs.ndim > 1 and outputs.shape[1] > 1:
                outputs = outputs[:, 0]

            Z = perturbations.detach().cpu().numpy()
            Y = outputs.detach().cpu().numpy()

            distances = self._compute_distances(Z)
            weights = self._kernel(distances)

            ridge = Ridge(alpha=self.alpha, fit_intercept=True)
            ridge.fit(Z, Y, sample_weight=weights)

            local_attributions = torch.tensor(ridge.coef_, device=X.device, dtype=X.dtype)
            attributions[i] = local_attributions

        return self._format_output(attributions)

    def _generate_local_perturbations(self, x: torch.Tensor) -> torch.Tensor:
        """Generate Gaussian local perturbations.

        :param torch.Tensor x: Input tensor (1, features).
        :return torch.Tensor: Perturbations (nsamples, features).
        """
        noise = torch.randn((self.nsamples, x.shape[1]), device=x.device) * self.noise_level
        return noise

    def _compute_distances(self, perturbations: np.ndarray) -> np.ndarray:
        """Compute distances from original point.

        :param np.ndarray perturbations: Perturbations.
        :return np.ndarray: Distances.
        """
        if self.distance_metric == "l2":
            return np.linalg.norm(perturbations, axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {self.distance_metric}")

    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        """Kernel to weight perturbations based on distance.

        :param np.ndarray distances: Distances from original point.
        :return np.ndarray: Sample weights.
        """
        sigma = np.std(distances) * 0.5 + 1e-8
        weights = np.exp(-(distances**2) / (2 * sigma**2))
        return weights
""" SHAP Interaction Explainer for tree-based models."""

from typing import Any, Optional
import torch
from shap_enhanced.explainers.tree.base_tree_explainer import BaseTreeExplainer
from shap_enhanced.algorithms.sampling import sample_subsets
from shap_enhanced.algorithms.shapley_kernel import shapley_kernel_weights
from shap_enhanced.algorithms.perturbation import mask_features

class ShapInteractionExplainer(BaseTreeExplainer):
    """SHAP Interaction Explainer for tree-based models.

    Computes pairwise SHAP interaction values via weighted subset sampling:

    .. math::
        \phi_{i,j} = \sum_{S \subseteq N\setminus\{i,j\}} w(|S|) \Bigl(f(S \cup \{i,j\}) - f(S \cup \{i}) - f(S \cup \{j\}) + f(S)\Bigr)
    """

    def __init__(
        self,
        model: Any,
        nsamples: int = 100,
        mask_value: float = 0.0,
        target_index: Optional[int] = 0
    ) -> None:
        """Initialize the ShapInteractionExplainer.

        :param Any model: Tree-based model (XGBoost, LightGBM, RandomForest).
        :param int nsamples: Number of subset samples for approximation.
        :param float mask_value: Value to mask features not in subset.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model, mask_value, nsamples, target_index)
        self.nsamples = nsamples

    def explain(self, X: Any) -> Any:
        """Generate SHAP interaction values for each input.

        :param Any X: Input data (torch.Tensor) [batch, features].
        :return Any: Interaction values (torch.Tensor) [batch, features, features].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for ShapInteractionExplainer.")

        batch_size, num_features = X.shape
        interactions = torch.zeros((batch_size, num_features, num_features), device=X.device)

        for idx in range(batch_size):
            x = X[idx:idx+1]
            interactions[idx] = self._compute_interaction_matrix(x)

        return self._format_output(interactions)

    def _compute_interaction_matrix(self, x: torch.Tensor) -> torch.Tensor:
        """Compute pairwise SHAP interaction matrix for a single input.

        :param torch.Tensor x: Single input (1, features).
        :return torch.Tensor: Interaction matrix (features, features).
        """
        num_features = x.shape[1]
        interaction_matrix = torch.zeros((num_features, num_features), device=x.device)

        subsets = sample_subsets(num_features, self.nsamples)

        for S in subsets:
            S = list(S)
            # f(S)
            x_S = mask_features(x, S, self.mask_value)
            f_S = self._predict(x_S)

            for i in range(num_features):
                if i in S:
                    continue
                # f(S+i)
                S_i = S + [i]
                x_Si = mask_features(x, S_i, self.mask_value)
                f_Si = self._predict(x_Si)

                for j in range(i+1, num_features):
                    if j in S:
                        continue
                    # f(S+j)
                    S_j = S + [j]
                    x_Sj = mask_features(x, S_j, self.mask_value)
                    f_Sj = self._predict(x_Sj)
                    # f(S+i+j)
                    S_ij = S + [i, j]
                    x_Sij = mask_features(x, S_ij, self.mask_value)
                    f_Sij = self._predict(x_Sij)

                    # marginal interaction contribution
                    delta = (f_Sij - f_Si - f_Sj + f_S).squeeze(0)
                    weight = shapley_kernel_weights(num_features, len(S))

                    interaction_matrix[i, j] += weight * delta
                    interaction_matrix[j, i] += weight * delta

        return interaction_matrix

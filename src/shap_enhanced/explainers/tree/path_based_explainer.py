# src/shap_enhanced/explainers/tree/path_based_shap_explainer.py

from typing import Any, Optional
import torch
from shap_enhanced.explainers.tree.base_tree_explainer import BaseTreeExplainer

class PathBasedShapExplainer(BaseTreeExplainer):
    """Explains SHAP values based on decision paths in a decision tree."""

    def __init__(self, model: Any, mask_value: float = 0.0, nsamples: int = 100,
                 target_index: Optional[int] = 0) -> None:
        """Initialize the PathBasedShapExplainer.

        :param Any model: Tree-based model (XGBoost, LightGBM, RandomForest).
        :param float mask_value: Value to mask features.
        :param int nsamples: Number of subset samples to approximate SHAP values.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model, mask_value, nsamples, target_index)

    def explain(self, X: Any) -> Any:
        """Generate SHAP values based on decision path contributions."""
        batch_size, num_features = X.shape
        path_values = torch.zeros((batch_size, num_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]
            path_values[i] = self._compute_path_contributions(x)

        return self._format_output(path_values)

    def _compute_path_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SHAP contributions based on decision path decisions."""
        path_contributions = torch.zeros(x.shape[1], device=x.device)

        # Traverse decision tree paths and calculate contributions
        for tree in self.model.get_booster().get_dump():
            path_contrib = self._calculate_path_contribution(tree, x)
            path_contributions += path_contrib

        return path_contributions

    def _calculate_path_contribution(self, tree, x: torch.Tensor) -> torch.Tensor:
        """Calculate the contribution of a given decision path in the tree."""
        path_contrib = torch.zeros(x.shape[1], device=x.device)

        # Traverse the path and compute contributions
        path = self._get_tree_path(tree, x)
        contribution = self._get_path_contribution(path, tree, x)
        
        path_contrib += contribution
        return path_contrib

    def _get_tree_path(self, tree, x: torch.Tensor) -> list:
        """Extract the path taken by a given input in the decision tree."""
        path = []
        for node in tree:
            if node['split'] < x[0, node['feature']]:
                path.append(1)  # Left child
            else:
                path.append(0)  # Right child
        return path

    def _get_path_contribution(self, path: list, tree, x: torch.Tensor) -> torch.Tensor:
        """Calculate the contribution of a path."""
        path_contribution = torch.zeros(len(path), device=tree.device)

        # Iterate over the path and compute the contribution of each decision
        for idx, node in enumerate(path):
            feature = node['feature']
            split_value = node['split']

            # Mask the feature and compute the model's output
            original_output = self._predict(x)
            masked_x = self._mask_feature(x, feature, split_value)
            masked_output = self._predict(masked_x)

            # Calculate the marginal contribution for this split
            contribution = original_output - masked_output

            # Update path contribution
            path_contribution[feature] += contribution

        return path_contribution

    def _mask_feature(self, x: torch.Tensor, feature_index: int, split_value: float) -> torch.Tensor:
        """Mask the feature by setting it to the split value."""
        masked_x = x.clone()
        masked_x[:, feature_index] = split_value  # Set the feature to the split value
        return masked_x

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run model prediction and extract target output."""
        output = self.model(x)

        # Handle multi-output models
        if output.ndim > 1 and output.shape[1] > 1 and self.target_index is not None:
            output = output[:, self.target_index]
        elif output.ndim > 1 and output.shape[1] > 1:
            output = output[:, 0]

        return output.squeeze(-1)
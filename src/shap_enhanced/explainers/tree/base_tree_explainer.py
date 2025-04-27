# src/shap_enhanced/explainers/tree/base_tree_explainer.py

from typing import Any, List, Optional
import torch
import numpy as np
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_subsets

class BaseTreeExplainer(BaseExplainer):
    """Base explainer for tree-based models (XGBoost, LightGBM, RandomForest).

    Computes feature contributions based on tree structure, using path-based SHAP values.
    """

    def __init__(self, model: Any, mask_value: float = 0.0, nsamples: int = 100,
                 target_index: Optional[int] = 0) -> None:
        """Initialize the BaseTreeExplainer.

        :param Any model: Tree-based model (XGBoost, LightGBM, RandomForest).
        :param float mask_value: Value to mask features.
        :param int nsamples: Number of subset samples to approximate SHAP values.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.mask_value = mask_value
        self.nsamples = nsamples
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate SHAP values using the decision tree structure.

        :param Any X: Input data (torch.Tensor) [batch, features].
        :return Any: SHAP values (torch.Tensor) [batch, features].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for BaseTreeExplainer.")

        batch_size, num_features = X.shape
        attributions = torch.zeros((batch_size, num_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]  # (1, features)
            leaf_contributions = self._compute_leaf_contributions(x)
            path_contributions = self._compute_path_contributions(x)
            
            attributions[i] = leaf_contributions + path_contributions

        return self._format_output(attributions)

    def _compute_leaf_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SHAP contributions based on the leaf node predictions."""
        leaf_contributions = torch.zeros(x.shape[1], device=x.device)

        # Perform forward pass to get the model prediction
        prediction = self._predict(x)

        # Traverse tree paths and compute leaf-level SHAP contributions
        for tree in self.model.get_booster().get_dump():
            leaf_contrib = self._calculate_leaf_contribution(tree, x)
            leaf_contributions += leaf_contrib

        return leaf_contributions

    def _compute_path_contributions(self, x: torch.Tensor) -> torch.Tensor:
        """Compute SHAP contributions based on decision path decisions."""
        path_contributions = torch.zeros(x.shape[1], device=x.device)

        # Traverse decision tree paths and calculate contributions
        for tree in self.model.get_booster().get_dump():
            path_contrib = self._calculate_path_contribution(tree, x)
            path_contributions += path_contrib

        return path_contributions

    def _calculate_leaf_contribution(self, tree, x: torch.Tensor) -> torch.Tensor:
        """Calculate contribution of leaf node in decision tree."""
        # Traverse the tree path to reach the leaf node
        path = self._get_tree_path(tree, x)
        leaf_value = self._get_leaf_value(tree, path)

        # Calculate the SHAP contribution for this leaf
        leaf_contrib = torch.tensor(leaf_value - np.mean(path), device=x.device)
        return leaf_contrib

    def _calculate_path_contribution(self, tree, x: torch.Tensor) -> torch.Tensor:
        """Calculate contribution of a given path in the decision tree."""
        path = self._get_tree_path(tree, x)
        path_contrib = torch.tensor(np.mean(path), device=x.device)
        return path_contrib

    def _get_tree_path(self, tree, x: torch.Tensor) -> List[int]:
        """Extract the path taken by a given input in the decision tree."""
        # We traverse the decision tree from root to leaf and record the decision path
        path = []
        for node in tree:
            if node['split'] < x[0, node['feature']]:
                path.append(1)  # Left child
            else:
                path.append(0)  # Right child
        return path

    def _get_leaf_value(self, tree, path: List[int]) -> float:
        """Extract the value of the leaf node."""
        # Traverse the path to find the leaf value
        leaf_node = path[-1]
        return tree[leaf_node]['leaf_value']

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run model prediction and extract target output."""
        output = self.model(x)

        # Handle multi-output models
        if output.ndim > 1 and output.shape[1] > 1 and self.target_index is not None:
            output = output[:, self.target_index]
        elif output.ndim > 1 and output.shape[1] > 1:
            output = output[:, 0]

        return output.squeeze(-1)
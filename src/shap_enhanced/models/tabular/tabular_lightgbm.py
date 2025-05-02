"""LightGBM model for Tabular Data."""

import torch
from typing import Optional
import lightgbm as lgb
from shap_enhanced.models.base_model import BaseModel

class TabularLightGBM(BaseModel):
    """LightGBM model for Tabular Data."""

    def __init__(self, params: dict):
        """Initialize the LightGBM model.

        :param dict params: LightGBM hyperparameters.
        """
        super().__init__()
        self.model = lgb.LGBMRegressor(**params)  # Use LGBMClassifier for classification tasks

    def fit(self, X: torch.Tensor, y: torch.Tensor, eval_set: Optional[tuple] = None, 
            early_stopping_rounds: int = 10, verbose: bool = True) -> None:
        """Train the LightGBM model on input data X and target labels y."""
        X, y = X.numpy(), y.numpy()  # Convert tensors to numpy arrays
        self.model.fit(X, y, eval_set=eval_set, early_stopping_rounds=early_stopping_rounds, verbose=verbose)

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input X."""
        X = X.numpy()  # Convert tensor to numpy array
        return torch.tensor(self.model.predict(X), device=X.device)

    def evaluate(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Evaluate the LightGBM model on input data X and labels y."""
        X, y = X.numpy(), y.numpy()
        return self.model.score(X, y)  # Returns R^2 score for regression tasks

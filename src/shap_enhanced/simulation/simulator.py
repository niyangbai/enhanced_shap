# src/shap_enhanced/simulation/simulator.py

import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from typing import Callable, Union
from shap_enhanced.simulation.synthetic_data import generate_data


class ModelSimulation:
    """Simulate and evaluate SHAP values for various models and explainers."""

    def __init__(self, model, explainer, true_shap_function: Callable, metric: Union[str, Callable] = 'mse'):
        """Initialize the simulation with a model, explainer, and true SHAP function.

        :param model: The trained model to explain.
        :param explainer: The explainer to use for SHAP value estimation.
        :param true_shap_function: Function to calculate the true SHAP or importance values.
        :param metric: Evaluation metric, either a string ('mse', 'correlation', etc.) or a callable.
        """
        self.model = model
        self.explainer = explainer
        self.true_shap_function = true_shap_function
        self.metric = metric

    def generate_synthetic_data(self, n_samples: int, n_features: int, task_type: str = "regression", noise: float = 0.1):
        """Generate synthetic data with known feature importance.

        :param n_samples: Number of data points.
        :param n_features: Number of features.
        :param task_type: Task type, 'regression' or 'classification'.
        :param noise: Standard deviation of noise to add to the target.
        :return: A tuple of features (X) and true labels (y).
        """
        X = np.random.randn(n_samples, n_features)  # Random features
        y = np.sum(X, axis=1) + np.random.randn(n_samples) * noise  # Linear model + noise for regression

        if task_type == "classification":
            y = (y > 0).astype(int)  # Binary classification (y > 0 => class 1, else class 0)

        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

    def evaluate_metric(self, true_values: np.ndarray, predicted_values: np.ndarray) -> float:
        """Evaluate the selected metric.

        :param true_values: Ground truth values.
        :param predicted_values: Predicted values.
        :return: Computed metric value.
        """
        if isinstance(self.metric, str):
            if self.metric == 'mse':
                return mean_squared_error(true_values, predicted_values)
            elif self.metric == 'correlation':
                return np.corrcoef(true_values.flatten(), predicted_values.flatten())[0, 1]
            elif self.metric == 'r2':
                return r2_score(true_values, predicted_values)
            else:
                raise ValueError(f"Unsupported metric: {self.metric}")
        elif callable(self.metric):
            return self.metric(true_values, predicted_values)
        else:
            raise ValueError("Metric must be a string or a callable function.")

    def compare_shap(self, X: torch.Tensor, y: torch.Tensor) -> float:
        """Compare the true SHAP values with the model's explained SHAP values.

        :param X: Features to explain.
        :param y: True labels.
        :return: Evaluation metric value.
        """
        # Get the true SHAP values
        true_shap_values = self.true_shap_function(X, y)

        # Get the model's SHAP values using the explainer
        explained_shap_values = self.explainer.explain(X)

        # Evaluate using the selected metric
        return self.evaluate_metric(true_shap_values.flatten(), explained_shap_values.flatten())

    def run_simulation(self, n_samples: int = 1000, n_features: int = 10, task_type: str = "regression", noise: float = 0.1) -> float:
        """Run the entire simulation.

        :param n_samples: Number of samples for the synthetic data.
        :param n_features: Number of features in the synthetic data.
        :param task_type: Task type, either 'regression' or 'classification'.
        :param noise: Standard deviation of noise to add to the target.
        :return: The evaluation metric value.
        """
        # Generate synthetic data
        X, y = self.generate_synthetic_data(n_samples, n_features, task_type, noise)

        # Run the comparison and return the evaluation metric
        return self.compare_shap(X, y)

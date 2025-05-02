"""Evaluation functions for SHAP values comparison."""

from sklearn.metrics import mean_squared_error
import numpy as np

def evaluate_shap_comparison(true_shap_values, explained_shap_values, metric='mse') -> float:
    """Evaluate the comparison between true SHAP and explained SHAP values."""
    if metric == 'mse':
        return mean_squared_error(true_shap_values.flatten(), explained_shap_values.flatten())
    elif metric == 'correlation':
        return np.corrcoef(true_shap_values.flatten(), explained_shap_values.flatten())[0, 1]
    else:
        raise ValueError(f"Unsupported metric: {metric}")

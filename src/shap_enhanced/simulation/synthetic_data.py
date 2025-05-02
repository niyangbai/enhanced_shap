"""Synthetic data generation module for SHAP Enhanced."""

from shap_enhanced.datasets.synthetic_sequential import generate_sequential_data
from shap_enhanced.datasets.synthetic_sparse import generate_sparse_data
from shap_enhanced.datasets.synthetic_tabular import generate_tabular_data
from shap_enhanced.datasets.synthetic_vision import generate_vision_data

def generate_data(data_type: str, n_samples: int, n_features: int, task_type: str = "regression"):
    """Generate synthetic data based on the data type."""
    if data_type == "sequential":
        return generate_sequential_data(n_samples, n_features, n_features)
    elif data_type == "sparse":
        return generate_sparse_data(n_samples, n_features)
    elif data_type == "tabular":
        return generate_tabular_data(n_samples, n_features)
    elif data_type == "vision":
        return generate_vision_data(n_samples)
    else:
        raise ValueError("Invalid data type specified.")

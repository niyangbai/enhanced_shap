# src/shap_enhanced/visualization/feature_importance_plot.py

import matplotlib.pyplot as plt
import torch

def plot_feature_importance(importances: torch.Tensor, feature_names=None, title: str = "Feature Importance", top_k: int = 10):
    """Plot top-k feature importances as a bar chart.

    :param torch.Tensor importances: Importance scores for features.
    :param list feature_names: Names of the features (optional).
    :param str title: Title of the plot.
    :param int top_k: Number of top features to plot.
    """
    importances = importances.detach().cpu().numpy()
    indices = importances.argsort()[-top_k:][::-1]
    top_importances = importances[indices]

    if feature_names is not None:
        labels = [feature_names[i] for i in indices]
    else:
        labels = [f"Feature {i}" for i in indices]

    plt.figure(figsize=(8, 6))
    plt.barh(labels[::-1], top_importances[::-1])
    plt.xlabel("Importance")
    plt.title(title)
    plt.grid(True, axis='x')
    plt.tight_layout()
    plt.show()

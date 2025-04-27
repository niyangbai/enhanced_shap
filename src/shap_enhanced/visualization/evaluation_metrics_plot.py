# src/shap_enhanced/visualization/evaluation_metrics_plot.py

import matplotlib.pyplot as plt

def plot_evaluation_scores(scores: dict, title: str = "Explainer Performance Comparison"):
    """Plot a bar chart of evaluation scores across different explainers.

    :param dict scores: Dictionary of {explainer_name: score}.
    :param str title: Title of the plot.
    """
    explainers = list(scores.keys())
    values = list(scores.values())

    plt.figure(figsize=(8, 6))
    plt.bar(explainers, values)
    plt.title(title)
    plt.ylabel("Score (lower is better for MSE)")
    plt.xticks(rotation=45)
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.show()

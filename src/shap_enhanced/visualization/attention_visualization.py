# src/shap_enhanced/visualization/attention_visualization.py

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_attention_weights(attention: torch.Tensor, title: str = "Attention Weights"):
    """Plot a heatmap of attention weights.

    :param torch.Tensor attention: Attention weights (2D tensor: query x key).
    :param str title: Title of the plot.
    """
    attention = attention.detach().cpu().numpy()

    plt.figure(figsize=(8, 6))
    sns.heatmap(attention, cmap="viridis", cbar=True)
    plt.title(title)
    plt.xlabel("Key")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.show()

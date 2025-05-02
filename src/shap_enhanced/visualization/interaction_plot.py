"""Plot interaction effects as a heatmap."""

import matplotlib.pyplot as plt
import seaborn as sns
import torch

def plot_interaction_matrix(interaction_matrix: torch.Tensor, feature_names=None, title: str = "Interaction Effects"):
    """Plot interaction effects as a heatmap.

    :param torch.Tensor interaction_matrix: Matrix of interaction effects.
    :param list feature_names: List of feature names (optional).
    :param str title: Title of the plot.
    """
    interaction_matrix = interaction_matrix.detach().cpu().numpy()

    plt.figure(figsize=(10, 8))
    sns.heatmap(interaction_matrix, annot=False, fmt=".2f", cmap="coolwarm",
                xticklabels=feature_names, yticklabels=feature_names)
    plt.title(title)
    plt.xlabel("Feature")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.show()

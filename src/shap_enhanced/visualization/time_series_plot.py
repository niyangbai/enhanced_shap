"""Plotting function for time-series model attributions."""

import matplotlib.pyplot as plt
import torch

def plot_time_series_attribution(attributions: torch.Tensor, title: str = "Time Series Attribution", timestep_labels=None):
    """Plot attributions over time steps for time-series models.

    :param torch.Tensor attributions: Attribution values for each timestep.
    :param str title: Title of the plot.
    :param list timestep_labels: Custom labels for timesteps (optional).
    """
    attributions = attributions.detach().cpu().numpy()
    steps = range(len(attributions)) if timestep_labels is None else timestep_labels

    plt.figure(figsize=(10, 4))
    plt.plot(steps, attributions, marker='o')
    plt.title(title)
    plt.xlabel("Time Steps")
    plt.ylabel("Attribution")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

"""
Tools Module
============

The tools module provides utilities for model evaluation, visualization, benchmarking, 
and synthetic data generation. These tools are designed to support SHAP explainer development, 
interpretability experiments, and reproducible performance analysis.

Included Utilities
------------------

- **Evaluation and Comparison**:
  - Compute quantitative metrics (MSE, Pearson) between SHAP attributions and ground truth.
  - Compare different explainers systematically.

- **Visualization**:
  - High-quality plotting of SHAP attributions (1D/tabular and 2D/time-series).
  - 3D surfaces and bar plots for presentation or publication.

- **Data Generators**:
  - Create synthetic datasets for benchmarking SHAP methods.
  - Tabular and sequential settings supported.

- **Timing Utility**:
  - Context-managed block timer for profiling code.

- **Model Definitions**:
  - Reference models for both tabular and sequential inputs (e.g., LSTM, MLP).

"""

from .comparison import Comparison
from .datasets import generate_synthetic_seqregression, generate_synthetic_tabular
from .evaluation import compute_shapley_gt_seq, compute_shapley_gt_tabular
from .predefined_models import RealisticLSTM, TabularMLP
from .timer import Timer
from .visulization import (
    plot_mse_pearson,
    plot_3d_surface,
    plot_3d_bars,
    plot_feature_comparison
)

__all__ = [
    "Comparison",
    "generate_synthetic_seqregression",
    "generate_synthetic_tabular", 
    "compute_shapley_gt_seq",
    "compute_shapley_gt_tabular",
    "RealisticLSTM",
    "TabularMLP",
    "Timer",
    "plot_mse_pearson",
    "plot_3d_surface", 
    "plot_3d_bars",
    "plot_feature_comparison",
]

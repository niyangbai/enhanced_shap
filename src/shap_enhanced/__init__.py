"""
Enhanced SHAP Explainers
=========================

This package provides a collection of advanced SHAP-style explainers and supporting tools
designed for structured, sequential, and tabular data. It extends traditional SHAP
methodology with interpretable, efficient, and domain-aware enhancements.

Core Modules
------------

- **explainers**:

  A suite of explainers including:
    - Latent SHAP
    - RL-SHAP (Reinforcement Learning)
    - Multi-Baseline SHAP (MB-SHAP)
    - Sparse Coalition SHAP (SC-SHAP)
    - Surrogate SHAP (SurroSHAP)
    - TimeSHAP and others

- **tools**:

  Utility functions and helper modules for:
    - Synthetic data generation
    - Ground-truth SHAP value estimation
    - Model evaluation and visualizations
    - Benchmark comparisons and profiling

- **base_explainer**:
  Abstract base class (`BaseExplainer`) that defines the core interface
  for all SHAP-style explainers in this package.

Usage
-----

Example:

.. code-block:: python

  from shap_enhanced.explainers import LatentSHAPExplainer
  from shap_enhanced.tools.datasets import generate_synthetic_seqregression
  from shap_enhanced.tools.predefined_models import RealisticLSTM

  X, y = generate_synthetic_seqregression()
  model = RealisticLSTM(input_dim=X.shape[2])
  explainer = LatentSHAPExplainer(model=model, ...)
  shap_values = explainer.shap_values(X[0])
"""

from . import explainers, tools
from .base_explainer import BaseExplainer
from ._version import __version__

__all__ = ["explainers", "tools", "BaseExplainer"]

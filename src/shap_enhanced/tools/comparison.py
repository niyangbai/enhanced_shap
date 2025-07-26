"""
Attribution Comparison Utility for SHAP Explainers
==================================================

Overview
--------

This module provides a utility class for quantitatively comparing SHAP attributions  
from multiple explainers against a reference ground truth. It is intended for use in benchmarking  
or evaluating new SHAP-based methods by computing standard performance metrics.

Currently supported evaluation metrics include:

- **Mean Squared Error (MSE)**: Measures the squared deviation between predicted and ground-truth attributions.
- **Pearson Correlation**: Measures the linear correlation between flattened attribution arrays.

Key Components
^^^^^^^^^^^^^^

- **Comparison Class**:
  - Accepts ground-truth SHAP values and a dictionary of predicted attribution maps.
  - Computes MSE and Pearson correlation for each explainer.
  - Handles flattened comparison over all timesteps and features.

Use Case
--------

This utility is ideal for:
- Benchmarking SHAP-style explainers on synthetic datasets with known ground truth.
- Evaluating the effect of surrogate or approximation methods.
- Comparing different explainer strategies in attribution consistency.

Example
-------

.. code-block:: python

    gt = np.random.rand(10, 5)  # Ground truth SHAP values
    pred1 = gt + np.random.normal(0, 0.1, size=gt.shape)
    pred2 = gt + np.random.normal(0, 0.2, size=gt.shape)

    comp = Comparison(ground_truth=gt, shap_models={"ExplainerA": pred1, "ExplainerB": pred2})
    mse_scores, pearson_scores = comp.calculate_kpis()
"""


import numpy as np
from scipy.stats import pearsonr

__all__ = ["Comparison"]

class Comparison:
    r"""
    Comparison: SHAP Attribution Evaluation Utility

    Provides evaluation metrics for comparing predicted SHAP attributions against a ground truth reference.
    Designed for benchmarking SHAP-based explainers using quantitative metrics.

    Supported Metrics
    -----------------
    - **Mean Squared Error (MSE)**: Measures squared deviation between predicted and true SHAP values.
    - **Pearson Correlation**: Measures linear correlation between flattened attribution vectors.

    :param np.ndarray ground_truth: Ground-truth SHAP values of shape (T, F).
    :param dict shap_models: Dictionary mapping explainer names to their SHAP attribution arrays.
    """
    def __init__(self, ground_truth, shap_models):
        self.ground_truth = ground_truth
        self.shap_models = shap_models
        self.results = {}
        self.pearson_results = {}

    def calculate_kpis(self):
        r"""
        Compute evaluation metrics (MSE and Pearson correlation) for each SHAP explainer.

        .. note::
            Flattened comparisons are used for both MSE and correlation.

        :return: Tuple of dictionaries:
            - MSE values for each explainer.
            - Pearson correlation values for each explainer.
        :rtype: (dict[str, float], dict[str, float])
        """
        for name, arr in self.shap_models.items():
            mse = np.mean((arr - self.ground_truth) ** 2)
            gt_flat = self.ground_truth.flatten()
            arr_flat = arr.flatten()
            try:
                pearson, _ = pearsonr(gt_flat, arr_flat)
            except Exception:
                pearson = np.nan
            self.results[name] = mse
            self.pearson_results[name] = pearson
        return self.results, self.pearson_results

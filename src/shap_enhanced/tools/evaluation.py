"""
Ground-Truth Shapley Value Estimation via Monte Carlo
======================================================

Overview
--------

This module provides brute-force Monte Carlo estimators for **ground-truth Shapley values**  
in both sequential and tabular settings. These estimators are model-agnostic and compute  
marginal contributions of each feature (or feature–time pair) by averaging the effect  
of masking/unmasking them across many random feature subsets.

These implementations serve as a reference baseline for benchmarking approximate SHAP methods,  
especially in experimental or synthetic settings where accuracy is paramount.

Key Functions
^^^^^^^^^^^^^

- **compute_shapley_gt_seq**:  
  Computes per-feature-per-timestep Shapley values for a sequence input via masked sampling.

- **compute_shapley_gt_tabular**:  
  Computes standard tabular Shapley values for a single input vector using random coalitions.

Methodology
-----------

For a given input and feature subset mask \( S \), the Shapley value of feature \( i \) is computed as:

.. math::

    \phi_i = \mathbb{E}_{S \subseteq N \setminus \{i\}} [f(x_{S \cup \{i\}}) - f(x_S)]

This expectation is approximated by sampling random subsets \( S \) and measuring the marginal contribution  
of feature \( i \) over those subsets.

Use Case
--------

These estimators are useful for:
- Generating ground-truth SHAP values for synthetic benchmarking.
- Evaluating surrogate or approximate SHAP methods.
- Debugging the sensitivity of models to masking-based perturbations.

Example
-------

.. code-block:: python

    shap_seq = compute_shapley_gt_seq(model, x_seq, baseline_seq, nsamples=500)
    shap_tab = compute_shapley_gt_tabular(model, x_tab, baseline_tab, nsamples=1000)
"""


import numpy as np
import torch

def compute_shapley_gt_seq(model, x, baseline, nsamples=200, device="cpu"):
    r"""
    Estimate ground-truth Shapley values for a sequential input using Monte Carlo sampling.

    For each feature–timestep pair \((t, f)\), this method approximates the marginal contribution  
    by computing the model output difference between including and excluding \((t, f)\) from randomly  
    sampled coalitions.

    .. math::
        \phi_{t,f} = \mathbb{E}_{S \subseteq N \setminus \{(t,f)\}} \left[ f(x_{S \cup \{(t,f)\}}) - f(x_S) \right]

    :param model: A trained PyTorch model supporting (1, T, F)-shaped input.
    :param np.ndarray x: Input instance of shape (T, F).
    :param np.ndarray baseline: Reference baseline of same shape (T, F).
    :param int nsamples: Number of random coalitions sampled.
    :param str device: Device on which to run model evaluation ('cpu' or 'cuda').
    :return: Estimated Shapley values for each (t, f) position.
    :rtype: np.ndarray of shape (T, F)
    """
    T, F = x.shape
    vals = np.zeros((T, F))
    model.eval()
    with torch.no_grad():
        for t in range(T):
            for f in range(F):
                diffs = []
                for _ in range(nsamples):
                    mask = np.random.rand(T, F) < 0.5
                    m_with = mask.copy(); m_with[t, f] = True
                    m_without = mask.copy(); m_without[t, f] = False

                    def apply_mask(m):
                        xm = baseline.copy()
                        xm[m] = x[m]
                        inp = torch.tensor(xm[None], dtype=torch.float32).to(device)
                        return model(inp).cpu().numpy().squeeze()

                    y0 = apply_mask(m_without)
                    y1 = apply_mask(m_with)
                    diffs.append(y1 - y0)
                vals[t, f] = np.mean(diffs)
    return vals

def compute_shapley_gt_tabular(model, x, baseline, nsamples=1000, device="cpu"):
    r"""
    Estimate ground-truth Shapley values for a tabular input using Monte Carlo sampling.

    Each feature’s contribution is computed as the expected marginal impact on model output  
    when added to a random subset of other features.

    .. math::
        \phi_i = \mathbb{E}_{S \subseteq N \setminus \{i\}} \left[ f(x_{S \cup \{i\}}) - f(x_S) \right]

    :param model: A trained PyTorch model that accepts (1, F)-shaped inputs.
    :param np.ndarray x: Input feature vector of shape (F,).
    :param np.ndarray baseline: Baseline vector of same shape (F,).
    :param int nsamples: Number of Monte Carlo samples.
    :param str device: Device on which to run model evaluation ('cpu' or 'cuda').
    :return: Estimated Shapley values for each feature.
    :rtype: np.ndarray of shape (F,)
    """
    F = x.shape[0]
    shap_vals = np.zeros(F)
    model.eval()
    with torch.no_grad():
        for i in range(F):
            contribs = []
            for _ in range(nsamples):
                mask = np.random.rand(F) < 0.5
                mask_with = mask.copy(); mask_with[i] = True
                mask_without = mask.copy(); mask_without[i] = False
                def apply_mask(m):
                    x_mask = baseline.copy()
                    x_mask[m] = x[m]
                    inp = torch.tensor(x_mask[None], dtype=torch.float32).to(device)
                    return model(inp).cpu().numpy().squeeze()
                contribs.append(apply_mask(mask_with) - apply_mask(mask_without))
            shap_vals[i] = np.mean(contribs)
    return shap_vals
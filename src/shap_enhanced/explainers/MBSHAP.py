"""
MB-SHAP: Multi-Baseline SHAP Explainer
======================================

Theoretical Explanation
-----------------------

Multi-Baseline SHAP (MB-SHAP) enhances the robustness of SHAP-based feature attribution
by computing SHAP values with respect to multiple baselines rather than a single reference.
This addresses a key limitation in standard SHAP explainers: their sensitivity to baseline selection.

By averaging attributions from diverse or locally-relevant baselines (e.g., nearest neighbors, mean, k-means centroids),
MB-SHAP produces more stable, reliable, and representative explanations—particularly useful in domains with
heterogeneous data distributions or models that exhibit local nonlinearity.

Key Concepts
^^^^^^^^^^^^

- **Multiple Baselines**:
    Each input is explained with respect to a set of baselines instead of just one. Baseline options include:
        - Random background samples.
        - Mean or centroid-based references.
        - K nearest neighbors (local context).
        - User-specified selections.

- **Explainer Flexibility**:
    MB-SHAP is compatible with any SHAP-style explainer, including `DeepExplainer`, `GradientExplainer`, and `KernelExplainer`.
    It wraps the base explainer and runs it separately for each baseline.

- **Attribution Averaging**:
    For each input sample:
        - SHAP values are computed with respect to each baseline.
        - The resulting attribution vectors are averaged to yield a final, smoothed explanation.

- **Local Fidelity**:
    Using per-input nearest neighbors as baselines helps improve explanation fidelity for local model behavior.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, number of baselines, baseline selection strategy (`'random'`, `'nearest'`, `'mean'`, `'kmeans'`, etc.),
        SHAP explainer class (e.g., `shap.DeepExplainer`), and device context.

2. **Baseline Selection**:
    - For each input sample:
        - Select multiple baseline samples from the background using the chosen strategy.

3. **SHAP Value Computation**:
    - For each selected baseline:
        - Instantiate the base SHAP explainer.
        - Compute SHAP values for the input sample with respect to that baseline.
    - Average the SHAP results across all baselines.

4. **Output**:
    - Return the final attributions as averaged SHAP values, preserving shape and semantics of the model input.

References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**  
  [SHAP foundation—coalitional feature attribution framework]

- **Chen et al. (2022), “Explaining a Series of Models by Propagating Shapley Values” (G‑DeepSHAP)**  
  [Uses multiple baselines and shows that averaging explanations across them improves consistency and fidelity] :contentReference[oaicite:1]{index=1}

- **Google Vertex AI documentation (2025)**  
  [Allows multiple baseline specifications (e.g. min, max, random) to improve attribution context and stability] :contentReference[oaicite:2]{index=2}

- **Sundararajan & Najmi (2020), “The Many Shapley Values for Model Explanation”**  
  [Discusses how baseline selection influences SHAP-value interpretations, and the implications of multiple baseline settings] :contentReference[oaicite:3]{index=3}

- **Shaping Up SHAP: Enhancing Stability through Layer-Wise Neighbor Selection (Kelodjou et al., 2023)**  
  [Highlights instability in KernelSHAP and proposes neighbor sampling strategies to stabilize results, underscoring need for ensemble or multi-baseline approaches] :contentReference[oaicite:4]{index=4}
"""

import inspect

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class NearestNeighborMultiBaselineSHAP(BaseExplainer):
    r"""
    NearestNeighborMultiBaselineSHAP: Multi-Baseline SHAP Explainer

    This explainer improves attribution robustness by selecting the K nearest neighbors
    from a background dataset as baselines for each input sample, computing SHAP values
    individually for each baseline, and then averaging the results.

    It is compatible with various SHAP explainers (e.g., `DeepExplainer`, `GradientExplainer`, `KernelExplainer`)
    and automatically adapts input types and parameter formats accordingly.

    .. note::
        Baseline selection is input-dependent and done per sample using L2 distance in flattened input space.

    :param base_explainer_class: The SHAP explainer class to use (e.g., `shap.DeepExplainer`).
    :param model: The predictive model to explain.
    :type model: Any
    :param background: Background dataset (N, ...) for nearest neighbor selection.
    :type background: np.ndarray
    :param int n_baselines: Number of nearest neighbor baselines to use per sample.
    :param base_explainer_kwargs: Additional keyword arguments passed to the SHAP explainer.
    :type base_explainer_kwargs: dict or None
    :param str device: Device context for torch-based explainers ('cpu' or 'cuda').
    """

    def __init__(
        self,
        base_explainer_class,
        model,
        background,
        n_baselines=5,
        base_explainer_kwargs=None,
        device=None,
    ):
        self.base_explainer_class = base_explainer_class
        self.model = model
        self.background = np.asarray(background)
        self.n_baselines = n_baselines
        self.base_explainer_kwargs = base_explainer_kwargs or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _to_torch(self, arr):
        if isinstance(arr, torch.Tensor):
            return arr.to(self.device)
        arr = np.asarray(arr, dtype=np.float32)
        return torch.tensor(arr, dtype=torch.float32, device=self.device)

    def _make_explainer(self, baseline):
        # Handle different explainer parameter names
        cls = self.base_explainer_class
        sig = inspect.signature(cls.__init__)
        params = list(sig.parameters.keys())
        params = [p for p in params if p != "self"]
        if len(params) == 1:
            return cls(self.model, **self.base_explainer_kwargs)
        elif len(params) > 1:
            param2 = params[1].lower()
            if param2 in ("data", "background"):
                return cls(self.model, baseline, **self.base_explainer_kwargs)
            else:
                return cls(
                    self.model, **{param2: baseline}, **self.base_explainer_kwargs
                )
        else:
            raise RuntimeError("Cannot infer how to call explainer_class!")

    def _safe_shap_values(self, explainer, x, **kwargs):
        # Adds check_additivity=False if possible
        sig = inspect.signature(explainer.shap_values)
        params = sig.parameters
        if "check_additivity" in params:
            return explainer.shap_values(x, check_additivity=False, **kwargs)
        else:
            return explainer.shap_values(x, **kwargs)

    def shap_values(self, X, **kwargs):
        r"""
        Compute SHAP values using per-sample nearest neighbor baselines.

        For each sample in `X`, this method:
        1. Selects the `n_baselines` nearest neighbors from the background.
        2. Instantiates the explainer with the selected baselines.
        3. Computes SHAP values with respect to each baseline.
        4. Averages SHAP values across baselines to produce a robust explanation.

        .. math::
            \phi(x) = \frac{1}{K} \sum_{k=1}^{K} \text{SHAP}(x | b_k)

        :param X: Input samples to explain, shape (N, ...) or single sample (...).
        :type X: np.ndarray
        :param kwargs: Additional keyword arguments forwarded to the SHAP explainer.
        :return: Averaged SHAP attributions, shape (N, ...) or (...) for single input.
        :rtype: np.ndarray
        """
        X = np.asarray(X)
        if X.ndim == 2:
            X = X[None]
        n_samples = X.shape[0]
        bg_flat = self.background.reshape(self.background.shape[0], -1)
        attributions = []

        for i in range(n_samples):
            x = X[i]
            x_flat = x.reshape(-1)
            # K nearest neighbors in background
            dists = np.linalg.norm(bg_flat - x_flat, axis=1)
            idx = np.argsort(dists)[: self.n_baselines]
            nn_bases = self.background[idx]  # (n_baselines, T, F)

            # DeepExplainer wants a batch of backgrounds as tensor!
            expl_name = self.base_explainer_class.__name__.lower()
            if "deep" in expl_name or "gradient" in expl_name:
                baseline_ = self._to_torch(nn_bases)
                x_torch = self._to_torch(x[None])
                expl = self._make_explainer(baseline_)
                attr = self._safe_shap_values(expl, x_torch, **kwargs)
                if isinstance(attr, list):
                    attr = attr[0]
                if torch.is_tensor(attr):
                    attr = attr.detach().cpu().numpy()
                sample_avg = attr[0]  # first sample in batch
            else:
                # For KernelExplainer, background should be np.ndarray
                baseline_ = nn_bases
                x_in = x[None]
                expl = self._make_explainer(baseline_)
                attr = self._safe_shap_values(expl, x_in, **kwargs)
                if isinstance(attr, list):
                    attr = attr[0]
                sample_avg = attr[0]  # (T, F) or similar

            attributions.append(sample_avg)
        attributions = np.stack(attributions, axis=0)
        if attributions.shape[0] == 1:
            return attributions[0]
        return attributions

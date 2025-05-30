"""
Multi-Baseline SHAP (MB-SHAP) Explainer

## Theoretical Explanation

Multi-Baseline SHAP (MB-SHAP) computes SHAP values using multiple baseline references (e.g., random samples, mean, or k-means centroids) and averages the results to reduce dependence on a single baseline. This approach increases robustness and stability of feature attributions, especially for models or data where the choice of baseline can significantly affect SHAP values.

### Key Concepts

- **Multiple Baselines:** Instead of using a single background reference, MB-SHAP selects multiple baselines (e.g., nearest neighbors, random samples, or user-supplied) for each input sample.
- **Explainer Flexibility:** Supports any SHAP-style explainer (e.g., DeepExplainer, KernelExplainer) by wrapping the base explainer and running it for each baseline.
- **Averaging Attributions:** For each input, SHAP values are computed with respect to each baseline and then averaged to produce the final attribution.
- **Nearest-Neighbor Option:** Optionally, for each input, the K nearest background samples are used as baselines, further improving local fidelity.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data, number of baselines, baseline selection strategy, base explainer class, and device.
2. **Baseline Selection:**
    - For each input sample, select multiple baselines (e.g., nearest neighbors, random, mean, k-means, or user-supplied).
3. **SHAP Value Computation:**
    - For each baseline, instantiate the base explainer and compute SHAP values for the input.
    - Aggregate (average) the SHAP values across all baselines for each input.
4. **Output:**
    - Returns the averaged attributions for each input sample.
"""


from shap_enhanced.base_explainer import BaseExplainer

import numpy as np
import torch
import inspect

class NearestNeighborMultiBaselineSHAP(BaseExplainer):
    """
    Nearest-Neighbor Multi-Baseline SHAP Explainer.

    For each sample, selects its K nearest backgrounds as the explainer's reference set.
    Robust to both DeepExplainer (LSTM, neural nets) and KernelExplainer (tabular).
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
                return cls(self.model, **{param2: baseline}, **self.base_explainer_kwargs)
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
            idx = np.argsort(dists)[:self.n_baselines]
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

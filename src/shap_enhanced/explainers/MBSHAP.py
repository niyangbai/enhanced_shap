"""
MB-SHAP: Multi-Baseline SHAP Explainer
======================================

This explainer improves attribution robustness by selecting the K nearest neighbors 
from a background dataset as baselines for each input sample, computing SHAP values 
individually for each baseline, and then averaging the results.
"""

import numpy as np
import torch
import inspect
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator

class NearestNeighborMultiBaselineSHAP(BaseExplainer):
    r"""
    NearestNeighborMultiBaselineSHAP: Multi-Baseline SHAP Explainer

    This explainer improves attribution robustness by selecting the K nearest neighbors 
    from a background dataset as baselines for each input sample, computing SHAP values 
    individually for each baseline, and then averaging the results.

    :param base_explainer_class: The SHAP explainer class to use (e.g., `shap.DeepExplainer`).
    :param model: The predictive model to explain.
    :param background: Background dataset (N, ...) for nearest neighbor selection.
    :param int n_baselines: Number of nearest neighbor baselines to use per sample.
    :param base_explainer_kwargs: Additional keyword arguments passed to the SHAP explainer.
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
        super().__init__(model, background)
        self.base_explainer_class = base_explainer_class
        self.model = model
        self.background = BackgroundProcessor.process_background(background)
        self.n_baselines = n_baselines
        self.base_explainer_kwargs = base_explainer_kwargs or {}
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model_evaluator = ModelEvaluator(model, device)

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
        r"""
        Compute SHAP values using per-sample nearest neighbor baselines.

        For each sample in `X`, this method:
        1. Selects the `n_baselines` nearest neighbors from the background.
        2. Instantiates the explainer with the selected baselines.
        3. Computes SHAP values with respect to each baseline.
        4. Averages SHAP values across baselines to produce a robust explanation.

        :param X: Input samples to explain, shape (N, ...) or single sample (...).
        :return: Averaged SHAP attributions, shape (N, ...) or (...) for single input.
        """
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        
        bg_flat = self.background.reshape(self.background.shape[0], -1)
        attributions = []

        for b in range(B):
            x = X_processed[b]
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
        return attributions[0] if is_single else attributions
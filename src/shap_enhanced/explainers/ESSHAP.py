"""
EnsembleSHAPWithNoise: Robust Ensemble Wrapper for SHAP/Custom Explainers

- By default uses shap.DeepExplainer if no explainer_class given.
- Works with all official SHAP explainers and your custom explainers.
- Handles background/data args as required per class.
"""

import numpy as np
import torch
import inspect
import shap  # Requires SHAP package installed
from shap_enhanced.base_explainer import BaseExplainer

def _needs_torch(explainer_class):
    """Return True if explainer expects torch input/background."""
    name = explainer_class.__name__.lower()
    return name.startswith("deep") or name.startswith("gradient")

def _needs_numpy(explainer_class):
    """Return True if explainer expects numpy input/background."""
    name = explainer_class.__name__.lower()
    return name.startswith("kernel") or name.startswith("partition")

def _to_type(arr, typ, device=None):
    """Convert arr to np.ndarray or torch.Tensor, preserving dtype/shape."""
    if typ == "torch":
        if isinstance(arr, torch.Tensor):
            if device is not None:
                arr = arr.to(device)
            return arr
        else:
            return torch.tensor(arr, dtype=torch.float32, device=device)
    elif typ == "numpy":
        if isinstance(arr, np.ndarray):
            return arr
        else:
            return arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.array(arr)
    else:
        raise ValueError(f"Unknown type: {typ}")

def _add_noise(arr, noise_level):
    """Add Gaussian noise to arr, preserving its type."""
    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy() + np.random.normal(0, noise_level, arr.shape)
        return torch.tensor(arr_np, dtype=arr.dtype, device=arr.device)
    elif isinstance(arr, np.ndarray):
        return arr + np.random.normal(0, noise_level, arr.shape)
    else:
        raise TypeError("Input must be np.ndarray or torch.Tensor.")

def _has_arg(cls, name):
    sig = inspect.signature(cls.__init__)
    return name in sig.parameters

class EnsembleSHAPWithNoise(BaseExplainer):
    def __init__(
        self,
        model,
        background=None,
        explainer_class=None,   # Defaults to shap.DeepExplainer
        explainer_kwargs=None,
        n_runs=5,
        noise_level=0.1,
        noise_target="input",  # "input", "background", or "both"
        aggregation="mean",
        device=None
    ):
        super().__init__(model, background)
        self.model = model
        self.background = background
        self.explainer_class = explainer_class or shap.DeepExplainer
        self.explainer_kwargs = explainer_kwargs or {}
        self.n_runs = n_runs
        self.noise_level = noise_level
        self.noise_target = noise_target
        self.aggregation = aggregation
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def shap_values(self, X, **kwargs):
        attributions = []
        for run in range(self.n_runs):
            # Decide type for this explainer run
            if _needs_torch(self.explainer_class):
                typ = "torch"
            elif _needs_numpy(self.explainer_class):
                typ = "numpy"
            else:
                # Default to numpy (most explainers)
                typ = "numpy"
            # Prepare background with noise (if needed), convert to type
            bg_noisy = self.background
            if self.noise_target in ("background", "both") and self.background is not None:
                bg_noisy = _add_noise(self.background, self.noise_level)
            bg_noisy = _to_type(bg_noisy, typ, device=self.device) if bg_noisy is not None else None

            # Prepare input with noise (if needed), convert to type
            X_noisy = X
            if self.noise_target in ("input", "both"):
                X_noisy = _add_noise(X, self.noise_level)
            X_noisy = _to_type(X_noisy, typ, device=self.device)

            # Build explainer with correct background/data kwarg (if needed)
            expl_args = {"model": self.model}
            if bg_noisy is not None:
                if _has_arg(self.explainer_class, "background"):
                    expl_args["background"] = bg_noisy
                elif _has_arg(self.explainer_class, "data"):
                    expl_args["data"] = bg_noisy
            expl_args.update(self.explainer_kwargs)
            expl = self.explainer_class(**expl_args)

            # Evaluate
            attr = expl.shap_values(X_noisy, **kwargs)
            if isinstance(attr, list):  # SHAP DeepExplainer returns list for multi-output
                attr = attr[0]
            attributions.append(np.array(attr))
        attributions = np.stack(attributions, axis=0)
        if self.aggregation == "mean":
            return np.mean(attributions, axis=0)
        elif self.aggregation == "median":
            return np.median(attributions, axis=0)
        else:
            raise ValueError("Unknown aggregation method: {}".format(self.aggregation))

"""
EnsembleSHAPWithNoise: Robust Ensemble Wrapper for SHAP/Custom Explainers

## Theoretical Explanation

EnsembleSHAPWithNoise is a robust ensemble wrapper for SHAP and custom explainers. It improves attribution stability and robustness by running the chosen explainer multiple times with added Gaussian noise to the input data and/or background samples, then aggregating the results. This approach helps mitigate sensitivity to small perturbations and provides more reliable feature attributions, especially for deep or unstable models.

### Key Concepts

- **Ensemble Averaging:** Runs the base explainer multiple times with different noise realizations and aggregates the resulting attributions (mean or median).
- **Noise Injection:** Gaussian noise can be added to the input, background, or both, simulating data variability and improving robustness.
- **Explainer Flexibility:** Supports all official SHAP explainers and custom explainers, automatically handling their input/background requirements.
- **Automatic Type Handling:** Converts data to the required type (NumPy or PyTorch) for the selected explainer.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data, explainer class (default: `shap.DeepExplainer`), explainer kwargs, number of runs, noise level, noise target, aggregation method, and device.
2. **Ensemble Loop:**
    - For each run:
        - Add Gaussian noise to the input and/or background as specified.
        - Convert data to the required type for the explainer.
        - Instantiate the explainer with the noisy background.
        - Compute SHAP values for the noisy input.
    - Collect all attribution results.
3. **Aggregation:**
    - Aggregate the attributions across runs using the specified method (mean or median).
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

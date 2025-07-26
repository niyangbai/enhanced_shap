"""
EnsembleSHAPWithNoise: Robust Ensemble Wrapper for SHAP/Custom Explainers
=========================================================================

Theoretical Explanation
-----------------------

**EnsembleSHAPWithNoise** is a robust ensemble-based enhancement for SHAP and custom explainer methods.
It addresses instability in feature attributions—especially in deep or highly sensitive models—by
adding Gaussian noise to the inputs and/or background data across multiple runs of a base explainer,
then aggregating the resulting attribution maps.

This technique improves robustness by simulating data perturbations and reduces the variance in
feature importance estimates, leading to more reliable and stable interpretations.

Key Concepts
^^^^^^^^^^^^

- **Ensemble Averaging**:
    The explainer is executed multiple times with noisy versions of the input/background.
    Attribution results are aggregated using the specified method (mean or median).

- **Noise Injection**:
    Gaussian noise is applied to:
        - **Input**: Simulates perturbations in the sample to be explained.
        - **Background**: Introduces variability into the reference distribution used for attribution.
        - **Both**: Simulates end-to-end variability.

- **Explainer Flexibility**:
    Compatible with all SHAP explainers (e.g., `DeepExplainer`, `KernelExplainer`) and custom user-defined explainers.
    Automatically adapts inputs to the required format (NumPy or PyTorch).

- **Type Safety and Compatibility**:
    Automatically handles conversions between NumPy arrays and PyTorch tensors, depending on the explainer's requirements.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background data, explainer class (default: `shap.DeepExplainer`), number of runs,
        noise level (`float`), target for noise injection (`'input'`, `'background'`, or `'both'`),
        aggregation method (`'mean'` or `'median'`), explainer kwargs, and device context.

2. **Ensemble Loop**:
    - For each of the specified number of runs:
        - Inject Gaussian noise into the background and/or input, as specified.
        - Convert noisy data into the appropriate type (NumPy or PyTorch).
        - Instantiate the explainer using the noisy background.
        - Compute SHAP values on the noisy input.
        - Store the resulting attributions.

3. **Aggregation**:
    - Combine all attribution maps using the specified aggregation method (mean or median)
        to produce the final, noise-robust attribution result.
"""

import inspect

import numpy as np
import shap  # Requires SHAP package installed
import torch

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
            return (
                arr.detach().cpu().numpy()
                if isinstance(arr, torch.Tensor)
                else np.array(arr)
            )
    else:
        raise ValueError(f"Unknown type: {typ}")


def _add_noise(arr, noise_level):
    """Add Gaussian noise to arr, preserving its type."""
    if isinstance(arr, torch.Tensor):
        arr_np = arr.detach().cpu().numpy() + np.random.normal(
            0, noise_level, arr.shape
        )
        return torch.tensor(arr_np, dtype=arr.dtype, device=arr.device)
    elif isinstance(arr, np.ndarray):
        return arr + np.random.normal(0, noise_level, arr.shape)
    else:
        raise TypeError("Input must be np.ndarray or torch.Tensor.")


def _has_arg(cls, name):
    sig = inspect.signature(cls.__init__)
    return name in sig.parameters


class EnsembleSHAPWithNoise(BaseExplainer):
    r"""
    EnsembleSHAPWithNoise: Robust Ensemble Wrapper for SHAP/Custom Explainers

    This class enhances the stability of SHAP (SHapley Additive exPlanations) values by performing multiple runs
    with Gaussian noise applied to inputs and/or background data, and aggregating the results. It wraps around
    standard SHAP explainers or custom user-defined ones, making them more robust in the presence of
    sensitivity or instability.

    .. note::
        This class automatically handles input conversion between NumPy and PyTorch, depending on the explainer type.

    :param model: The model to explain.
    :param background: Background data used for SHAP attribution (can be None if not required).
    :param explainer_class: The SHAP or custom explainer class to wrap. Defaults to `shap.DeepExplainer`.
    :param explainer_kwargs: Dictionary of keyword arguments to pass to the explainer during instantiation.
    :param int n_runs: Number of noisy runs to perform for ensemble aggregation.
    :param float noise_level: Standard deviation of Gaussian noise to inject.
    :param str noise_target: Target for noise injection: "input", "background", or "both".
    :param str aggregation: Aggregation method across runs: "mean" or "median".
    :param device: Device context (e.g., 'cpu', 'cuda') for tensor-based explainers. Defaults to available GPU or CPU.
    """

    def __init__(
        self,
        model,
        background=None,
        explainer_class=None,  # Defaults to shap.DeepExplainer
        explainer_kwargs=None,
        n_runs=5,
        noise_level=0.1,
        noise_target="input",  # "input", "background", or "both"
        aggregation="mean",
        device=None,
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
        r"""
        Compute noise-robust SHAP values via ensemble averaging over multiple noisy runs.

        For each run, Gaussian noise is added to the input and/or background (as configured),
        then the SHAP explainer is applied to compute attribution values. These are aggregated
        (mean or median) to produce a stable final output.

        .. math::
            \text{Attribution}_{final}(i) =
            \begin{cases}
                \frac{1}{N} \sum_{j=1}^N \text{SHAP}_j(i) & \text{if aggregation = mean} \\
                \text{median}\{\text{SHAP}_1(i), \ldots, \text{SHAP}_N(i)\} & \text{if aggregation = median}
            \end{cases}

        :param X: Input sample(s) to explain (NumPy array or torch.Tensor).
        :type X: np.ndarray or torch.Tensor
        :param kwargs: Additional keyword arguments passed to the underlying explainer's `shap_values` method.
        :return: Aggregated attribution values across ensemble runs.
        :rtype: np.ndarray
        """
        attributions = []
        for _run in range(self.n_runs):
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
            if (
                self.noise_target in ("background", "both")
                and self.background is not None
            ):
                bg_noisy = _add_noise(self.background, self.noise_level)
            bg_noisy = (
                _to_type(bg_noisy, typ, device=self.device)
                if bg_noisy is not None
                else None
            )

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
            if isinstance(
                attr, list
            ):  # SHAP DeepExplainer returns list for multi-output
                attr = attr[0]
            attributions.append(np.array(attr))
        attributions = np.stack(attributions, axis=0)
        if self.aggregation == "mean":
            return np.mean(attributions, axis=0)
        elif self.aggregation == "median":
            return np.median(attributions, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")

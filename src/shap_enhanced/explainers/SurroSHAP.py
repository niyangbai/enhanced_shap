"""
SurroSHAP: Surrogate Model SHAP Explainer
=========================================

Theoretical Explanation
-----------------------

**SurroSHAP** is a surrogate modeling approach to SHAP that accelerates feature attribution
by training a regression model to **mimic SHAP values** produced by a base explainer.
Once trained, the surrogate regressor can produce fast, approximate SHAP values for new inputs,
bypassing the computational expense of re-running the base SHAP explainer.

This method is particularly useful for large datasets, expensive black-box models, or scenarios
where near-real-time interpretability is needed.

Key Concepts
^^^^^^^^^^^^

- **Surrogate Regression**:
    A regression model (e.g., Random Forest, Kernel Ridge, MLP) is trained to predict SHAP attributions
    using inputs as features and base SHAP values as targets.

- **Base SHAP Explainer**:
    Any standard SHAP-style explainer (e.g., `DeepExplainer`, `GradientExplainer`, `KernelExplainer`)
    can be used to generate training labels (pseudo-ground-truth SHAP values).

- **Optional Scaling**:
    Input features and/or SHAP attributions can be standardized to improve the surrogate's learning performance.

- **Fast Inference**:
    Once trained, the surrogate model can rapidly produce SHAP attributions for unseen inputs
    without invoking the base SHAP explainer again.

Algorithm
---------

1. **Initialization**:
    - Accepts the following:
        - A predictive model to explain.
        - Background dataset for base SHAP explainer.
        - SHAP-style base explainer class.
        - Surrogate regressor class (e.g., `sklearn.ensemble.RandomForestRegressor`).
        - Number of training samples for surrogate fitting.
        - Options for input/output scaling.
        - Device context (if applicable).

2. **Surrogate Training**:
    - Sample training points from the background dataset.
    - For each sample:
        - Compute SHAP values using the base explainer.
        - Flatten the input and corresponding SHAP vector.
    - Optionally scale both inputs and targets.
    - Fit the surrogate regressor on the collected (input, attribution) pairs.

3. **SHAP Value Prediction**:
    - For a new sample:
        - Flatten and optionally scale the input.
        - Predict SHAP attributions using the surrogate model.
        - Inverse-transform and reshape predictions to original attribution shape if needed.

Use Case
--------

SurroSHAP is best suited for:
    - Large-scale datasets requiring rapid SHAP value generation.
    - Scenarios where base SHAP computation is slow or expensive.
    - Situations where approximate explanations are acceptable in exchange for speed.

References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**  
  [SHAP foundation—coalitional feature attribution framework]

- **Zhou, Chen & Hu (2022), “Shapley Computations Using Surrogate Model‑Based Trees”**  
  [Uses surrogate tree models to compute SHAP values via conditional expectation, trading accuracy for speed] :contentReference[oaicite:1]{index=1}

- **ShapGAP (2024)**  
  [A metric for evaluating surrogate model fidelity by comparing SHAP explanations of surrogate vs. black‑box models, ensuring surrogate explanations align in reasoning] :contentReference[oaicite:2]{index=2}

- **Arize MimicExplainer documentation**  
  [Describes practical use of surrogate explainability: fitting a model (e.g., RandomForest) to mimic black‑box outputs and generating SHAP values from surrogate] :contentReference[oaicite:3]{index=3}

- **Interpretable Machine Learning book (Molnar, 2022), SHAP chapter**  
  [Discusses surrogate/approximation strategies and trade‑offs between fidelity and computational efficiency] :contentReference[oaicite:4]{index=4}
"""

import inspect
from typing import Any

import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

from shap_enhanced.base_explainer import BaseExplainer


def ensure_shap_input(x, explainer, device="cpu"):
    r"""
    Ensure compatibility of input format with the SHAP explainer type.

    This function inspects the explainer type (e.g., DeepExplainer, KernelExplainer)
    and converts the input `x` into the appropriate format—either a NumPy array or a
    PyTorch tensor—based on the explainer's requirements.

        - Deep/Gradient explainers require `torch.Tensor` input.
        - Kernel/Partition explainers require `np.ndarray` input.

    :param x: Input sample to format, shape (T, F) or (1, T, F).
    :type x: np.ndarray or torch.Tensor
    :param explainer: Instantiated SHAP explainer object.
    :type explainer: Any
    :param str device: Target torch device ('cpu' or 'cuda').
    :return: Properly formatted input for SHAP explainer.
    :rtype: np.ndarray or torch.Tensor
    :raises TypeError: If the input type is unsupported.
    """
    name = explainer.__class__.__name__.lower()
    is_deep = name.startswith("deep") or name.startswith("gradient")
    is_kernel = name.startswith("kernel") or name.startswith("partition")
    if isinstance(x, np.ndarray):
        if x.ndim == 2:
            x = x[None, ...]
        if is_deep:
            x = torch.tensor(x, dtype=torch.float32, device=device)
    elif isinstance(x, torch.Tensor):
        if x.ndim == 2:
            x = x[None, ...]
        if is_kernel:
            x = x.detach().cpu().numpy()
    else:
        raise TypeError("Unsupported type for SHAP input.")
    return x


def shap_values_with_nsamples(base_explainer, x, nsamples, **kwargs):
    r"""
    Safely compute SHAP values with optional support for `nsamples` argument.

    This utility inspects the signature of the `shap_values` method and attempts
    to call it with `nsamples`, `check_additivity`, and any additional kwargs.
    It includes fallback logic for older SHAP versions that may not support these parameters.

    :param base_explainer: SHAP explainer instance.
    :type base_explainer: Any
    :param x: Input to explain (already formatted for the explainer).
    :type x: np.ndarray or torch.Tensor
    :param int nsamples: Number of samples to use for SHAP estimation.
    :param kwargs: Additional keyword arguments for `shap_values`.
    :return: SHAP attributions for the input.
    :rtype: np.ndarray or list
    """
    sig = inspect.signature(base_explainer.shap_values)
    params = sig.parameters
    # Try nsamples if supported
    if "nsamples" in params:
        try:
            return base_explainer.shap_values(
                x, nsamples=nsamples, check_additivity=False, **kwargs
            )
        except TypeError:
            try:
                return base_explainer.shap_values(x, nsamples=nsamples, **kwargs)
            except TypeError:
                return base_explainer.shap_values(x, **kwargs)
    else:
        try:
            return base_explainer.shap_values(x, check_additivity=False, **kwargs)
        except TypeError:
            return base_explainer.shap_values(x, **kwargs)


class SurrogateSHAPExplainer(BaseExplainer):
    r"""
    SurrogateSHAPExplainer: Fast SHAP Approximation via Supervised Regression

    SurroSHAP accelerates SHAP attribution by training a surrogate regression model that maps
    input features to SHAP attributions. This is useful when repeated SHAP computation is too
    costly or when near-instant explanations are needed for deployment.

    The surrogate model is trained on a background dataset where "true" SHAP values are first computed
    using a base explainer (e.g., `DeepExplainer`, `KernelExplainer`), and then used as regression targets.

    .. note::
        Any sklearn-style regressor can be used (e.g., `RandomForestRegressor`, `KernelRidge`, etc.).

    :param model: Predictive model to be explained.
    :type model: Any
    :param background: Background dataset for training surrogate and computing SHAP targets. Shape: (N, T, F).
    :type background: np.ndarray
    :param base_explainer: A SHAP-style explainer instance (already constructed).
    :type base_explainer: Any
    :param regressor_class: Regressor class implementing fit/predict API. Defaults to RandomForestRegressor.
    :type regressor_class: type
    :param regressor_kwargs: Optional keyword arguments for the regressor.
    :type regressor_kwargs: dict
    :param int nsamples_base: Number of SHAP samples used for each background point.
    :param bool scale_inputs: Whether to standardize input features during training.
    :param bool scale_outputs: Whether to standardize SHAP values during training.
    :param device: Torch device (e.g., 'cpu' or 'cuda').
    :type device: str or torch.device
    """

    def __init__(
        self,
        model: Any,
        background: np.ndarray,
        base_explainer: Any,
        regressor_class=RandomForestRegressor,
        regressor_kwargs=None,
        nsamples_base=100,
        scale_inputs=True,
        scale_outputs=False,
        device=None,
    ):
        super().__init__(model, background)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_explainer = base_explainer
        self.regressor_class = regressor_class
        self.regressor_kwargs = regressor_kwargs or {}
        self.nsamples_base = nsamples_base
        self.scale_inputs = scale_inputs
        self.scale_outputs = scale_outputs
        self.input_scaler = None
        self.output_scaler = None
        self.T = background.shape[1]
        self.F = background.shape[2]
        self._fit_surrogate(background)

    def _fit_surrogate(self, X_bg):
        print("[SurroSHAP] Computing SHAP attributions for background data...")
        Y_shap = []
        for i, x in enumerate(X_bg):
            x_shap = ensure_shap_input(x, self.base_explainer, device=self.device)
            shap_val = shap_values_with_nsamples(
                self.base_explainer, x_shap, self.nsamples_base
            )
            if isinstance(shap_val, list):
                shap_val = shap_val[0]
            Y_shap.append(shap_val.flatten())
            if (i + 1) % 10 == 0 or i == len(X_bg) - 1:
                print(f"  {i+1}/{len(X_bg)} attributions done...")
        Y_shap = np.stack(Y_shap, axis=0)  # (N, T*F)
        X_feat = X_bg.reshape(X_bg.shape[0], -1)

        if self.scale_inputs:
            self.input_scaler = StandardScaler().fit(X_feat)
            X_feat = self.input_scaler.transform(X_feat)
        if self.scale_outputs:
            self.output_scaler = StandardScaler().fit(Y_shap)
            Y_shap = self.output_scaler.transform(Y_shap)

        print("[SurroSHAP] Training surrogate regression model...")
        reg = self.regressor_class(**self.regressor_kwargs)
        reg.fit(X_feat, Y_shap)
        self.regressor = reg
        print("[SurroSHAP] Surrogate trained.")

    def shap_values(self, X: np.ndarray | torch.Tensor, **kwargs) -> np.ndarray:
        r"""
        Predict SHAP attributions for new inputs using the trained surrogate model.

        The input is reshaped and (optionally) standardized to match the format used
        during surrogate training, and the predicted SHAP values are inverse-transformed
        (if scaling was applied).

        .. note::
            This bypasses SHAP computation entirely and relies on the surrogate regressor.

        :param X: Input instance or batch, shape (T, F) or (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :return: Approximated SHAP attributions, same shape as input.
        :rtype: np.ndarray
        """
        is_torch = hasattr(X, "detach")
        X_np = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = False
        if X_np.ndim == 2:  # (T, F)
            X_np = X_np[None, ...]
            single = True
        X_feat = X_np.reshape(X_np.shape[0], -1)
        if self.scale_inputs:
            X_feat = self.input_scaler.transform(X_feat)
        pred = self.regressor.predict(X_feat)
        if self.scale_outputs:
            pred = self.output_scaler.inverse_transform(pred)
        pred = pred.reshape(X_np.shape[0], self.T, self.F)
        return pred[0] if single else pred

"""
Surrogate Model SHAP (SurroSHAP) Explainer

## Theoretical Explanation

SurroSHAP is a surrogate-based SHAP explainer that trains a regression model to predict SHAP attributions, using pseudo-ground-truth from a base SHAP-style explainer. This approach enables fast, approximate SHAP value computation for new samples after the surrogate is trained, making it practical for large datasets or expensive models.

### Key Concepts

- **Surrogate Regression:** A regression model (e.g., Random Forest, Kernel Ridge) is trained to map input features to SHAP attributions, using attributions computed by a base explainer as targets.
- **Base Explainer:** Any SHAP-style explainer (e.g., DeepExplainer, KernelExplainer) can be used to generate the pseudo-ground-truth attributions for training.
- **Input/Output Scaling:** Optionally standardizes inputs and/or attributions to improve surrogate learning.
- **Fast Inference:** Once trained, the surrogate regressor can rapidly predict SHAP values for new samples without running the base explainer.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data, base explainer, regressor class, regressor kwargs, number of samples for base explainer, scaling options, and device.
2. **Surrogate Training:**
    - For each background sample:
        - Compute SHAP attributions using the base explainer.
        - Flatten and collect attributions as regression targets.
    - Optionally scale inputs and/or outputs.
    - Train the surrogate regressor on (input, attribution) pairs.
3. **SHAP Value Prediction:**
    - For new samples:
        - Flatten and scale inputs as needed.
        - Predict attributions using the surrogate regressor.
        - Optionally inverse-transform outputs and reshape to original dimensions.
"""


import numpy as np
import torch
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import StandardScaler
from typing import Any, Union
import inspect

from shap_enhanced.base_explainer import BaseExplainer

def ensure_shap_input(x, explainer, device="cpu"):
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
    sig = inspect.signature(base_explainer.shap_values)
    params = sig.parameters
    # Try nsamples if supported
    if "nsamples" in params:
        try:
            return base_explainer.shap_values(x, nsamples=nsamples, check_additivity=False, **kwargs)
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
    """
    Surrogate Model SHAP (SurroSHAP) Explainer.

    Parameters
    ----------
    model : Any
        The model to be explained.
    background : np.ndarray
        Background/training data (N, T, F).
    base_explainer : Any
        SHAP-style explainer to provide attributions.
    regressor_class : class (default: RandomForestRegressor)
        Any sklearn-style regressor for multi-output regression.
    regressor_kwargs : dict
        Arguments to regressor_class.
    nsamples_base : int
        Number of samples to use for base explainer attributions (if supported).
    scale_inputs : bool
        Whether to standardize inputs for the surrogate regressor.
    scale_outputs : bool
        Whether to standardize attributions for surrogate learning.
    device : str or torch.device
        Torch device.
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
        device=None
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
            shap_val = shap_values_with_nsamples(self.base_explainer, x_shap, self.nsamples_base)
            if isinstance(shap_val, list):
                shap_val = shap_val[0]
            Y_shap.append(shap_val.flatten())
            if (i+1) % 10 == 0 or i == len(X_bg)-1:
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

    def shap_values(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        is_torch = hasattr(X, 'detach')
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

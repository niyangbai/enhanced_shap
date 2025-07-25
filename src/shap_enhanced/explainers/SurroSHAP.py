"""
SurroSHAP: Surrogate Model SHAP Explainer
=========================================

**SurroSHAP** is a surrogate modeling approach to SHAP that accelerates feature attribution  
by training a regression model to **mimic SHAP values** produced by a base explainer.  
Once trained, the surrogate regressor can produce fast, approximate SHAP values for new inputs,  
bypassing the computational expense of re-running the base SHAP explainer.
"""

import numpy as np
import torch
from typing import Any, Union
import inspect
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.data_processing import process_inputs

def ensure_shap_input(x, explainer, device="cpu"):
    r"""
    Ensure compatibility of input format with the SHAP explainer type.
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
    """
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
    r"""
    SurrogateSHAPExplainer: Fast SHAP Approximation via Supervised Regression

    SurroSHAP accelerates SHAP attribution by training a surrogate regression model that maps 
    input features to SHAP attributions. This is useful when repeated SHAP computation is too 
    costly or when near-instant explanations are needed for deployment.

    :param model: Predictive model to be explained.
    :param background: Background dataset for training surrogate and computing SHAP targets. Shape: (N, T, F).
    :param base_explainer: A SHAP-style explainer instance (already constructed).
    :param regressor_class: Regressor class implementing fit/predict API.
    :param regressor_kwargs: Optional keyword arguments for the regressor.
    :param int nsamples_base: Number of SHAP samples used for each background point.
    :param bool scale_inputs: Whether to standardize input features during training.
    :param bool scale_outputs: Whether to standardize SHAP values during training.
    :param device: Torch device (e.g., 'cpu' or 'cuda').
    """
    def __init__(
        self,
        model: Any,
        background: np.ndarray,
        base_explainer: Any,
        regressor_class=None,
        regressor_kwargs=None,
        nsamples_base=100,
        scale_inputs=True,
        scale_outputs=False,
        device=None
    ):
        super().__init__(model, background)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_explainer = base_explainer
        
        # Default regressor
        if regressor_class is None:
            try:
                from sklearn.ensemble import RandomForestRegressor
                self.regressor_class = RandomForestRegressor
            except ImportError:
                raise ImportError("sklearn is required for default regressor")
        else:
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
            try:
                from sklearn.preprocessing import StandardScaler
                self.input_scaler = StandardScaler().fit(X_feat)
                X_feat = self.input_scaler.transform(X_feat)
            except ImportError:
                print("Warning: sklearn not available, skipping input scaling")
                self.scale_inputs = False
                
        if self.scale_outputs:
            try:
                from sklearn.preprocessing import StandardScaler
                self.output_scaler = StandardScaler().fit(Y_shap)
                Y_shap = self.output_scaler.transform(Y_shap)
            except ImportError:
                print("Warning: sklearn not available, skipping output scaling")
                self.scale_outputs = False

        print("[SurroSHAP] Training surrogate regression model...")
        reg = self.regressor_class(**self.regressor_kwargs)
        reg.fit(X_feat, Y_shap)
        self.regressor = reg
        print("[SurroSHAP] Surrogate trained.")

    def shap_values(self, X: Union[np.ndarray, torch.Tensor], **kwargs) -> np.ndarray:
        r"""
        Predict SHAP attributions for new inputs using the trained surrogate model.

        The input is reshaped and (optionally) standardized to match the format used 
        during surrogate training, and the predicted SHAP values are inverse-transformed 
        (if scaling was applied).

        :param X: Input instance or batch, shape (T, F) or (B, T, F).
        :return: Approximated SHAP attributions, same shape as input.
        """
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        
        X_feat = X_processed.reshape(X_processed.shape[0], -1)
        if self.scale_inputs and self.input_scaler is not None:
            X_feat = self.input_scaler.transform(X_feat)
        pred = self.regressor.predict(X_feat)
        if self.scale_outputs and self.output_scaler is not None:
            pred = self.output_scaler.inverse_transform(pred)
        pred = pred.reshape(X_processed.shape[0], self.T, self.F)
        return pred[0] if is_single else pred
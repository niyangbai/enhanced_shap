"""
Surrogate Model SHAP (SurroSHAP) Explainer.

This explainer trains a regression surrogate (e.g., kernel ridge, tree, or any sklearn regressor)
to learn the mapping: input → SHAP attributions, where SHAP attributions are obtained from
a base SHAP-style explainer on a background/training set.

The surrogate is then used to predict attributions for new data.

No access to ground-truth SHAP is assumed; only a base explainer is used for the "teacher" role.
"""

import numpy as np
from sklearn.kernel_ridge import KernelRidge
from typing import Any, Optional, Union

from shap_enhanced.base_explainer import BaseExplainer

class SurrogateSHAPExplainer(BaseExplainer):
    """
    Surrogate Model SHAP (SurroSHAP) Explainer.

    Parameters
    ----------
    model : Any
        The model to be explained.
    background : np.ndarray
        Training/background data (N, T, F) for surrogate.
    base_explainer : BaseExplainer or compatible
        Any SHAP-style explainer (must have .shap_values(X)) to provide pseudo-ground-truth attributions.
    regressor_class : class (default: KernelRidge)
        Any sklearn-style regressor for multi-output regression.
    regressor_kwargs : dict
        Arguments to regressor_class.
    """
    def __init__(
        self,
        model: Any,
        background: np.ndarray,
        base_explainer: Any,
        regressor_class=KernelRidge,
        regressor_kwargs=None
    ):
        super().__init__(model, background)
        self.base_explainer = base_explainer
        self.regressor_class = regressor_class
        self.regressor_kwargs = regressor_kwargs or {}
        self.regressor = None
        self.T = background.shape[1]
        self.F = background.shape[2]
        self._fit_surrogate(background)

    def _fit_surrogate(self, X_bg):
        print("[SurroSHAP] Computing SHAP attributions for background data...")
        # Generate pseudo-labels (attributions) for surrogate training
        Y_shap = []
        for x in X_bg:
            shap_val = self.base_explainer.shap_values(x)  # shape (T, F)
            Y_shap.append(shap_val.flatten())
        Y_shap = np.stack(Y_shap, axis=0)  # (N, T*F)
        X_feat = X_bg.reshape(X_bg.shape[0], -1)
        print("[SurroSHAP] Training surrogate regression model...")
        reg = self.regressor_class(**self.regressor_kwargs)
        reg.fit(X_feat, Y_shap)
        self.regressor = reg
        print("[SurroSHAP] Surrogate trained.")

    def shap_values(self, X: Union[np.ndarray, 'torch.Tensor'], **kwargs) -> np.ndarray:
        is_torch = hasattr(X, 'detach')
        X_np = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = False
        if len(X_np.shape) == 2:  # (T, F)
            X_np = X_np[None, ...]
            single = True
        X_feat = X_np.reshape(X_np.shape[0], -1)
        pred = self.regressor.predict(X_feat)
        pred = pred.reshape(X_np.shape[0], self.T, self.F)
        return pred[0] if single else pred

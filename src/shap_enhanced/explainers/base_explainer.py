"""
Abstract base class for SHAP-style explainers.
Defines the core interface and required methods for all explainers in the enhanced SHAP package.
This allows drop-in replacement for shap.Explainer, and ensures consistent input/output structure.

Note: All explainers must implement the `explain` or `shap_values` method with the signature below.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np

class BaseExplainer(ABC):
    """
    Abstract base class for model explainers.
    
    Parameters
    ----------
    model : Any
        The model to be explained.
    background : Optional[Any]
        Background data for marginalization or imputation.
    """
    def __init__(self, model: Any, background: Optional[Any] = None):
        self.model = model
        self.background = background

    @abstractmethod
    def shap_values(
        self, 
        X: Union[np.ndarray, 'torch.Tensor', list], 
        check_additivity: bool = True, 
        **kwargs
    ) -> Union[np.ndarray, list]:
        """
        Compute SHAP values for the input data.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor or list
            Input samples for which to compute SHAP values.
        check_additivity : bool
            Whether to verify that SHAP values sum to model output difference.
        kwargs : dict
            Additional arguments for derived explainers.

        Returns
        -------
        shap_values : np.ndarray or list of np.ndarray
            The computed SHAP values, shape matches input X.
        """
        pass

    def explain(
        self, 
        X: Union[np.ndarray, 'torch.Tensor', list], 
        **kwargs
    ) -> Union[np.ndarray, list]:
        """
        Alias for shap_values, for compatibility and future flexibility.
        """
        return self.shap_values(X, **kwargs)

    def __call__(self, X, **kwargs):
        """
        Allow the explainer to be called as a function, like shap.Explainer.
        """
        return self.shap_values(X, **kwargs)

    @property
    def expected_value(self):
        """
        (Optional) Return the expected value of the model output on the background dataset.
        """
        # Most explainers can compute this, but not all must.
        return getattr(self, "_expected_value", None)

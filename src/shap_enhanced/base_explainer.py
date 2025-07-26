r"""
Enhanced SHAP Base Interface
============================

Overview
--------

This module defines the abstract base class for all SHAP-style explainers  
within the Enhanced SHAP framework. It enforces a common API across all implementations  
to ensure consistency, flexibility, and SHAP compatibility.

Any explainer that inherits from `BaseExplainer` must implement the `shap_values` method,  
which computes SHAP attributions given input data and optional arguments.  
The class also provides useful aliases such as `explain` and a callable `__call__` interface  
to align with `shap.Explainer` behavior.

Key Concepts
^^^^^^^^^^^^

- **Abstract SHAP API**:  
  All custom explainers must subclass this interface and define `shap_values`.

- **Compatibility Wrappers**:  
  Methods like `explain` and `__call__` make the interface flexible for different usage styles.

- **Expected Value Access**:  
  The `expected_value` property allows subclasses to expose the model’s mean output over background data.

Use Case
--------

`BaseExplainer` is the foundation of the enhanced SHAP ecosystem, supporting custom attribution algorithms  
like TimeSHAP, Multi-Baseline SHAP, or Surrogate SHAP. By inheriting from this interface, all explainers  
can be used interchangeably and plugged into benchmarking, visualization, or evaluation tools.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union
import numpy as np

__all__ = ["BaseExplainer"]

class BaseExplainer(ABC):
    r"""
    BaseExplainer: Abstract Interface for SHAP-style Explainers

    This abstract class defines the required interface for all SHAP-style explainers in the enhanced SHAP framework.
    Subclasses must implement the `shap_values` method, and optionally support `expected_value` computation.

    Ensures compatibility with SHAP-style usage patterns such as callable explainers (`explainer(X)`).

    :param Any model: The model to explain (e.g., PyTorch or scikit-learn model).
    :param Optional[Any] background: Background data for imputation or marginalization (used in SHAP computation).
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
        r"""
        Abstract method to compute SHAP values for input samples.

        .. math::
            \phi_i = \mathbb{E}_{S \subseteq N \setminus \{i\}} \left[ f(x_{S \cup \{i\}}) - f(x_S) \right]

        :param X: Input samples to explain (e.g., numpy array, torch tensor, or list).
        :type X: Union[np.ndarray, torch.Tensor, list]
        :param bool check_additivity: If True, verifies SHAP values sum to model prediction difference.
        :param kwargs: Additional arguments for explainer-specific control.
        :return: SHAP values matching the shape and structure of `X`.
        :rtype: Union[np.ndarray, list]
        """
        pass

    def explain(
        self, 
        X: Union[np.ndarray, 'torch.Tensor', list], 
        **kwargs
    ) -> Union[np.ndarray, list]:
        r"""
        Alias to `shap_values` for flexibility and API compatibility.

        :param X: Input samples to explain.
        :type X: Union[np.ndarray, torch.Tensor, list]
        :param kwargs: Additional arguments.
        :return: SHAP values.
        :rtype: Union[np.ndarray, list]
        """
        return self.shap_values(X, **kwargs)

    def __call__(self, X, **kwargs):
        r"""
        Callable interface for explainers.

        Enables usage like `explainer(X)` similar to `shap.Explainer`.

        :param X: Input samples.
        :param kwargs: Additional arguments.
        :return: SHAP values.
        """
        return self.shap_values(X, **kwargs)

    @property
    def expected_value(self):
        r"""
        Optional property returning the expected model output on the background dataset.

        :return: Expected value if defined by the subclass, else None.
        :rtype: float or None
        """
        # Most explainers can compute this, but not all must.
        return getattr(self, "_expected_value", None)

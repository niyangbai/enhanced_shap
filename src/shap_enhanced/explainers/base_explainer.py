"""
BaseExplainer: Unified Abstract Base for SHAP-style Explainers

All custom explainers must inherit from this class to ensure a unified interface.
Supports time-series or tabular models with shape (T, F), with flexible baseline and sampling.

Interface Requirements:
- Accepts `model`, `background`, and all inputs in the constructor.
- Returns attributions of shape (T, F) or (B, T, F) for batched input.
- Supports `.shap_values(x)` and `__call__(x)` for SHAP-style usage.
- Documents masking, sampling, and imputation defaults.
- Accepts baseline selection and mask/sampling strategies via constructor args.
"""

from abc import ABC, abstractmethod
from typing import Any, Optional, Union, Callable
import numpy as np

class BaseExplainer(ABC):
    """
    Abstract base for all SHAP-style explainers.

    Parameters
    ----------
    model : Any
        Model to be explained.
    background : np.ndarray or torch.Tensor or list
        Reference/background dataset for baselines or imputations (shape: (N, T, F) or similar).
    baseline_strategy : str
        How baselines are chosen. Options: 'mean', 'random', 'kmeans', 'user' (default: 'mean').
    n_samples : int
        Number of Monte Carlo samples or coalition samplings (default: 100).
    mask_type : str
        How to perturb/mask features. Options: 'zero', 'mean', 'interpolate', 'custom' (default: 'zero').
    imputer : Optional[Callable]
        Custom imputer for missing/masked values.
    random_seed : Optional[int]
        Random seed for reproducibility.
    Other explainer-specific kwargs may also be accepted.
    """
    def __init__(
        self,
        model: Any,
        background: Optional[Any] = None,
        baseline_strategy: str = "mean",
        n_samples: int = 100,
        mask_type: str = "zero",
        imputer: Optional[Callable] = None,
        random_seed: Optional[int] = 42,
        **kwargs
    ):
        self.model = model
        self.background = background
        self.baseline_strategy = baseline_strategy
        self.n_samples = n_samples
        self.mask_type = mask_type
        self.imputer = imputer
        self.random_seed = random_seed
        np.random.seed(random_seed)
        self._init_baseline()  # Implement as needed in derived class

    def _init_baseline(self):
        """
        Optionally initialize baseline(s) according to baseline_strategy.
        To be overridden by derived classes if needed.
        """
        pass

    @abstractmethod
    def shap_values(
        self,
        X: Union[np.ndarray, 'torch.Tensor', list],
        **kwargs
    ) -> np.ndarray:
        """
        Compute SHAP attributions for input X.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor or list
            Input(s) of shape (T, F) or (B, T, F).
        kwargs : dict
            Additional explainer-specific arguments.

        Returns
        -------
        shap_values : np.ndarray
            Attribution array(s) of shape (T, F) or (B, T, F).
        """
        pass

    def explain(self, X, **kwargs):
        """Alias for shap_values, for compatibility."""
        return self.shap_values(X, **kwargs)

    def __call__(self, X, **kwargs):
        """Allow explainer(X) syntax (SHAP style)."""
        return self.shap_values(X, **kwargs)

    @property
    def expected_value(self):
        """
        Return expected model output on the background set, if applicable.
        """
        return getattr(self, "_expected_value", None)

    @property
    def mask_strategy(self):
        """
        Returns a human-readable description of the masking/imputation method used.
        """
        return self.mask_type

    @property
    def sampling_strategy(self):
        """
        Returns a human-readable description of the coalition sampling method used.
        """
        return getattr(self, "_sampling_strategy", "uniform")

    @property
    def baseline(self):
        """Returns the baseline(s) actually used by the explainer."""
        return getattr(self, "_baseline", self.background)

    def doc(self):
        """Returns docstring for the explainer, including defaults."""
        return self.__doc__


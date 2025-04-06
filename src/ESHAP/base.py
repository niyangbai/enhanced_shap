import numpy as np
from abc import ABC, abstractmethod


class ESHAPAABC(ABC):
    """Abstract base class for SHAP value approximators."""

    @abstractmethod
    def approximate_shap_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Approximate SHAP values for a single sample.

        :param x_sample: Input sample for which to compute SHAP values. Shape (n_features,).
        :type x_sample: np.ndarray
        :return: Approximated SHAP values for the input sample. Shape (n_features,).
        :rtype: np.ndarray
        """
        pass

    @abstractmethod
    def compute_shap_values(self, X_test: np.ndarray) -> np.ndarray:
        """Compute SHAP values for multiple samples.

        :param X_test: Input samples for which to compute SHAP values. Shape (n_samples, n_features).
        :type X_test: np.ndarray
        :return: Approximated SHAP values for the input samples. Shape (n_samples, n_features).
        :rtype: np.ndarray
        """
        pass
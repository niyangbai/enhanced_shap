from ESHAP.base import ESHAPAABC
from typing import Any
import numpy as np
import warnings

class TabularSHAPApproximator(ESHAPAABC):
    """class TabularSHAPApproximator(BaseSHAPApproximator):
    
    Approximates SHAP (SHapley Additive exPlanations) values for tabular data by
    perturbing individual features and observing the model’s prediction change. This method
    estimates each feature’s contribution to the prediction by comparing the prediction on the
    original sample against predictions where one feature at a time is replaced with values
    from a background dataset.
    """
    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        num_samples: int = 100
        ) -> None:
        """Initialize the TabularSHAPApproximator.

        :param model: A predictive model with a `predict` method (e.g., scikit-learn estimator).
        :type model: Any
        :param X_background: A 2D numpy array representing background data used to sample alternative feature values.
        :type X_background: np.ndarray
        :param num_samples: Number of samples to use per feature when approximating SHAP values.
                            Each feature is replaced with `num_samples` values from the background.
        :type num_samples: int

        :return: An instance of TabularSHAPApproximator configured with the model and background data.
        :rtype: TabularSHAPApproximator
        """
        if not isinstance(X_background, np.ndarray) or X_background.ndim != 2:
            raise ValueError("X_background must be a 2D numpy array.")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        self.model = model
        self.X_background = X_background
        self.num_samples = num_samples

        try:
            self.expected_value = np.mean(self.model.predict(X_background))
        except Exception as e:
            warnings.warn(f"Model prediction on X_background failed: {e}")
            self.expected_value = 0.0

    def approximate_shap_values(
        self, 
        x_sample: np.ndarray
        ) -> np.ndarray:
        """Computes approximate SHAP values for a single input sample using perturbation-based estimation.
        For each feature :math:`i`, the SHAP value is approximated by:

        .. math::
            \phi_i \approx f(x) - \mathbb{E}_{x'_i \sim \text{background}}[f(x_{-i}, x'_i)]

        where :math:`x_{-i}` are the original features except feature :math:`i`,
        and :math:`x'_i` are sampled from the background data.

        :param x_sample: A 1D numpy array representing a single input sample.
        :type x_sample: np.ndarray

        :return: A 1D numpy array of SHAP values corresponding to each feature in the input.
        :rtype: np.ndarray
        """
        if not isinstance(x_sample, np.ndarray) or x_sample.ndim != 1:
            raise ValueError("x_sample must be a 1D numpy array.")

        if x_sample.shape[0] != self.X_background.shape[1]:
            raise ValueError(
                "x_sample length must match the number of features in X_background."
            )

        shap_values = np.zeros_like(x_sample)

        try:
            pred_x = self.model.predict(x_sample.reshape(1, -1))[0]
        except Exception as e:
            raise Exception(f"Model prediction failed on x_sample: {e}")

        for i in range(len(x_sample)):
            # Generate modified samples where feature i is replaced
            modified_samples = np.tile(x_sample, (self.num_samples, 1))
            # Replace feature i with values from random background samples
            random_indices = np.random.choice(
                self.X_background.shape[0], size=self.num_samples, replace=True
            )
            modified_samples[:, i] = self.X_background[random_indices, i]
            # Predict using the modified samples
            try:
                preds_modified = self.model.predict(modified_samples)
            except Exception as e:
                raise Exception(f"Model prediction failed on modified samples: {e}")
            # Compute the average prediction when feature i is replaced
            mean_pred_modified = np.mean(preds_modified)
            # Compute the approximate SHAP value for feature i
            shap_values[i] = pred_x - mean_pred_modified

        return shap_values

    def compute_shap_values(
        self, 
        X_test: np.ndarray
        ) -> np.ndarray:
        """Computes approximate SHAP values for a batch of input samples by iteratively applying
        the approximation method to each row in the input matrix.

        :param X_test: A 2D numpy array where each row is a sample to compute SHAP values for.
        :type X_test: np.ndarray

        :return: A 2D numpy array where each row contains SHAP values for the corresponding input sample.
        :rtype: np.ndarray
        """
        if not isinstance(X_test, np.ndarray) or X_test.ndim != 2:
            raise ValueError("X_test must be a 2D numpy array.")

        if X_test.shape[1] != self.X_background.shape[1]:
            raise ValueError(
                "Number of features in X_test must match X_background."
            )

        shap_values_list = []

        for idx, x_sample in enumerate(X_test):
            try:
                shap_values = self.approximate_shap_values(x_sample)
                shap_values_list.append(shap_values)
            except Exception as e:
                warnings.warn(f"Failed to compute SHAP values for sample {idx}: {e}")
                shap_values_list.append(np.zeros_like(x_sample))

        return np.array(shap_values_list)
from ESHAP.base import ESHAPAABC
from typing import Any
import numpy as np
import torch
import warnings


class SequentialSHAPApproximator(ESHAPAABC):
    """Approximates SHAP (SHapley Additive exPlanations) values for models trained on sequential data
    such as time series or sequence-based inputs. The approximation method perturbs specific time-step
    and feature combinations to estimate their contributions to model predictions.
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        num_samples: int = 100
        ) -> None:
        """Initialize the SequentialSHAPApproximator.

        :param model: A sequential model (e.g., PyTorch model) with a callable interface for prediction.
        :type model: Any
        :param X_background: A 3D numpy array used as the background dataset for feature substitution.
                            Shape must be (n_samples, timesteps, features).
        :type X_background: np.ndarray
        :param num_samples: Number of background samples used to approximate the SHAP values for each input.
                            Higher values improve approximation accuracy but increase computation.
        :type num_samples: int

        :return: A SequentialSHAPApproximator instance for SHAP value estimation in sequence models.
        :rtype: SequentialSHAPApproximator
        """
        if not isinstance(X_background, np.ndarray) or X_background.ndim != 3:
            raise ValueError("X_background must be a 3D numpy array.")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        self.model = model
        self.X_background = X_background
        self.num_samples = num_samples

    def approximate_shap_values(
        self, 
        x_sample: np.ndarray
        ) -> np.ndarray:
        """Approximates SHAP values for a single input sequence using feature perturbation.
        For each time step :math:`t` and feature :math:`f`, the SHAP value is approximated by:

        .. math::
            \phi_{t,f} \approx f(x) - \mathbb{E}_{x'_{t,f} \sim \text{background}}[f(x_{-t,f}, x'_{t,f})]

        where :math:`x_{-t,f}` represents the original sequence excluding the specific time-feature
        position being replaced with background samples.

        :param x_sample: A 2D array representing the input sequence, shaped as (timesteps, features).
        :type x_sample: np.ndarray

        :return: A 2D array of SHAP values, corresponding to each (time, feature) pair.
        :rtype: np.ndarray
        """
        if not isinstance(x_sample, np.ndarray) or x_sample.ndim != 2:
            raise ValueError("x_sample must be a 2D numpy array.")

        if x_sample.shape != self.X_background.shape[1:]:
            raise ValueError(
                "x_sample shape must match the shape of sequences in X_background."
            )

        shap_values = np.zeros_like(x_sample)

        try:
            x_sample_tensor = torch.tensor(x_sample, dtype=torch.float32).unsqueeze(0)
            pred_x = self.model(x_sample_tensor).item()
        except Exception as e:
            raise Exception(f"Model prediction failed on x_sample: {e}")

        timesteps, features = x_sample.shape

        for t in range(timesteps):
            for f in range(features):
                # Generate modified sequences where input at time t, feature f is replaced
                modified_samples = np.tile(x_sample, (self.num_samples, 1, 1))
                # Replace input at time t, feature f with values from random background samples
                random_indices = np.random.choice(
                    self.X_background.shape[0], size=self.num_samples, replace=True
                )
                modified_samples[:, t, f] = self.X_background[random_indices, t, f]
                # Convert modified samples to tensor
                modified_samples_tensor = torch.tensor(modified_samples, dtype=torch.float32)
                # Predict using the modified samples
                try:
                    preds_modified = self.model(modified_samples_tensor).detach().numpy().flatten()
                except Exception as e:
                    raise Exception(f"Model prediction failed on modified samples: {e}")
                # Compute the average prediction when input at time t, feature f is replaced
                mean_pred_modified = np.mean(preds_modified)
                # Compute the approximate SHAP value for input at time t, feature f
                shap_values[t, f] = pred_x - mean_pred_modified

        return shap_values

    def compute_shap_values(
        self, 
        X_test: np.ndarray
        ) -> np.ndarray:
        """Computes approximate SHAP values for a batch of sequential input samples.

        Each sequence in the batch is processed independently using the perturbation method,
        and the results are stacked into a 3D output array.

        :param X_test: A 3D array of input sequences with shape (n_samples, timesteps, features).
        :type X_test: np.ndarray

        :return: A 3D array of SHAP values for each sequence, shaped (n_samples, timesteps, features).
        :rtype: np.ndarray
        """
        if not isinstance(X_test, np.ndarray) or X_test.ndim != 3:
            raise ValueError("X_test must be a 3D numpy array.")

        if X_test.shape[1:] != self.X_background.shape[1:]:
            raise ValueError(
                "Sequence shape in X_test must match the shape of sequences in X_background."
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
    
    def batch_approximate_shap_values(
        self, 
        X_test: np.ndarray
        ) -> np.ndarray:
        """Approximates SHAP values in a batched and more efficient manner by replacing each
        (time, feature) pair across all samples in the batch simultaneously.

        This method reduces the number of model evaluations by leveraging vectorized operations
        across the batch, providing a faster alternative to individual SHAP estimation.

        :param X_test: A 3D array of input sequences with shape (n_samples, timesteps, features).
        :type X_test: np.ndarray

        :return: A 3D array of SHAP values with shape (n_samples, timesteps, features).
        :rtype: np.ndarray
        """
        if not isinstance(X_test, np.ndarray) or X_test.ndim != 3:
            raise ValueError("X_test must be a 3D numpy array.")

        if X_test.shape[1:] != self.X_background.shape[1:]:
            raise ValueError(
                "Sequence shape in X_test must match the shape of sequences in X_background."
            )

        batch_shap_values = np.zeros_like(X_test)
        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
        baseline_tensor = torch.tensor(self.X_background, dtype=torch.float32)

        # Forward pass for original predictions
        preds_original = self.model(X_test_tensor).detach().numpy().flatten()

        for t in range(X_test.shape[1]):
            for f in range(X_test.shape[2]):
                # Modify the input batch at time t, feature f with random background values
                modified_X = X_test.copy()
                random_indices = np.random.choice(self.X_background.shape[0], size=X_test.shape[0], replace=True)
                modified_X[:, t, f] = self.X_background[random_indices, t, f]
                modified_X_tensor = torch.tensor(modified_X, dtype=torch.float32)
                # Forward pass for modified predictions
                preds_modified = self.model(modified_X_tensor).detach().numpy().flatten()
                # Compute SHAP values
                batch_shap_values[:, t, f] = preds_original - preds_modified

        return batch_shap_values

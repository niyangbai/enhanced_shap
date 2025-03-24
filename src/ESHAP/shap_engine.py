import numpy as np
import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Any
import warnings


class BaseSHAPApproximator(ABC):
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


class TabularSHAPApproximator(BaseSHAPApproximator):
    """Approximates SHAP values for tabular data models.

    This class provides methods to approximate SHAP values for models trained on tabular data.
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        num_samples: int = 100
    ) -> None:
        """Initialize the TabularSHAPApproximator.

        Args:
            model: The trained model for which SHAP values are to be approximated.
                Must have a `predict` method.
            X_background (np.ndarray): Background dataset used for approximations.
                Shape should be (n_samples, n_features).
            num_samples (int, optional): Number of samples to use in the approximation.
                Defaults to 100.

        Raises:
            ValueError: If `X_background` is not a 2D numpy array.
            ValueError: If `num_samples` is not a positive integer.
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

    def approximate_shap_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Approximate SHAP values for a single sample.

        Args:
            x_sample (np.ndarray): The input sample for which to compute SHAP values.
                Shape should be (n_features,).

        Returns:
            np.ndarray: The approximated SHAP values for the input sample.

        Raises:
            ValueError: If `x_sample` is not a 1D numpy array.
            ValueError: If `x_sample` length does not match the number of features in `X_background`.
            Exception: If model prediction fails.
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

    def compute_shap_values(self, X_test: np.ndarray) -> np.ndarray:
        """Compute SHAP values for multiple samples.

        Args:
            X_test (np.ndarray): The input samples for which to compute SHAP values.
                Shape should be (n_samples, n_features).

        Returns:
            np.ndarray: The approximated SHAP values for the input samples.
                Shape is (n_samples, n_features).

        Raises:
            ValueError: If `X_test` is not a 2D numpy array.
            ValueError: If number of features in `X_test` does not match `X_background`.
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


class SequentialSHAPApproximator(BaseSHAPApproximator):
    """Approximates SHAP values for sequential data models.

    This class provides methods to approximate SHAP values for models trained on sequential data,
    such as time series or sequences.
    """

    def __init__(
        self,
        model: Any,
        X_background: np.ndarray,
        num_samples: int = 100
    ) -> None:
        """Initialize the SequentialSHAPApproximator.

        Args:
            model: The trained model for which SHAP values are to be approximated.
                Must be a PyTorch model.
            X_background (np.ndarray): Background dataset used for approximations.
                Shape should be (n_samples, timesteps, features).
            num_samples (int, optional): Number of samples to use in the approximation.
                Defaults to 100.

        Raises:
            ValueError: If `X_background` is not a 3D numpy array.
            ValueError: If `num_samples` is not a positive integer.
        """
        if not isinstance(X_background, np.ndarray) or X_background.ndim != 3:
            raise ValueError("X_background must be a 3D numpy array.")
        if not isinstance(num_samples, int) or num_samples <= 0:
            raise ValueError("num_samples must be a positive integer.")

        self.model = model
        self.X_background = X_background
        self.num_samples = num_samples

    def approximate_shap_values(self, x_sample: np.ndarray) -> np.ndarray:
        """Approximate SHAP values for a single sequential sample.

        Args:
            x_sample (np.ndarray): The input sample (sequence) for which to compute SHAP values.
                Expected shape is (timesteps, features).

        Returns:
            np.ndarray: The approximated SHAP values for the input sample.
                Shape is (timesteps, features).

        Raises:
            ValueError: If `x_sample` is not a 2D numpy array.
            ValueError: If `x_sample` shape does not match sequences in `X_background`.
            Exception: If model prediction fails.
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

    def compute_shap_values(self, X_test: np.ndarray) -> np.ndarray:
        """Compute SHAP values for multiple sequential samples.

        Args:
            X_test (np.ndarray): The input samples (sequences) for which to compute SHAP values.
                Expected shape is (n_samples, timesteps, features).

        Returns:
            np.ndarray: The approximated SHAP values for the input samples.
                Shape is (n_samples, timesteps, features).

        Raises:
            ValueError: If `X_test` is not a 3D numpy array.
            ValueError: If sequence shape in `X_test` does not match `X_background`.
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
    
    def batch_approximate_shap_values(self, X_test: np.ndarray) -> np.ndarray:
        """Approximate SHAP values for multiple sequential samples in a batch manner.

        Args:
            X_test (np.ndarray): The input samples (sequences) for which to compute SHAP values.
                Expected shape is (n_samples, timesteps, features).

        Returns:
            np.ndarray: The approximated SHAP values for the input samples.
                Shape is (n_samples, timesteps, features).

        Raises:
            ValueError: If `X_test` is not a 3D numpy array.
            ValueError: If sequence shape in `X_test` does not match `X_background`.
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

import numpy as np
from abc import ABC, abstractmethod
from typing import Tuple
import warnings


class BaseSimulator(ABC):
    """Abstract base class for data simulators."""

    @abstractmethod
    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing the features (X) and target (y) data.
        """
        pass

    @abstractmethod
    def get_true_importances(self) -> np.ndarray:
        """Get the true feature importances used in data simulation.

        Returns:
            np.ndarray: The true feature importances.
        """
        pass


class TabularDataSimulator(BaseSimulator):
    """Simulates tabular data with known feature importances."""

    def __init__(self, n_samples: int = 1000, n_features: int = 10) -> None:
        """Initialize the TabularDataSimulator.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 1000.
            n_features (int, optional): Number of features. Defaults to 10.

        Raises:
            ValueError: If `n_samples` or `n_features` are not positive integers.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(n_features, int) or n_features <= 0:
            raise ValueError("n_features must be a positive integer.")

        self.n_samples = n_samples
        self.n_features = n_features
        # Generate true coefficients for features
        self.true_coefficients = np.random.randn(n_features)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated tabular data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features matrix `X` and target vector `y`.

        Raises:
            Exception: If data generation fails.
        """
        try:
            X = np.random.randn(self.n_samples, self.n_features)
            noise = np.random.randn(self.n_samples) * 0.1
            y = X @ self.true_coefficients + noise
            return X, y
        except Exception as e:
            raise Exception(f"Failed to generate data: {e}")

    def get_true_importances(self) -> np.ndarray:
        """Get the normalized absolute true coefficients.

        Returns:
            np.ndarray: Normalized absolute true coefficients.

        Raises:
            Exception: If true coefficients are not available.
        """
        if self.true_coefficients is None:
            raise Exception("True coefficients are not initialized.")
        absolute_coefficients = np.abs(self.true_coefficients)
        sum_coefficients = np.sum(absolute_coefficients)
        if sum_coefficients == 0:
            warnings.warn("Sum of absolute true coefficients is zero. Cannot normalize.")
            return absolute_coefficients
        return absolute_coefficients / sum_coefficients


class SequentialDataSimulator(BaseSimulator):
    """Simulates sequential data with known time-step influences."""

    def __init__(self, n_samples: int = 1000, timesteps: int = 10, features: int = 1) -> None:
        """Initialize the SequentialDataSimulator.

        Args:
            n_samples (int, optional): Number of samples to generate. Defaults to 1000.
            timesteps (int, optional): Number of time steps in each sequence. Defaults to 10.
            features (int, optional): Number of features at each time step. Defaults to 1.

        Raises:
            ValueError: If `n_samples`, `timesteps`, or `features` are not positive integers.
        """
        if not isinstance(n_samples, int) or n_samples <= 0:
            raise ValueError("n_samples must be a positive integer.")
        if not isinstance(timesteps, int) or timesteps <= 0:
            raise ValueError("timesteps must be a positive integer.")
        if not isinstance(features, int) or features <= 0:
            raise ValueError("features must be a positive integer.")

        self.n_samples = n_samples
        self.timesteps = timesteps
        self.features = features
        # Generate true influences over time steps
        self.true_influences = np.random.randn(timesteps, features)

    def get_data(self) -> Tuple[np.ndarray, np.ndarray]:
        """Generate simulated sequential data.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Features tensor `X` and target vector `y`.

        Raises:
            Exception: If data generation fails.
        """
        try:
            X = np.random.randn(self.n_samples, self.timesteps, self.features)
            noise = np.random.randn(self.n_samples) * 0.1
            y = np.sum(X * self.true_influences, axis=(1, 2)) + noise
            return X, y
        except Exception as e:
            raise Exception(f"Failed to generate data: {e}")

    def get_true_importances(self) -> np.ndarray:
        """Get the normalized absolute true influences.

        Returns:
            np.ndarray: Normalized absolute true influences flattened into a 1D array.

        Raises:
            Exception: If true influences are not available.
        """
        if self.true_influences is None:
            raise Exception("True influences are not initialized.")
        true_influences_flat = self.true_influences.flatten()
        absolute_influences = np.abs(true_influences_flat)
        sum_influences = np.sum(absolute_influences)
        if sum_influences == 0:
            warnings.warn("Sum of absolute true influences is zero. Cannot normalize.")
            return absolute_influences
        return absolute_influences / sum_influences

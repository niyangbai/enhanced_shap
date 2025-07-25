"""
Data Processing Utilities
=========================

Common utilities for input/output format handling and data preprocessing.
Extracted from repeated patterns across SHAP explainers.
"""

import numpy as np
import torch
from typing import Union, Tuple, Optional, Any


class InputProcessor:
    """Handles consistent input format processing for SHAP explainers."""
    
    @staticmethod
    def process_input(X: Union[np.ndarray, torch.Tensor]) -> Tuple[np.ndarray, bool]:
        """Process input into consistent format.
        
        Args:
            X: Input data, either (T, F) or (B, T, F)
            
        Returns:
            Tuple of (processed_array, is_single_sample)
            - processed_array: Always 3D with shape (B, T, F)
            - is_single_sample: True if input was 2D
        """
        # Determine if input is torch tensor
        is_torch = hasattr(X, 'detach')
        
        # Convert to numpy
        if is_torch:
            X_np = X.detach().cpu().numpy()
        else:
            X_np = np.asarray(X)
        
        # Handle dimensionality
        if X_np.ndim == 2:
            # Single sample: (T, F) -> (1, T, F)
            X_processed = X_np[None, ...]
            is_single = True
        elif X_np.ndim == 3:
            # Batch: (B, T, F)
            X_processed = X_np
            is_single = False
        else:
            raise ValueError(f"Input must be 2D or 3D, got shape {X_np.shape}")
            
        return X_processed, is_single
    
    @staticmethod
    def validate_input_shape(X: np.ndarray, expected_dims: int = 3):
        """Validate input has expected dimensions."""
        if X.ndim != expected_dims:
            raise ValueError(f"Expected {expected_dims}D input, got {X.ndim}D with shape {X.shape}")
    
    @staticmethod
    def ensure_float32(X: Union[np.ndarray, torch.Tensor]) -> Union[np.ndarray, torch.Tensor]:
        """Ensure input has float32 dtype."""
        if isinstance(X, np.ndarray):
            return X.astype(np.float32)
        elif isinstance(X, torch.Tensor):
            return X.to(dtype=torch.float32)
        else:
            return np.asarray(X, dtype=np.float32)


class OutputProcessor:
    """Handles consistent output format processing for SHAP explainers."""
    
    @staticmethod
    def process_output(output: np.ndarray, was_single_sample: bool) -> np.ndarray:
        """Process output to match original input format.
        
        Args:
            output: Processed output array, shape (B, T, F) or (B, ...)
            was_single_sample: Whether original input was single sample
            
        Returns:
            Output with batch dimension removed if input was single sample
        """
        if was_single_sample and output.ndim > 0:
            return output[0]  # Remove batch dimension
        return output
    
    @staticmethod
    def ensure_numpy_output(output: Union[np.ndarray, torch.Tensor, Any]) -> np.ndarray:
        """Ensure output is numpy array."""
        if hasattr(output, 'detach'):
            return output.detach().cpu().numpy()
        elif hasattr(output, 'cpu'):
            return output.cpu().numpy()
        else:
            return np.asarray(output)


class BackgroundProcessor:
    """Handles background data processing and shape normalization."""
    
    @staticmethod
    def process_background(background: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        """Process background data into consistent format.
        
        Args:
            background: Background data, various shapes possible
            
        Returns:
            Background data with shape (N, T, F)
        """
        # Convert to numpy
        if hasattr(background, 'detach'):
            bg_np = background.detach().cpu().numpy()
        else:
            bg_np = np.asarray(background)
        
        # Handle different input shapes
        if bg_np.ndim == 2:
            # Single background sample: (T, F) -> (1, T, F)
            bg_processed = bg_np[None, ...]
        elif bg_np.ndim == 3:
            # Multiple background samples: (N, T, F)
            bg_processed = bg_np
        else:
            raise ValueError(f"Background data must be 2D or 3D, got shape {bg_np.shape}")
            
        return bg_processed.astype(np.float32)
    
    @staticmethod
    def compute_background_statistics(background: np.ndarray) -> dict:
        """Compute useful statistics from background data.
        
        Args:
            background: Background data, shape (N, T, F)
            
        Returns:
            Dictionary with background statistics
        """
        return {
            'mean': np.mean(background, axis=0),  # (T, F)
            'std': np.std(background, axis=0),    # (T, F)
            'min': np.min(background, axis=0),    # (T, F)
            'max': np.max(background, axis=0),    # (T, F)
            'median': np.median(background, axis=0),  # (T, F)
            'n_samples': background.shape[0]
        }
    
    @staticmethod
    def sample_background(background: np.ndarray, n_samples: int,
                         random_state: Optional[int] = None) -> np.ndarray:
        """Sample subset of background data.
        
        Args:
            background: Background data, shape (N, T, F)
            n_samples: Number of samples to draw
            random_state: Random seed
            
        Returns:
            Sampled background data, shape (n_samples, T, F)
        """
        rng = np.random.RandomState(random_state)
        n_available = background.shape[0]
        
        if n_samples >= n_available:
            return background
        
        indices = rng.choice(n_available, n_samples, replace=False)
        return background[indices]


class DataValidator:
    """Validates data consistency and compatibility."""
    
    @staticmethod
    def validate_shapes_compatible(x: np.ndarray, background: np.ndarray):
        """Validate that input and background have compatible shapes."""
        if x.ndim == 2:
            x_shape = x.shape  # (T, F)
        elif x.ndim == 3:
            x_shape = x.shape[1:]  # (T, F) from (B, T, F)
        else:
            raise ValueError(f"Input must be 2D or 3D, got {x.ndim}D")
        
        if background.ndim == 2:
            bg_shape = background.shape  # (T, F)
        elif background.ndim == 3:
            bg_shape = background.shape[1:]  # (T, F) from (N, T, F)
        else:
            raise ValueError(f"Background must be 2D or 3D, got {background.ndim}D")
        
        if x_shape != bg_shape:
            raise ValueError(f"Shape mismatch: input {x_shape} vs background {bg_shape}")
    
    @staticmethod
    def validate_position_bounds(positions: list, shape: Tuple[int, int]):
        """Validate that positions are within data bounds."""
        T, F = shape
        for t, f in positions:
            if not (0 <= t < T and 0 <= f < F):
                raise ValueError(f"Position ({t}, {f}) out of bounds for shape {shape}")
    
    @staticmethod
    def check_for_nan_inf(data: np.ndarray, data_name: str = "data"):
        """Check for NaN or infinite values in data."""
        if np.any(np.isnan(data)):
            raise ValueError(f"{data_name} contains NaN values")
        if np.any(np.isinf(data)):
            raise ValueError(f"{data_name} contains infinite values")


def process_inputs(X: Union[np.ndarray, torch.Tensor],
                  background: Optional[Union[np.ndarray, torch.Tensor]] = None,
                  validate: bool = True) -> Tuple[np.ndarray, bool, Optional[np.ndarray]]:
    """Convenience function to process all inputs consistently.
    
    Args:
        X: Input data
        background: Optional background data
        validate: Whether to perform validation
        
    Returns:
        Tuple of (processed_X, is_single_sample, processed_background)
    """
    # Process input
    X_processed, is_single = InputProcessor.process_input(X)
    
    # Process background if provided
    background_processed = None
    if background is not None:
        background_processed = BackgroundProcessor.process_background(background)
        
        if validate:
            DataValidator.validate_shapes_compatible(X_processed, background_processed)
    
    # Validation
    if validate:
        DataValidator.check_for_nan_inf(X_processed, "input")
        if background_processed is not None:
            DataValidator.check_for_nan_inf(background_processed, "background")
    
    return X_processed, is_single, background_processed


__all__ = [
    "InputProcessor",
    "OutputProcessor", 
    "BackgroundProcessor",
    "DataValidator",
    "process_inputs"
]
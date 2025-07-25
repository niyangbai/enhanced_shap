"""
Model Evaluation Utilities
==========================

Common utilities for model output evaluation with device/tensor handling.
Extracted from repeated patterns across all SHAP explainers.
"""

import numpy as np
import torch
from typing import Union, Optional, Any


class ModelEvaluator:
    """Handles model evaluation with consistent device and tensor management."""
    
    def __init__(self, model: Any, device: Optional[Union[str, torch.device]] = None):
        """
        Args:
            model: The model to evaluate
            device: Device for computation ("cpu", "cuda", etc.)
        """
        self.model = model
        self.device = self._get_device(device)
        
    def _get_device(self, device: Optional[Union[str, torch.device]]) -> torch.device:
        """Get appropriate device for computation."""
        if device is None:
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        elif isinstance(device, str):
            return torch.device(device)
        else:
            return device
    
    def evaluate(self, X: Union[np.ndarray, torch.Tensor], 
                 batch_size: Optional[int] = None) -> np.ndarray:
        """Evaluate model on input with proper device/tensor handling.
        
        Args:
            X: Input data, shape (B, T, F) or (T, F)
            batch_size: Optional batch size for processing
            
        Returns:
            Model output as numpy array
        """
        # Handle single sample case
        single_sample = False
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = X[None, ...]  # Add batch dimension
            single_sample = True
        elif isinstance(X, torch.Tensor) and X.ndim == 2:
            X = X.unsqueeze(0)  # Add batch dimension
            single_sample = True
            
        # Convert to tensor and move to device
        if isinstance(X, np.ndarray):
            X_tensor = torch.tensor(X, dtype=torch.float32, device=self.device)
        elif isinstance(X, torch.Tensor):
            X_tensor = X.to(device=self.device, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
        
        # Evaluate model
        if batch_size is None or X_tensor.shape[0] <= batch_size:
            output = self._single_evaluation(X_tensor)
        else:
            output = self._batch_evaluation(X_tensor, batch_size)
        
        # Convert to numpy and handle single sample case
        if hasattr(output, "cpu"):
            output_np = output.cpu().numpy()
        else:
            output_np = np.asarray(output)
            
        if single_sample and output_np.ndim > 0:
            output_np = output_np[0]  # Remove batch dimension
            
        return output_np
    
    def _single_evaluation(self, X_tensor: torch.Tensor) -> torch.Tensor:
        """Single forward pass evaluation."""
        with torch.no_grad():
            return self.model(X_tensor)
    
    def _batch_evaluation(self, X_tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
        """Batched evaluation for large inputs."""
        outputs = []
        n_samples = X_tensor.shape[0]
        
        with torch.no_grad():
            for i in range(0, n_samples, batch_size):
                batch = X_tensor[i:i + batch_size]
                batch_output = self.model(batch)
                outputs.append(batch_output)
                
        return torch.cat(outputs, dim=0)
    
    def evaluate_single(self, x: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        """Evaluate model on single sample, return scalar if possible."""
        output = self.evaluate(x)
        if output.shape == () or (output.ndim == 1 and output.shape[0] == 1):
            return float(output)
        return output
    
    def get_expected_value(self, background_data: Union[np.ndarray, torch.Tensor]) -> Union[float, np.ndarray]:
        """Compute expected value over background data."""
        background_output = self.evaluate(background_data)
        return np.mean(background_output, axis=0)


def evaluate_model(model: Any, X: Union[np.ndarray, torch.Tensor], 
                  device: Optional[Union[str, torch.device]] = None,
                  batch_size: Optional[int] = None) -> np.ndarray:
    """Convenience function for model evaluation."""
    evaluator = ModelEvaluator(model, device)
    return evaluator.evaluate(X, batch_size)


def convert_to_tensor(X: Union[np.ndarray, torch.Tensor], 
                     device: Optional[Union[str, torch.device]] = None,
                     dtype: torch.dtype = torch.float32) -> torch.Tensor:
    """Convert input to tensor with proper device placement."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif isinstance(device, str):
        device = torch.device(device)
        
    if isinstance(X, np.ndarray):
        return torch.tensor(X, dtype=dtype, device=device)
    elif isinstance(X, torch.Tensor):
        return X.to(device=device, dtype=dtype)
    else:
        raise ValueError(f"Unsupported input type: {type(X)}")


def convert_to_numpy(X: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """Convert tensor to numpy array."""
    if isinstance(X, torch.Tensor):
        return X.detach().cpu().numpy()
    elif isinstance(X, np.ndarray):
        return X
    else:
        return np.asarray(X)


__all__ = [
    "ModelEvaluator",
    "evaluate_model",
    "convert_to_tensor", 
    "convert_to_numpy"
]
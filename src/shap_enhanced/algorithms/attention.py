"""
Attention and Importance Computation
===================================

Methods for computing feature attention/importance for weighted sampling.
Extracted from attention computation patterns in various explainers.
"""

import numpy as np
import torch
from typing import Union, Optional, Any
from .model_evaluation import ModelEvaluator, convert_to_tensor


class BaseAttention:
    """Abstract base class for attention computation methods."""
    
    def compute_attention(self, x: np.ndarray, model: Any) -> np.ndarray:
        """Compute attention weights for input features.
        
        Args:
            x: Input array, shape (T, F)
            model: Model to analyze
            
        Returns:
            Attention weights, same shape as input
        """
        raise NotImplementedError


class GradientAttention(BaseAttention):
    """Gradient-based attention computation."""
    
    def __init__(self, device: Optional[Union[str, torch.device]] = None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def compute_attention(self, x: np.ndarray, model: Any) -> np.ndarray:
        """Compute attention using input gradients.
        
        Args:
            x: Input array, shape (T, F)
            model: PyTorch model
            
        Returns:
            Gradient-based attention weights
        """
        # Convert to tensor with gradient tracking
        x_tensor = convert_to_tensor(x, self.device)
        x_tensor.requires_grad_(True)
        
        # Forward pass
        output = model(x_tensor.unsqueeze(0))  # Add batch dimension
        
        # Handle different output types
        if output.dim() > 1:
            # Multi-output case: use sum of outputs
            scalar_output = output.sum()
        else:
            scalar_output = output.squeeze()
        
        # Backward pass
        scalar_output.backward()
        
        # Get gradients and compute attention
        gradients = x_tensor.grad.detach().cpu().numpy()
        attention = np.abs(gradients)
        
        # Normalize attention weights
        attention = attention / (np.sum(attention) + 1e-8)
        
        return attention
    
    def compute_integrated_gradients(self, x: np.ndarray, baseline: np.ndarray,
                                   model: Any, n_steps: int = 50) -> np.ndarray:
        """Compute integrated gradients for more stable attention.
        
        Args:
            x: Input array, shape (T, F)
            baseline: Baseline array, same shape as x
            model: PyTorch model
            n_steps: Number of integration steps
            
        Returns:
            Integrated gradient attention weights
        """
        # Create interpolation path
        alphas = np.linspace(0, 1, n_steps)
        gradients = []
        
        for alpha in alphas:
            # Interpolate between baseline and input
            interpolated = baseline + alpha * (x - baseline)
            
            # Convert to tensor with gradient tracking
            x_tensor = convert_to_tensor(interpolated, self.device)
            x_tensor.requires_grad_(True)
            
            # Forward and backward pass
            output = model(x_tensor.unsqueeze(0))
            if output.dim() > 1:
                scalar_output = output.sum()
            else:
                scalar_output = output.squeeze()
            
            scalar_output.backward()
            gradients.append(x_tensor.grad.detach().cpu().numpy())
        
        # Average gradients and multiply by input difference
        avg_gradients = np.mean(gradients, axis=0)
        integrated_grads = avg_gradients * (x - baseline)
        
        # Convert to attention weights
        attention = np.abs(integrated_grads)
        attention = attention / (np.sum(attention) + 1e-8)
        
        return attention


class InputMagnitudeAttention(BaseAttention):
    """Input magnitude-based attention computation."""
    
    def __init__(self, aggregation: str = "sum"):
        """
        Args:
            aggregation: How to aggregate across features ("sum", "max", "mean")
        """
        self.aggregation = aggregation
    
    def compute_attention(self, x: np.ndarray, model: Any = None) -> np.ndarray:
        """Compute attention based on input magnitudes.
        
        Args:
            x: Input array, shape (T, F)
            model: Not used in this method
            
        Returns:
            Magnitude-based attention weights
        """
        # Use absolute values
        abs_x = np.abs(x)
        
        if self.aggregation == "sum":
            # Sum across features for each time step, then broadcast
            time_importance = np.sum(abs_x, axis=1, keepdims=True)  # (T, 1)
            attention = abs_x * time_importance  # (T, F)
        elif self.aggregation == "max":
            attention = abs_x
        elif self.aggregation == "mean":
            # Mean-centered magnitude
            mean_mag = np.mean(abs_x)
            attention = np.maximum(abs_x - mean_mag, 0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        
        # Normalize
        attention = attention / (np.sum(attention) + 1e-8)
        
        return attention


class PerturbationAttention(BaseAttention):
    """Perturbation-based attention using local sensitivity."""
    
    def __init__(self, perturbation_size: float = 0.01, n_samples: int = 10,
                 device: Optional[Union[str, torch.device]] = None):
        """
        Args:
            perturbation_size: Size of perturbations
            n_samples: Number of perturbation samples per feature
            device: Device for computation
        """
        self.perturbation_size = perturbation_size
        self.n_samples = n_samples
        self.model_evaluator = None
        self.device = device
    
    def compute_attention(self, x: np.ndarray, model: Any) -> np.ndarray:
        """Compute attention using perturbation sensitivity.
        
        Args:
            x: Input array, shape (T, F)
            model: Model to analyze
            
        Returns:
            Perturbation-based attention weights
        """
        if self.model_evaluator is None:
            self.model_evaluator = ModelEvaluator(model, self.device)
        
        # Get baseline prediction
        baseline_pred = self.model_evaluator.evaluate_single(x)
        
        attention = np.zeros_like(x)
        T, F = x.shape
        
        # Compute sensitivity for each position
        for t in range(T):
            for f in range(F):
                sensitivities = []
                
                for _ in range(self.n_samples):
                    # Create perturbation
                    perturbation = np.random.normal(0, self.perturbation_size)
                    x_pert = x.copy()
                    x_pert[t, f] += perturbation
                    
                    # Compute prediction change
                    pert_pred = self.model_evaluator.evaluate_single(x_pert)
                    sensitivity = abs(pert_pred - baseline_pred) / abs(perturbation + 1e-8)
                    sensitivities.append(sensitivity)
                
                attention[t, f] = np.mean(sensitivities)
        
        # Normalize
        attention = attention / (np.sum(attention) + 1e-8)
        
        return attention


class TemporalAttention(BaseAttention):
    """Temporal attention based on sequence patterns."""
    
    def __init__(self, window_size: int = 3, decay_factor: float = 0.9):
        """
        Args:
            window_size: Size of temporal window
            decay_factor: Temporal decay factor
        """
        self.window_size = window_size
        self.decay_factor = decay_factor
    
    def compute_attention(self, x: np.ndarray, model: Any = None) -> np.ndarray:
        """Compute temporal attention weights.
        
        Args:
            x: Input array, shape (T, F)
            model: Not used in this method
            
        Returns:
            Temporal attention weights
        """
        T, F = x.shape
        attention = np.zeros_like(x)
        
        # Compute temporal importance based on local variability
        for t in range(T):
            # Define temporal window
            window_start = max(0, t - self.window_size // 2)
            window_end = min(T, t + self.window_size // 2 + 1)
            
            # Compute local variability
            if window_end - window_start > 1:
                window_data = x[window_start:window_end]
                local_var = np.var(window_data, axis=0)
            else:
                local_var = np.abs(x[t])
            
            # Apply temporal decay
            for t2 in range(T):
                distance = abs(t2 - t)
                decay = self.decay_factor ** distance
                attention[t2] += local_var * decay
        
        # Normalize
        attention = attention / (np.sum(attention) + 1e-8)
        
        return attention


def compute_attention_weights(x: np.ndarray, model: Any, method: str = "gradient",
                            device: Optional[Union[str, torch.device]] = None,
                            **kwargs) -> np.ndarray:
    """Convenience function for computing attention weights.
    
    Args:
        x: Input array, shape (T, F)
        model: Model to analyze
        method: Attention method ("gradient", "magnitude", "perturbation", "temporal")
        device: Device for computation
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Attention weights, same shape as input
    """
    attention_methods = {
        "gradient": GradientAttention,
        "magnitude": InputMagnitudeAttention,
        "perturbation": PerturbationAttention,
        "temporal": TemporalAttention
    }
    
    if method not in attention_methods:
        raise ValueError(f"Unknown attention method: {method}")
    
    if method == "gradient":
        attention_computer = attention_methods[method](device=device, **kwargs)
    else:
        attention_computer = attention_methods[method](**kwargs)
    
    return attention_computer.compute_attention(x, model)


__all__ = [
    "BaseAttention",
    "GradientAttention",
    "InputMagnitudeAttention", 
    "PerturbationAttention",
    "TemporalAttention",
    "compute_attention_weights"
]
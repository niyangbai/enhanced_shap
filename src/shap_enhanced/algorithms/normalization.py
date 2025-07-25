"""
SHAP Additivity Normalization
=============================

Normalization utilities to enforce SHAP additivity constraint.
Extracted from common normalization patterns across all explainers.
"""

import numpy as np
import torch
from typing import Union, Optional, Any
from .model_evaluation import ModelEvaluator, convert_to_numpy


class AdditivityNormalizer:
    """Enforces SHAP additivity constraint: sum(shap_values) = f(x) - f(baseline)."""
    
    def __init__(self, model: Any, device: Optional[Union[str, Any]] = None):
        """
        Args:
            model: Model being explained
            device: Device for computation
        """
        self.model = model
        self.model_evaluator = ModelEvaluator(model, device)
    
    def normalize_additive(self, shap_values: np.ndarray, x: np.ndarray, 
                          baseline: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
        """Apply additivity normalization to SHAP values.
        
        Args:
            shap_values: SHAP values to normalize, shape (T, F)
            x: Original input, shape (T, F)
            baseline: Baseline/reference input, shape (T, F)
            tolerance: Tolerance for numerical stability
            
        Returns:
            Normalized SHAP values maintaining additivity
        """
        # Compute model predictions
        orig_pred = self.model_evaluator.evaluate_single(x)
        baseline_pred = self.model_evaluator.evaluate_single(baseline)
        
        # Expected difference
        expected_diff = orig_pred - baseline_pred
        
        # Current sum of SHAP values
        current_sum = np.sum(shap_values)
        
        # Apply normalization if sum is significantly different
        if abs(current_sum) > tolerance:
            normalized_shap = shap_values * (expected_diff / current_sum)
        else:
            # If sum is near zero, distribute difference equally
            n_features = shap_values.size
            normalized_shap = shap_values + (expected_diff / n_features)
            
        return normalized_shap
    
    def normalize_proportional(self, shap_values: np.ndarray, x: np.ndarray,
                              baseline: np.ndarray, tolerance: float = 1e-8) -> np.ndarray:
        """Proportional normalization preserving relative magnitudes.
        
        Args:
            shap_values: SHAP values to normalize, shape (T, F)
            x: Original input, shape (T, F)
            baseline: Baseline/reference input, shape (T, F)
            tolerance: Tolerance for numerical stability
            
        Returns:
            Proportionally normalized SHAP values
        """
        orig_pred = self.model_evaluator.evaluate_single(x)
        baseline_pred = self.model_evaluator.evaluate_single(baseline)
        expected_diff = orig_pred - baseline_pred
        
        current_sum = np.sum(shap_values)
        
        if abs(current_sum) > tolerance:
            # Scale all values proportionally
            scale_factor = expected_diff / current_sum
            return shap_values * scale_factor
        else:
            # Distribute equally if sum is near zero
            return np.full_like(shap_values, expected_diff / shap_values.size)
    
    def normalize_truncated(self, shap_values: np.ndarray, x: np.ndarray,
                           baseline: np.ndarray, max_adjustment: float = 0.1) -> np.ndarray:
        """Truncated normalization with limited adjustment per feature.
        
        Args:
            shap_values: SHAP values to normalize, shape (T, F)
            x: Original input, shape (T, F)
            baseline: Baseline/reference input, shape (T, F)
            max_adjustment: Maximum relative adjustment per feature
            
        Returns:
            Normalized SHAP values with limited adjustments
        """
        orig_pred = self.model_evaluator.evaluate_single(x)
        baseline_pred = self.model_evaluator.evaluate_single(baseline)
        expected_diff = orig_pred - baseline_pred
        
        current_sum = np.sum(shap_values)
        adjustment_needed = expected_diff - current_sum
        
        if abs(adjustment_needed) < 1e-8:
            return shap_values
        
        # Compute adjustment per feature, limited by max_adjustment
        normalized_shap = shap_values.copy()
        n_features = shap_values.size
        
        # Distribute adjustment proportionally with limits
        feature_adjustments = np.zeros_like(shap_values)
        remaining_adjustment = adjustment_needed
        
        for _ in range(10):  # Iterative adjustment with max iterations
            if abs(remaining_adjustment) < 1e-8:
                break
                
            per_feature_adjustment = remaining_adjustment / n_features
            
            for i, (t, f) in enumerate(np.ndindex(shap_values.shape)):
                original_value = shap_values[t, f]
                max_change = abs(original_value) * max_adjustment + 1e-6
                
                actual_adjustment = np.clip(per_feature_adjustment, 
                                           -max_change, max_change)
                feature_adjustments[t, f] = actual_adjustment
                remaining_adjustment -= actual_adjustment
        
        return normalized_shap + feature_adjustments


class MultiBaselineNormalizer:
    """Normalization for multi-baseline SHAP approaches."""
    
    def __init__(self, model: Any, device: Optional[Union[str, Any]] = None):
        self.model = model
        self.model_evaluator = ModelEvaluator(model, device)
    
    def normalize_multi_baseline(self, shap_values: np.ndarray, x: np.ndarray,
                                baselines: np.ndarray, weights: Optional[np.ndarray] = None) -> np.ndarray:
        """Normalize SHAP values for multiple baselines.
        
        Args:
            shap_values: SHAP values, shape (T, F)
            x: Original input, shape (T, F)
            baselines: Multiple baselines, shape (N, T, F)
            weights: Optional weights for baselines, shape (N,)
            
        Returns:
            Normalized SHAP values
        """
        orig_pred = self.model_evaluator.evaluate_single(x)
        baseline_preds = self.model_evaluator.evaluate(baselines)
        
        if weights is None:
            weights = np.ones(len(baselines)) / len(baselines)
        else:
            weights = weights / np.sum(weights)
        
        # Weighted average baseline prediction
        weighted_baseline_pred = np.average(baseline_preds, weights=weights)
        expected_diff = orig_pred - weighted_baseline_pred
        
        current_sum = np.sum(shap_values)
        
        if abs(current_sum) > 1e-8:
            return shap_values * (expected_diff / current_sum)
        else:
            return np.full_like(shap_values, expected_diff / shap_values.size)


def normalize_shap_values(shap_values: np.ndarray, model: Any, x: np.ndarray,
                         baseline: np.ndarray, method: str = "additive",
                         device: Optional[Union[str, Any]] = None,
                         **kwargs) -> np.ndarray:
    """Convenience function for SHAP value normalization.
    
    Args:
        shap_values: SHAP values to normalize
        model: Model being explained
        x: Original input
        baseline: Baseline input
        method: Normalization method ("additive", "proportional", "truncated")
        device: Device for computation
        **kwargs: Additional arguments for specific methods
        
    Returns:
        Normalized SHAP values
    """
    normalizer = AdditivityNormalizer(model, device)
    
    if method == "additive":
        return normalizer.normalize_additive(shap_values, x, baseline, **kwargs)
    elif method == "proportional":
        return normalizer.normalize_proportional(shap_values, x, baseline, **kwargs)
    elif method == "truncated":
        return normalizer.normalize_truncated(shap_values, x, baseline, **kwargs)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def check_additivity(shap_values: np.ndarray, model: Any, x: np.ndarray,
                    baseline: np.ndarray, tolerance: float = 1e-6,
                    device: Optional[Union[str, Any]] = None) -> bool:
    """Check if SHAP values satisfy additivity constraint.
    
    Args:
        shap_values: SHAP values to check
        model: Model being explained
        x: Original input
        baseline: Baseline input
        tolerance: Tolerance for additivity check
        device: Device for computation
        
    Returns:
        True if additivity is satisfied within tolerance
    """
    evaluator = ModelEvaluator(model, device)
    
    orig_pred = evaluator.evaluate_single(x)
    baseline_pred = evaluator.evaluate_single(baseline)
    expected_diff = orig_pred - baseline_pred
    
    actual_sum = np.sum(shap_values)
    
    return abs(actual_sum - expected_diff) <= tolerance


__all__ = [
    "AdditivityNormalizer",
    "MultiBaselineNormalizer",
    "normalize_shap_values",
    "check_additivity"
]
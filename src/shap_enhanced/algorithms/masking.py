"""
Feature Masking and Imputation Strategies
=========================================

Common masking/imputation methods extracted from SHAP explainers.
Each masker defines how to replace feature values when they're excluded from coalitions.
"""

import numpy as np
import torch
from abc import ABC, abstractmethod
from typing import List, Tuple, Union, Optional


class BaseMasker(ABC):
    """Abstract base class for feature masking strategies."""
    
    @abstractmethod
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Apply masking to specified feature positions.
        
        Args:
            x: Input array of shape (T, F) 
            mask_positions: List of (time, feature) positions to mask
            
        Returns:
            Masked version of input
        """
        pass


class ZeroMasker(BaseMasker):
    """Hard zero masking - most common pattern."""
    
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Replace masked positions with zeros."""
        x_masked = x.copy()
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                x_masked[t, f] = 0.0
        return x_masked


class MeanMasker(BaseMasker):
    """Background mean imputation masking."""
    
    def __init__(self, background_mean: np.ndarray):
        """
        Args:
            background_mean: Mean values for imputation, shape (T, F)
        """
        self.background_mean = background_mean
    
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Replace masked positions with background mean."""
        x_masked = x.copy()
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                x_masked[t, f] = self.background_mean[t, f]
        return x_masked


class AdaptiveMasker(BaseMasker):
    """Adaptive masking that learns appropriate baseline values."""
    
    def __init__(self, background_data: np.ndarray, adaptation_rate: float = 0.1):
        """
        Args:
            background_data: Background dataset, shape (N, T, F)
            adaptation_rate: Rate of baseline adaptation
        """
        self.background_data = background_data
        self.adaptation_rate = adaptation_rate
        self.baseline = np.mean(background_data, axis=0)  # (T, F)
        
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Replace masked positions with adaptive baseline."""
        x_masked = x.copy()
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                # Adapt baseline based on current input
                current_baseline = self.baseline[t, f]
                adapted_value = (1 - self.adaptation_rate) * current_baseline + \
                               self.adaptation_rate * x[t, f]
                x_masked[t, f] = adapted_value
        return x_masked


class InterpolationMasker(BaseMasker):
    """Temporal interpolation masking for sequential data."""
    
    def __init__(self, method: str = "linear"):
        """
        Args:
            method: Interpolation method ("linear", "nearest", "cubic")
        """
        self.method = method
    
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Replace masked positions with interpolated values."""
        x_masked = x.copy()
        
        # Group by feature for efficient interpolation
        positions_by_feature = {}
        for t, f in mask_positions:
            if f not in positions_by_feature:
                positions_by_feature[f] = []
            positions_by_feature[f].append(t)
        
        for feature_idx, time_positions in positions_by_feature.items():
            if feature_idx >= x.shape[1]:
                continue
                
            # Get unmasked time points for this feature
            all_times = set(range(x.shape[0]))
            masked_times = set(time_positions)
            unmasked_times = sorted(all_times - masked_times)
            
            if len(unmasked_times) >= 2:
                # Interpolate between unmasked points
                unmasked_values = x_masked[unmasked_times, feature_idx]
                
                for t in time_positions:
                    if 0 <= t < x.shape[0]:
                        # Find closest unmasked neighbors
                        before = [ut for ut in unmasked_times if ut < t]
                        after = [ut for ut in unmasked_times if ut > t]
                        
                        if before and after:
                            # Linear interpolation
                            t_before, t_after = before[-1], after[0]
                            v_before, v_after = x_masked[t_before, feature_idx], x_masked[t_after, feature_idx]
                            alpha = (t - t_before) / (t_after - t_before)
                            x_masked[t, feature_idx] = (1 - alpha) * v_before + alpha * v_after
                        elif before:
                            # Use last available value
                            x_masked[t, feature_idx] = x_masked[before[-1], feature_idx]
                        elif after:
                            # Use next available value
                            x_masked[t, feature_idx] = x_masked[after[0], feature_idx]
            else:
                # Not enough points for interpolation, use zero or mean
                for t in time_positions:
                    if 0 <= t < x.shape[0]:
                        x_masked[t, feature_idx] = 0.0
                        
        return x_masked


class ConditionalMasker(BaseMasker):
    """Empirical conditional distribution masking."""
    
    def __init__(self, background_data: np.ndarray, n_neighbors: int = 5):
        """
        Args:
            background_data: Background dataset for conditional sampling, shape (N, T, F)
            n_neighbors: Number of nearest neighbors to use
        """
        self.background_data = background_data
        self.n_neighbors = n_neighbors
    
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Replace masked positions using conditional distribution from background."""
        x_masked = x.copy()
        
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                # Find similar samples in background data based on unmasked features
                unmasked_positions = [(t2, f2) for t2 in range(x.shape[0]) for f2 in range(x.shape[1]) 
                                    if (t2, f2) not in mask_positions]
                
                if unmasked_positions:
                    # Compute distances to background samples
                    distances = []
                    for bg_sample in self.background_data:
                        dist = 0.0
                        for t2, f2 in unmasked_positions:
                            dist += (x[t2, f2] - bg_sample[t2, f2]) ** 2
                        distances.append(np.sqrt(dist))
                    
                    # Get k nearest neighbors
                    neighbor_indices = np.argsort(distances)[:self.n_neighbors]
                    neighbor_values = [self.background_data[i][t, f] for i in neighbor_indices]
                    x_masked[t, f] = np.mean(neighbor_values)
                else:
                    x_masked[t, f] = 0.0
                    
        return x_masked


class BackgroundSamplingMasker(BaseMasker):
    """Random sampling from background distribution."""
    
    def __init__(self, background_data: np.ndarray, random_state: Optional[int] = None):
        """
        Args:
            background_data: Background dataset, shape (N, T, F)
        """
        self.background_data = background_data
        self.rng = np.random.RandomState(random_state)
    
    def mask_features(self, x: np.ndarray, mask_positions: List[Tuple[int, int]]) -> np.ndarray:
        """Replace masked positions with random background samples."""
        x_masked = x.copy()
        
        for t, f in mask_positions:
            if 0 <= t < x.shape[0] and 0 <= f < x.shape[1]:
                # Sample random background value for this position
                bg_idx = self.rng.randint(len(self.background_data))
                x_masked[t, f] = self.background_data[bg_idx, t, f]
                
        return x_masked


def create_masker(masker_type: str, **kwargs) -> BaseMasker:
    """Factory function for creating maskers."""
    masker_classes = {
        "zero": ZeroMasker,
        "mean": MeanMasker,
        "adaptive": AdaptiveMasker,
        "interpolation": InterpolationMasker,
        "conditional": ConditionalMasker,
        "background_sampling": BackgroundSamplingMasker
    }
    
    if masker_type not in masker_classes:
        raise ValueError(f"Unknown masker type: {masker_type}")
        
    return masker_classes[masker_type](**kwargs)


__all__ = [
    "BaseMasker",
    "ZeroMasker",
    "MeanMasker", 
    "AdaptiveMasker",
    "InterpolationMasker",
    "ConditionalMasker",
    "BackgroundSamplingMasker",
    "create_masker"
]
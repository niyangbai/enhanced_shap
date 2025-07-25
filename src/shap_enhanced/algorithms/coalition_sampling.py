"""
Coalition Sampling Algorithms
============================

Common algorithms for sampling feature coalitions in SHAP explainers.
Extracted from repeated patterns across multiple explainer implementations.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Union


class BaseCoalitionSampler(ABC):
    """Abstract base class for coalition sampling strategies."""
    
    @abstractmethod
    def sample_coalition(self, all_positions: List[Tuple[int, int]], 
                        exclude: Optional[Tuple[int, int]] = None,
                        size_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Sample a coalition of feature positions.
        
        Args:
            all_positions: All possible (time, feature) positions
            exclude: Position to exclude from coalition (target feature)
            size_range: Optional (min_size, max_size) for coalition size
            
        Returns:
            List of (time, feature) positions in the coalition
        """
        pass


class RandomCoalitionSampler(BaseCoalitionSampler):
    """Uniform random coalition sampling - most common pattern."""
    
    def __init__(self, random_state: Optional[int] = None):
        self.rng = np.random.RandomState(random_state)
    
    def sample_coalition(self, all_positions: List[Tuple[int, int]], 
                        exclude: Optional[Tuple[int, int]] = None,
                        size_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Sample random coalition excluding target feature."""
        available = [pos for pos in all_positions if pos != exclude] if exclude else all_positions
        
        if not available:
            return []
            
        if size_range:
            min_size, max_size = size_range
            min_size = max(1, min(min_size, len(available)))
            max_size = min(max_size, len(available))
            k = self.rng.randint(min_size, max_size + 1)
        else:
            k = self.rng.randint(1, len(available) + 1)
            
        return [available[i] for i in self.rng.choice(len(available), k, replace=False)]


class WeightedCoalitionSampler(BaseCoalitionSampler):
    """Attention/importance-weighted coalition sampling."""
    
    def __init__(self, weights: np.ndarray, random_state: Optional[int] = None):
        """
        Args:
            weights: Importance weights for each position, shape (T, F)
        """
        self.weights = weights
        self.rng = np.random.RandomState(random_state)
    
    def sample_coalition(self, all_positions: List[Tuple[int, int]], 
                        exclude: Optional[Tuple[int, int]] = None,
                        size_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Sample coalition with probability proportional to feature importance."""
        available = [pos for pos in all_positions if pos != exclude] if exclude else all_positions
        
        if not available:
            return []
            
        # Get weights for available positions
        position_weights = np.array([self.weights[t, f] for t, f in available])
        position_weights = position_weights / (position_weights.sum() + 1e-8)
        
        if size_range:
            min_size, max_size = size_range
            min_size = max(1, min(min_size, len(available)))
            max_size = min(max_size, len(available))
            k = self.rng.randint(min_size, max_size + 1)
        else:
            k = self.rng.randint(1, len(available) + 1)
            
        selected_indices = self.rng.choice(len(available), k, replace=False, p=position_weights)
        return [available[i] for i in selected_indices]


class ConstrainedCoalitionSampler(BaseCoalitionSampler):
    """Coalition sampling with structural constraints (e.g., temporal locality)."""
    
    def __init__(self, constraint_type: str = "temporal", window_size: int = 3, 
                 random_state: Optional[int] = None):
        """
        Args:
            constraint_type: Type of constraint ("temporal", "spatial", etc.)
            window_size: Size of constraint window
        """
        self.constraint_type = constraint_type
        self.window_size = window_size
        self.rng = np.random.RandomState(random_state)
    
    def sample_coalition(self, all_positions: List[Tuple[int, int]], 
                        exclude: Optional[Tuple[int, int]] = None,
                        size_range: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """Sample coalition respecting structural constraints."""
        available = [pos for pos in all_positions if pos != exclude] if exclude else all_positions
        
        if not available:
            return []
            
        if self.constraint_type == "temporal" and exclude:
            # Sample positions within temporal window of target
            target_t, target_f = exclude
            constrained = [
                (t, f) for t, f in available 
                if abs(t - target_t) <= self.window_size
            ]
            available = constrained if constrained else available
            
        if size_range:
            min_size, max_size = size_range
            min_size = max(1, min(min_size, len(available)))
            max_size = min(max_size, len(available))
            k = self.rng.randint(min_size, max_size + 1)
        else:
            k = self.rng.randint(1, len(available) + 1)
            
        return [available[i] for i in self.rng.choice(len(available), k, replace=False)]


def create_all_positions(T: int, F: int) -> List[Tuple[int, int]]:
    """Create all possible (time, feature) positions for given dimensions."""
    return [(t, f) for t in range(T) for f in range(F)]


def sample_random_coalition(all_positions: List[Tuple[int, int]], 
                           exclude: Optional[Tuple[int, int]] = None,
                           size_range: Optional[Tuple[int, int]] = None,
                           random_state: Optional[int] = None) -> List[Tuple[int, int]]:
    """Convenience function for random coalition sampling."""
    sampler = RandomCoalitionSampler(random_state)
    return sampler.sample_coalition(all_positions, exclude, size_range)


__all__ = [
    "BaseCoalitionSampler",
    "RandomCoalitionSampler", 
    "WeightedCoalitionSampler",
    "ConstrainedCoalitionSampler",
    "create_all_positions",
    "sample_random_coalition"
]
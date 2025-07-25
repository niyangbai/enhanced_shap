"""
Core SHAP Value Computation
===========================

Core algorithms for computing SHAP values using different sampling strategies.
Extracted from common marginal contribution patterns across explainers.
"""

import numpy as np
from typing import List, Tuple, Union, Optional, Any
from .coalition_sampling import BaseCoalitionSampler, RandomCoalitionSampler
from .masking import BaseMasker, ZeroMasker
from .model_evaluation import ModelEvaluator


class SHAPEstimator:
    """Core SHAP value estimation using coalition sampling."""
    
    def __init__(self, model: Any, coalition_sampler: BaseCoalitionSampler,
                 masker: BaseMasker, model_evaluator: Optional[ModelEvaluator] = None,
                 device: Optional[Union[str, Any]] = None):
        """
        Args:
            model: Model to explain
            coalition_sampler: Strategy for sampling coalitions
            masker: Strategy for masking features
            model_evaluator: Optional model evaluator (created if not provided)
            device: Device for computation
        """
        self.model = model
        self.coalition_sampler = coalition_sampler
        self.masker = masker
        self.model_evaluator = model_evaluator or ModelEvaluator(model, device)
        
    def compute_marginal_contribution(self, x: np.ndarray, target_position: Tuple[int, int],
                                    n_samples: int = 100) -> float:
        """Compute marginal contribution of a single feature position.
        
        Args:
            x: Input array, shape (T, F)
            target_position: (time, feature) position to compute contribution for
            n_samples: Number of coalition samples
            
        Returns:
            Estimated marginal contribution (SHAP value)
        """
        T, F = x.shape
        all_positions = [(t, f) for t in range(T) for f in range(F)]
        contributions = []
        
        for _ in range(n_samples):
            # Sample coalition excluding target feature
            coalition = self.coalition_sampler.sample_coalition(
                all_positions, exclude=target_position
            )
            
            # Evaluate model with coalition (without target feature)
            x_coalition = self.masker.mask_features(x, coalition)
            pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
            
            # Evaluate model with coalition + target feature
            coalition_plus_target = coalition + [target_position]
            x_coalition_plus = self.masker.mask_features(x, coalition_plus_target)
            pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
            
            # Marginal contribution
            contribution = pred_coalition_plus - pred_coalition
            contributions.append(contribution)
            
        return np.mean(contributions)
    
    def compute_shap_values(self, x: np.ndarray, n_samples: int = 100,
                           positions: Optional[List[Tuple[int, int]]] = None) -> np.ndarray:
        """Compute SHAP values for all or specified positions.
        
        Args:
            x: Input array, shape (T, F)
            n_samples: Number of coalition samples per feature
            positions: Optional list of positions to compute (default: all)
            
        Returns:
            SHAP values array, same shape as input
        """
        T, F = x.shape
        shap_values = np.zeros_like(x)
        
        if positions is None:
            positions = [(t, f) for t in range(T) for f in range(F)]
            
        for t, f in positions:
            shap_values[t, f] = self.compute_marginal_contribution(
                x, (t, f), n_samples
            )
            
        return shap_values
    
    def compute_batch_shap_values(self, X: np.ndarray, n_samples: int = 100) -> np.ndarray:
        """Compute SHAP values for batch of inputs.
        
        Args:
            X: Input batch, shape (B, T, F)
            n_samples: Number of coalition samples per feature
            
        Returns:
            SHAP values array, same shape as input
        """
        batch_shap_values = np.zeros_like(X)
        
        for i, x in enumerate(X):
            batch_shap_values[i] = self.compute_shap_values(x, n_samples)
            
        return batch_shap_values


class FastSHAPEstimator(SHAPEstimator):
    """Optimized SHAP estimation with caching and batch processing."""
    
    def __init__(self, model: Any, coalition_sampler: BaseCoalitionSampler,
                 masker: BaseMasker, model_evaluator: Optional[ModelEvaluator] = None,
                 device: Optional[Union[str, Any]] = None, cache_size: int = 1000):
        super().__init__(model, coalition_sampler, masker, model_evaluator, device)
        self.cache = {}
        self.cache_size = cache_size
        
    def _cache_key(self, x: np.ndarray, masked_positions: List[Tuple[int, int]]) -> str:
        """Generate cache key for masked input."""
        positions_str = str(sorted(masked_positions))
        x_hash = hash(x.tobytes())
        return f"{x_hash}_{positions_str}"
    
    def _get_cached_prediction(self, x: np.ndarray, 
                              masked_positions: List[Tuple[int, int]]) -> Optional[float]:
        """Get cached prediction if available."""
        key = self._cache_key(x, masked_positions)
        return self.cache.get(key)
    
    def _cache_prediction(self, x: np.ndarray, masked_positions: List[Tuple[int, int]], 
                         prediction: float):
        """Cache prediction result."""
        if len(self.cache) >= self.cache_size:
            # Simple cache eviction - remove oldest entries
            oldest_keys = list(self.cache.keys())[:len(self.cache) // 2]
            for key in oldest_keys:
                del self.cache[key]
                
        key = self._cache_key(x, masked_positions)
        self.cache[key] = prediction
    
    def compute_marginal_contribution(self, x: np.ndarray, target_position: Tuple[int, int],
                                    n_samples: int = 100) -> float:
        """Compute marginal contribution with caching."""
        T, F = x.shape
        all_positions = [(t, f) for t in range(T) for f in range(F)]
        contributions = []
        
        for _ in range(n_samples):
            coalition = self.coalition_sampler.sample_coalition(
                all_positions, exclude=target_position
            )
            
            # Try to get cached predictions
            pred_coalition = self._get_cached_prediction(x, coalition)
            if pred_coalition is None:
                x_coalition = self.masker.mask_features(x, coalition)
                pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                self._cache_prediction(x, coalition, pred_coalition)
            
            coalition_plus_target = coalition + [target_position]
            pred_coalition_plus = self._get_cached_prediction(x, coalition_plus_target)
            if pred_coalition_plus is None:
                x_coalition_plus = self.masker.mask_features(x, coalition_plus_target)
                pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                self._cache_prediction(x, coalition_plus_target, pred_coalition_plus)
            
            contribution = pred_coalition_plus - pred_coalition
            contributions.append(contribution)
            
        return np.mean(contributions)


def compute_marginal_contributions(model: Any, x: np.ndarray, 
                                 target_positions: List[Tuple[int, int]],
                                 coalition_sampler: Optional[BaseCoalitionSampler] = None,
                                 masker: Optional[BaseMasker] = None,
                                 n_samples: int = 100,
                                 device: Optional[Union[str, Any]] = None) -> List[float]:
    """Convenience function for computing marginal contributions."""
    if coalition_sampler is None:
        coalition_sampler = RandomCoalitionSampler()
    if masker is None:
        masker = ZeroMasker()
        
    estimator = SHAPEstimator(model, coalition_sampler, masker, device=device)
    
    contributions = []
    for target_pos in target_positions:
        contrib = estimator.compute_marginal_contribution(x, target_pos, n_samples)
        contributions.append(contrib)
        
    return contributions


def estimate_shap_values(model: Any, x: np.ndarray,
                        coalition_sampler: Optional[BaseCoalitionSampler] = None,
                        masker: Optional[BaseMasker] = None,
                        n_samples: int = 100,
                        device: Optional[Union[str, Any]] = None) -> np.ndarray:
    """Convenience function for SHAP value estimation."""
    if coalition_sampler is None:
        coalition_sampler = RandomCoalitionSampler()
    if masker is None:
        masker = ZeroMasker()
        
    estimator = SHAPEstimator(model, coalition_sampler, masker, device=device)
    return estimator.compute_shap_values(x, n_samples)


__all__ = [
    "SHAPEstimator",
    "FastSHAPEstimator", 
    "compute_marginal_contributions",
    "estimate_shap_values"
]
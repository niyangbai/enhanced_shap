"""
TimeSHAP Explainer: Pruning-Enhanced SHAP for Sequential Models
================================================================

**TimeSHAP** is a SHAP-style feature attribution method specifically designed for **sequential, temporal,  
or event-based models**. It builds upon the principles of KernelSHAP but introduces **pruning** to handle  
the combinatorial complexity of time-series input structures.

Rather than sampling all possible feature–timestep or event subsets, TimeSHAP first performs a **rough importance  
scan**, then prunes the space down to the **top-k most relevant events**, windows, or timesteps.
"""

import numpy as np
import torch
import random
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import RandomCoalitionSampler
from shap_enhanced.algorithms.masking import ZeroMasker, MeanMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs, BackgroundProcessor

class PrunedMasker:
    """Custom masker for TimeSHAP with pruning support."""
    
    def __init__(self, mask_strategy="mean", background_mean=None):
        self.mask_strategy = mask_strategy
        self.background_mean = background_mean
    
    def mask_positions(self, x, positions):
        """Apply masking to specific (t, f) positions."""
        x_masked = x.copy()
        
        for t, f in positions:
            if self.mask_strategy == "zero":
                x_masked[t, f] = 0.0
            elif self.mask_strategy == "mean" and self.background_mean is not None:
                x_masked[t, f] = self.background_mean[t, f]
        
        return x_masked

class TimeSHAPExplainer(BaseExplainer):
    r"""
    TimeSHAPExplainer: Pruned SHAP Attribution for Sequential Models

    Implements a SHAP-style explainer for time-series and sequential data using pruning
    to efficiently estimate per-(t, f) or event-level attributions.

    :param model: The model to be explained.
    :param background: Background dataset for imputation and mean estimation.
    :param str mask_strategy: Masking method, either 'zero' or 'mean'.
    :param int or None event_window: Optional window size for event-based attribution.
    :param int or None prune_topk: If specified, retain only top-k units for refinement.
    :param str device: Computation device ('cpu' or 'cuda').
    """
    def __init__(
        self, model, background,
        mask_strategy="mean",
        event_window=None,
        prune_topk=None,
        device=None
    ):
        super().__init__(model, background)
        self.mask_strategy = mask_strategy
        self.event_window = event_window
        self.prune_topk = prune_topk
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Process background data 
        self.background = BackgroundProcessor.process_background(background)
        
        # Initialize common algorithm components
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)
        
        if mask_strategy == "mean":
            bg_stats = BackgroundProcessor.compute_background_statistics(self.background)
            self.masker = PrunedMasker(mask_strategy, bg_stats['mean'])
        else:
            self.masker = PrunedMasker(mask_strategy)

    def _get_mask_positions(self, T, F, idx, window=None):
        if window is None:
            # idx = (t, f)
            return [idx]
        # For event/window-level, mask all (t, f) within the window centered at t
        t0 = max(0, idx - window // 2)
        t1 = min(T, idx + window // 2 + 1)
        return [(t, f) for t in range(t0, t1) for f in range(F)]

    def shap_values(
        self, X,
        nsamples=100,
        level="timestep",  # or 'feature' or 'event'
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        r"""
        Compute SHAP values for sequential input with optional pruning and window-based attribution.

        :param X: Input tensor or array of shape (T, F) or (B, T, F).
        :param int nsamples: Number of coalitions to sample per unit.
        :param str level: Attribution level: 'timestep', 'feature', or 'event'.
        :param bool check_additivity: If True, print additivity diagnostics.
        :param int random_seed: Random seed for reproducibility.
        :return: SHAP values with shape (T, F) or (B, T, F).
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape

        # Build units for masking
        if self.event_window is not None and level == "event":
            # Each unit is a window of timesteps
            units = list(range(T - self.event_window + 1))
            mask_unit = lambda u: self._get_mask_positions(T, F, u, window=self.event_window)
        elif level == "timestep":
            # Each unit is a (t, f) pair
            units = [(t, f) for t in range(T) for f in range(F)]
            mask_unit = lambda u: [u]
        elif level == "feature":
            # Each unit is a feature index
            units = list(range(F))
            mask_unit = lambda u: [(t, u) for t in range(T)]
        else:
            raise ValueError(f"Unknown level {level}")

        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_processed[b]

            # === First pass: rough importance for all units ===
            approx_contribs = []
            for idx in units:
                contribs = []
                for _ in range(nsamples):
                    unit_candidates = [u for u in units if u != idx]
                    if len(unit_candidates) == 0:
                        C = []
                    else:
                        k = np.random.randint(1, len(unit_candidates)+1)
                        C = random.sample(unit_candidates, k) if len(unit_candidates) >= k else unit_candidates
                    # Build mask positions from coalition
                    mask_positions = []
                    for u in C:
                        mask_positions.extend(mask_unit(u))
                    x_S = self.masker.mask_positions(x_orig, mask_positions)
                    # Mask union: coalition + current idx
                    mask_positions_union = mask_positions + mask_unit(idx)
                    x_S_union = self.masker.mask_positions(x_orig, mask_positions_union)
                    
                    out_S = self.model_evaluator.evaluate_single(x_S)
                    out_S_union = self.model_evaluator.evaluate_single(x_S_union)
                    contribs.append(out_S - out_S_union)
                approx_contribs.append(np.mean(contribs))

            # === Prune to top-k units if enabled ===
            active_units = units
            if self.prune_topk is not None and self.prune_topk < len(units):
                topk_units_idx = np.argsort(np.abs(approx_contribs))[-self.prune_topk:]
                active_units = [units[i] for i in topk_units_idx]

            # === Second pass: refined estimation on pruned set ===
            for idx in active_units:
                contribs = []
                unit_candidates = [u for u in active_units if u != idx]
                for _ in range(nsamples):
                    if len(unit_candidates) == 0:
                        C = []
                    else:
                        k = np.random.randint(1, len(unit_candidates)+1)
                        C = random.sample(unit_candidates, k) if len(unit_candidates) >= k else unit_candidates
                    mask_positions = []
                    for u in C:
                        mask_positions.extend(mask_unit(u))
                    x_S = self.masker.mask_positions(x_orig, mask_positions)
                    mask_positions_union = mask_positions + mask_unit(idx)
                    x_S_union = self.masker.mask_positions(x_orig, mask_positions_union)
                    
                    out_S = self.model_evaluator.evaluate_single(x_S)
                    out_S_union = self.model_evaluator.evaluate_single(x_S_union)
                    contribs.append(out_S - out_S_union)

                # Assign to output
                if self.event_window is not None and level == "event":
                    for t, f in mask_unit(idx):
                        shap_vals[b, t, f] += np.mean(contribs) / self.event_window  # distribute over window
                elif level == "timestep":
                    t, f = idx
                    shap_vals[b, t, f] = np.mean(contribs)
                elif level == "feature":
                    for t in range(T):
                        shap_vals[b, t, idx] = np.mean(contribs)

            # Apply additivity normalization using common algorithm
            all_positions = [(t, f) for t in range(T) for f in range(F)]
            fully_masked = self.masker.mask_positions(x_orig, all_positions)
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )

        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[TimeSHAP Additivity] sum(SHAP)={result.sum():.4f}")
            
        return result
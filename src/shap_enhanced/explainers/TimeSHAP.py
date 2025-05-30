"""
TimeSHAP Explainer (pruning-enhanced SHAP for sequential models).

## Theoretical Explanation

TimeSHAP is a SHAP-style feature attribution method for sequential or event-based models that uses pruning to efficiently estimate attributions over time-event windows. It implements KernelSHAP-style estimation, but avoids the combinatorial explosion of possible feature/time subsets by pruning to the most important events or windows. TimeSHAP supports attribution at the timestep, feature, or event (window) level.

### Key Concepts

- **Pruned Coalition Sampling:** Instead of exhaustively sampling all possible coalitions, TimeSHAP first estimates rough importances for all units (timesteps, features, or events), then prunes to the top-k most important units for refined estimation.
- **Event/Window Support:** Attributions can be computed for individual timesteps, features, or over sliding windows of events.
- **Flexible Masking:** Masked features can be imputed with zeros or mean values from the background data.
- **Additivity Normalization:** Attributions are normalized so their sum matches the model output difference between the original and fully-masked input.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data, masking strategy, event window size, pruning parameter, and device.
2. **Rough Importance Estimation:**
    - For each unit (timestep, feature, or event window), estimate its marginal contribution by sampling random coalitions and measuring the change in model output.
3. **Pruning:**
    - If pruning is enabled, select the top-k most important units for further analysis.
4. **Refined Attribution:**
    - For each selected unit, sample random coalitions and compute the marginal contribution more accurately.
    - Assign attributions to the appropriate timesteps, features, or events.
5. **Normalization:**
    - Scale attributions so their sum matches the difference between the original and fully-masked model output.
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class TimeSHAPExplainer(BaseExplainer):
    """
    TimeSHAP: SHAP with pruning for sequential/event models.

    Parameters
    ----------
    model : Any
        The model to be explained.
    background : np.ndarray or torch.Tensor
        Background data for imputation (N, T, F).
    mask_strategy : str
        'zero' or 'mean' (default).
    event_window : int or None
        If set, computes SHAP over windows of this size instead of single time steps.
    prune_topk : int or None
        If set, after initial run, prune to top-k most important events/windows, then resample only these.
    device : str
        'cpu' or 'cuda'.
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
        if mask_strategy == "mean":
            self._mean = background.mean(axis=0)
        else:
            self._mean = None

    def _impute(self, X, mask_idxs):
        X_imp = X.copy()
        for (t, f) in mask_idxs:
            if self.mask_strategy == "zero":
                X_imp[t, f] = 0.0
            elif self.mask_strategy == "mean":
                X_imp[t, f] = self._mean[t, f]
            else:
                raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")
        return X_imp

    def _get_mask_idxs(self, T, F, idx, window=None):
        if window is None:
            return [(idx, f) for f in range(F)]
        # Return all (t,f) in window centered at idx
        half = window // 2
        t0 = max(0, idx - half)
        t1 = min(T, idx + half + 1)
        return [(t, f) for t in range(t0, t1) for f in range(F)]

    def shap_values(
        self, X,
        nsamples=100,
        level="timestep",  # or 'feature' or 'event'
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        Computes SHAP attributions at timestep, feature, or event level using pruned coalition sampling.
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        if len(X_in.shape) == 2:
            X_in = X_in[None, ...]
            single = True
        else:
            single = False
        B, T, F = X_in.shape

        # For event/window-level, build set of windows
        if self.event_window is not None and level == "event":
            num_events = T - self.event_window + 1
        else:
            num_events = T

        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_in[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]

            # Determine units: single step, window/event, or feature
            if level == "event" and self.event_window is not None:
                units = list(range(num_events))  # event/window indices
            elif level == "timestep":
                units = list(range(T))
            elif level == "feature":
                units = list(range(F))
            else:
                raise ValueError(f"Unknown level {level}")

            # First pass: get rough importance for all units
            approx_contribs = []
            for idx in units:
                contribs = []
                for _ in range(nsamples):
                    # Sample a coalition S ⊆ units \ {idx}
                    k = np.random.randint(1, len(units))
                    C = np.random.choice([u for u in units if u != idx], size=k, replace=False)
                    mask_idxs = []
                    if level == "event" and self.event_window is not None:
                        for u in C:
                            mask_idxs.extend(self._get_mask_idxs(T, F, u, window=self.event_window))
                    elif level == "timestep":
                        for u in C:
                            mask_idxs.extend(self._get_mask_idxs(T, F, u, window=None))
                    elif level == "feature":
                        mask_idxs.extend([(t, idx) for t in range(T)])
                        mask_idxs.extend([(t, f) for t in range(T) for f in range(F) if f != idx])
                    # x_S: mask coalition only
                    x_S = self._impute(x_orig, mask_idxs)
                    # x_S_union: mask coalition plus current unit
                    if level == "event" and self.event_window is not None:
                        mask_idxs_union = mask_idxs + self._get_mask_idxs(T, F, idx, window=self.event_window)
                    elif level == "timestep":
                        mask_idxs_union = mask_idxs + self._get_mask_idxs(T, F, idx, window=None)
                    elif level == "feature":
                        mask_idxs_union = mask_idxs + [(t, idx) for t in range(T)]
                    x_S_union = self._impute(x_orig, mask_idxs_union)
                    out_S = self.model(torch.tensor(x_S[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                    out_S_union = self.model(torch.tensor(x_S_union[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                    contribs.append(out_S - out_S_union)
                approx_contribs.append(np.mean(contribs))

            # Prune to top-k units if needed
            if self.prune_topk is not None and self.prune_topk < len(units):
                topk_units = np.argsort(np.abs(approx_contribs))[-self.prune_topk:]
                units = [units[i] for i in topk_units]

            # Second pass: refined estimation on pruned set
            for idx in units:
                contribs = []
                for _ in range(nsamples):
                    # Sample coalition (exclude idx)
                    k = np.random.randint(1, len(units))
                    C = np.random.choice([u for u in units if u != idx], size=k, replace=False)
                    mask_idxs = []
                    if level == "event" and self.event_window is not None:
                        for u in C:
                            mask_idxs.extend(self._get_mask_idxs(T, F, u, window=self.event_window))
                    elif level == "timestep":
                        for u in C:
                            mask_idxs.extend(self._get_mask_idxs(T, F, u, window=None))
                    elif level == "feature":
                        mask_idxs.extend([(t, idx) for t in range(T)])
                        mask_idxs.extend([(t, f) for t in range(T) for f in range(F) if f != idx])
                    x_S = self._impute(x_orig, mask_idxs)
                    if level == "event" and self.event_window is not None:
                        mask_idxs_union = mask_idxs + self._get_mask_idxs(T, F, idx, window=self.event_window)
                    elif level == "timestep":
                        mask_idxs_union = mask_idxs + self._get_mask_idxs(T, F, idx, window=None)
                    elif level == "feature":
                        mask_idxs_union = mask_idxs + [(t, idx) for t in range(T)]
                    x_S_union = self._impute(x_orig, mask_idxs_union)
                    out_S = self.model(torch.tensor(x_S[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                    out_S_union = self.model(torch.tensor(x_S_union[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                    contribs.append(out_S - out_S_union)

                # Assign to correct output slots
                if level == "event" and self.event_window is not None:
                    for t, f in self._get_mask_idxs(T, F, idx, window=self.event_window):
                        shap_vals[b, t, f] += np.mean(contribs) / self.event_window  # distribute over window
                elif level == "timestep":
                    for f in range(F):
                        shap_vals[b, idx, f] = np.mean(contribs)
                elif level == "feature":
                    for t in range(T):
                        shap_vals[b, t, idx] = np.mean(contribs)

            # Optionally: normalize for additivity
            orig_pred = self.model(torch.tensor(x_orig[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            x_all_masked = self._impute(x_orig, all_pos)
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[TimeSHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")
        return shap_vals

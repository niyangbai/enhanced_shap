"""
TimeSHAP Explainer: Pruning-Enhanced SHAP for Sequential Models
================================================================

Theoretical Explanation
-----------------------

**TimeSHAP** is a SHAP-style feature attribution method specifically designed for **sequential, temporal,
or event-based models**. It builds upon the principles of KernelSHAP but introduces **pruning** to handle
the combinatorial complexity of time-series input structures.

Rather than sampling all possible feature–timestep or event subsets, TimeSHAP first performs a **rough importance
scan**, then prunes the space down to the **top-k most relevant events**, windows, or timesteps. This makes it
scalable to long sequences while maintaining fidelity.

Key Concepts
^^^^^^^^^^^^

- **Pruned Coalition Sampling**:
    Performs an initial round of random sampling to estimate rough feature/event importance.
    Only the top-k units are retained for precise SHAP estimation.

- **Event/Window Attribution**:
    Supports attribution across:
        - Individual timesteps (fine-grained),
        - Features (vertical slices),
        - Event windows (e.g., rolling sequences).

- **Flexible Masking**:
    Masked features can be:
        - Set to zero (hard masking), or
        - Replaced with the mean from background data (soft masking).

- **Additivity Normalization**:
    Final SHAP attributions are normalized so that their total equals the model output difference
    between the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a target model, background data, event window size, masking strategy, pruning parameter (e.g., top-k),
        and device context.

2. **Rough Importance Estimation**:
    - For each unit (feature, timestep, or window):
        - Sample random coalitions excluding the unit.
        - Compute model outputs with and without the unit masked.
        - Estimate marginal contribution based on output difference.

3. **Pruning**:
    - If pruning is enabled:
        - Retain only the top-k most important units from the rough scan.
        - Discard lower-importance units from further evaluation.

4. **Refined Attribution**:
    - For each selected unit:
        - Sample coalitions and compute more precise SHAP values.
        - Assign contributions to the appropriate location in the attribution map
            (e.g., timestep, feature, or window).

5. **Normalization**:
    - Rescale all SHAP values so that their sum equals the difference between
        the model prediction on the original and fully-masked input.

Use Case
--------

TimeSHAP is ideal for:
    - Event sequences, medical time-series, or log data.
    - Models where full SHAP computation is infeasible due to input length.
    - Explaining model behavior over time, including “when” and “what” drove a prediction.
"""

import random

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class TimeSHAPExplainer(BaseExplainer):
    r"""
    TimeSHAPExplainer: Pruned SHAP Attribution for Sequential Models

    Implements a SHAP-style explainer for time-series and sequential data using pruning
    to efficiently estimate per-(t, f) or event-level attributions.

    Combines:
        - Masking strategy (zero or mean-based).
        - Optional event windowing (for segment-level attribution).
        - Top-k pruning to reduce the coalition space before final SHAP estimation.

    :param model: The model to be explained.
    :type model: Any
    :param background: Background dataset for imputation and mean estimation.
    :type background: np.ndarray or torch.Tensor
    :param str mask_strategy: Masking method, either 'zero' or 'mean'.
    :param int or None event_window: Optional window size for event-based attribution.
    :param int or None prune_topk: If specified, retain only top-k units (based on rough attribution) for refinement.
    :param str device: Computation device ('cpu' or 'cuda').
    """

    def __init__(
        self,
        model,
        background,
        mask_strategy="mean",
        event_window=None,
        prune_topk=None,
        device=None,
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
        for t, f in mask_idxs:
            if self.mask_strategy == "zero":
                X_imp[t, f] = 0.0
            elif self.mask_strategy == "mean":
                X_imp[t, f] = self._mean[t, f]
            else:
                raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")
        return X_imp

    def _get_mask_idxs(self, T, F, idx, window=None):
        if window is None:
            # idx = (t, f)
            return [idx]
        # For event/window-level, mask all (t, f) within the window centered at t
        t0 = max(0, idx - window // 2)
        t1 = min(T, idx + window // 2 + 1)
        return [(t, f) for t in range(t0, t1) for f in range(F)]

    def shap_values(
        self,
        X,
        nsamples=100,
        level="timestep",  # or 'feature' or 'event'
        check_additivity=True,
        random_seed=42,
        **kwargs,
    ):
        r"""
        Compute SHAP values for sequential input with optional pruning and window-based attribution.

        .. note::
            Pruned estimation uses an initial coarse pass to identify important units
            (features, timesteps, or windows), followed by refined SHAP estimation over that subset.

        :param X: Input tensor or array of shape (T, F) or (B, T, F).
        :type X: Union[np.ndarray, torch.Tensor]
        :param int nsamples: Number of coalitions to sample per unit.
        :param str level: Attribution level: 'timestep', 'feature', or 'event'.
        :param bool check_additivity: If True, print additivity diagnostics.
        :param int random_seed: Random seed for reproducibility.
        :return: SHAP values with shape (T, F) or (B, T, F).
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        random.seed(random_seed)
        is_torch = hasattr(X, "detach")
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        if len(X_in.shape) == 2:
            X_in = X_in[None, ...]
            single = True
        else:
            single = False
        B, T, F = X_in.shape

        # Build units for masking
        if self.event_window is not None and level == "event":
            # Each unit is a window of timesteps
            units = list(range(T - self.event_window + 1))

            def mask_unit(u):
                return self._get_mask_idxs(T, F, u, window=self.event_window)

        elif level == "timestep":
            # Each unit is a (t, f) pair
            units = [(t, f) for t in range(T) for f in range(F)]

            def mask_unit(u):
                return [u]

        elif level == "feature":
            # Each unit is a feature index
            units = list(range(F))

            def mask_unit(u):
                return [(t, u) for t in range(T)]

        else:
            raise ValueError(f"Unknown level {level}")

        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_in[b]

            # === First pass: rough importance for all units ===
            approx_contribs = []
            for idx in units:
                contribs = []
                for _ in range(nsamples):
                    unit_candidates = [u for u in units if u != idx]
                    if len(unit_candidates) == 0:
                        C = []
                    else:
                        k = np.random.randint(1, len(unit_candidates) + 1)
                        C = (
                            random.sample(unit_candidates, k)
                            if len(unit_candidates) >= k
                            else unit_candidates
                        )
                    # Build mask indices from coalition
                    mask_idxs = []
                    for u in C:
                        mask_idxs.extend(mask_unit(u))
                    x_S = self._impute(x_orig, mask_idxs)
                    # Mask union: coalition + current idx
                    mask_idxs_union = mask_idxs + mask_unit(idx)
                    x_S_union = self._impute(x_orig, mask_idxs_union)
                    out_S = (
                        self.model(
                            torch.tensor(
                                x_S[None], dtype=torch.float32, device=self.device
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    out_S_union = (
                        self.model(
                            torch.tensor(
                                x_S_union[None], dtype=torch.float32, device=self.device
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    contribs.append(out_S - out_S_union)
                approx_contribs.append(np.mean(contribs))

            # === Prune to top-k units if enabled ===
            active_units = units
            if self.prune_topk is not None and self.prune_topk < len(units):
                topk_units_idx = np.argsort(np.abs(approx_contribs))[-self.prune_topk :]
                active_units = [units[i] for i in topk_units_idx]

            # === Second pass: refined estimation on pruned set ===
            for idx in active_units:
                contribs = []
                unit_candidates = [u for u in active_units if u != idx]
                for _ in range(nsamples):
                    if len(unit_candidates) == 0:
                        C = []
                    else:
                        k = np.random.randint(1, len(unit_candidates) + 1)
                        C = (
                            random.sample(unit_candidates, k)
                            if len(unit_candidates) >= k
                            else unit_candidates
                        )
                    mask_idxs = []
                    for u in C:
                        mask_idxs.extend(mask_unit(u))
                    x_S = self._impute(x_orig, mask_idxs)
                    mask_idxs_union = mask_idxs + mask_unit(idx)
                    x_S_union = self._impute(x_orig, mask_idxs_union)
                    out_S = (
                        self.model(
                            torch.tensor(
                                x_S[None], dtype=torch.float32, device=self.device
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    out_S_union = (
                        self.model(
                            torch.tensor(
                                x_S_union[None], dtype=torch.float32, device=self.device
                            )
                        )
                        .detach()
                        .cpu()
                        .numpy()
                        .squeeze()
                    )
                    contribs.append(out_S - out_S_union)

                # Assign to output
                if self.event_window is not None and level == "event":
                    for t, f in mask_unit(idx):
                        shap_vals[b, t, f] += (
                            np.mean(contribs) / self.event_window
                        )  # distribute over window
                elif level == "timestep":
                    t, f = idx
                    shap_vals[b, t, f] = np.mean(contribs)
                elif level == "feature":
                    for t in range(T):
                        shap_vals[b, t, idx] = np.mean(contribs)

            # === Normalization for additivity ===
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            orig_pred = (
                self.model(
                    torch.tensor(x_orig[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            x_all_masked = self._impute(x_orig, all_pos)
            masked_pred = (
                self.model(
                    torch.tensor(
                        x_all_masked[None], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(
                f"[TimeSHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}"
            )
        return shap_vals

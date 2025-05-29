"""
Adaptive Baseline SHAP (Sparse)

Estimates feature attributions using SHAP, but with baselines always drawn from real observed samples (never mean or zero). Baseline selection is either by support similarity or random from the data. All perturbations are guaranteed to be valid binary/one-hot patterns.
"""

import numpy as np
import torch
from ..base_explainer import BaseExplainer

class AdaptiveBaselineSHAPExplainer(BaseExplainer):
    """
    SHAP with Adaptive (Sparse) Baseline.

    Parameters
    ----------
    model : Any
        The model to be explained.
    background : np.ndarray or torch.Tensor
        Candidate baseline samples (N, T, F), must be valid sparse (binary or one-hot).
    baseline_strategy : str
        'random' (default): baselines are chosen randomly from background set.
        'support': baselines are chosen to match the support (nonzero positions) of the sample.
    n_baselines : int
        How many baselines to sample per coalition (trade-off: speed vs. variance).
    mask_strategy : str
        'baseline' (default): Masked features are set to corresponding value from selected baseline.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        background,
        baseline_strategy="random",
        n_baselines=10,
        mask_strategy="baseline",
        device=None
    ):
        super().__init__(model, background)
        self.baseline_strategy = baseline_strategy
        self.n_baselines = n_baselines
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.background_np = background.detach().cpu().numpy() if hasattr(background, 'detach') else np.asarray(background)
        assert self.background_np.dtype in (np.float32, np.float64, np.int32, np.int64, np.uint8)

    def _select_baselines(self, x, n):
        # x: (T, F), return (n, T, F)
        if self.baseline_strategy == "random":
            idx = np.random.choice(len(self.background_np), n, replace=True)
            return self.background_np[idx]
        elif self.baseline_strategy == "support":
            support = (x != 0)
            mask = (self.background_np != 0)
            # Find all background samples with exactly same support (can relax to subset if strict is too rare)
            match = np.all(mask == support, axis=(1,2))
            candidates = self.background_np[match]
            if len(candidates) == 0:
                # fallback: random
                return self._select_baselines(x, n)
            idx = np.random.choice(len(candidates), n, replace=True)
            return candidates[idx]
        else:
            raise ValueError(f"Unknown baseline_strategy: {self.baseline_strategy}")

    def _impute(self, x, baseline, mask_idxs):
        # For a single coalition: set mask_idxs to baseline's value, others unchanged
        x_imp = x.copy()
        for (t, f) in mask_idxs:
            x_imp[t, f] = baseline[t, f]
        return x_imp

    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        Adaptive Baseline SHAP for (B, T, F) or (T, F).
        All perturbed inputs are valid sparse patterns.
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = len(X_in.shape) == 2
        if single:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        for b in range(B):
            x = X_in[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        # Random coalition (avoid empty/full)
                        k = np.random.randint(1, len(available)+1)
                        mask_idxs = [available[i] for i in np.random.choice(len(available), k, replace=False)]
                        # Sample n_baselines for each coalition
                        baselines = self._select_baselines(x, self.n_baselines)
                        for baseline in baselines:
                            x_masked = self._impute(x, baseline, mask_idxs)
                            x_masked_tf = self._impute(x_masked, baseline, [(t, f)])
                            out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                            out_masked_tf = self.model(torch.tensor(x_masked_tf[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                            mc.append(out_masked_tf - out_masked)
                    shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            # Mask all with a baseline (randomly chosen)
            x_all_masked = self._impute(x, self._select_baselines(x, 1)[0], all_pos)
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[AdaptiveBaselineSHAP] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")
        return shap_vals

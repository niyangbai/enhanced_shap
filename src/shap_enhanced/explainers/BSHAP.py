"""
BShapExplainer (Distribution-Free SHAP, LSTM/sequence edition)

Estimates feature attributions by masking with uninformative, *random* values (uniform or noise),
never using the empirical data distribution. Suited for sequence models such as LSTM.

Reference:
    - https://www.tandfonline.com/doi/full/10.1080/02331888.2025.2487853
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class BShapExplainer(BaseExplainer):
    """
    BShap: Distribution-Free SHAP Explainer (LSTM/sequence)

    Parameters
    ----------
    model : Any
        Sequence model to be explained.
    input_range : tuple or None
        (min, max) or per-feature min/max, to sample uninformative values for each feature.
        If None, uses (-1, 1) for continuous, (0, 1) for binary.
    n_samples : int
        Number of random coalitions per feature to average.
    mask_strategy : str
        'random' (default): sample uniformly at random for each mask.
        'noise': add Gaussian noise to masked features (for continuous).
        'zero': set masked features to zero (not fully background-free).
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        input_range=None,
        n_samples=100,
        mask_strategy="random",
        device=None
    ):
        super().__init__(model, background=None)
        self.input_range = input_range
        self.n_samples = n_samples
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _mask(self, x, mask_idxs):
        x_masked = x.copy()
        T, F = x.shape
        for (t, f) in mask_idxs:
            if self.mask_strategy == "random":
                # Sample independently for each mask (distribution-free)
                if self.input_range is not None:
                    mn, mx = self.input_range
                    # Per-feature or global
                    if isinstance(mn, np.ndarray):
                        x_masked[t, f] = np.random.uniform(mn[f], mx[f])
                    else:
                        x_masked[t, f] = np.random.uniform(mn, mx)
                else:
                    # Default: (-1, 1) for continuous, (0, 1) for binary
                    x_masked[t, f] = np.random.uniform(0, 1)
            elif self.mask_strategy == "noise":
                x_masked[t, f] = x[t, f] + np.random.normal(0, 0.5)
            elif self.mask_strategy == "zero":
                x_masked[t, f] = 0.0
            else:
                raise ValueError("Unknown mask_strategy")
        return x_masked

    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        BShap for (B, T, F) or (T, F).
        Returns: (T, F) or (B, T, F)
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = len(X_in.shape) == 2
        if single:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        all_pos = [(t, f) for t in range(T) for f in range(F)]
        for b in range(B):
            x = X_in[b]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        # Sample random coalition of available features
                        k = np.random.randint(1, len(available)+1)
                        mask_idxs = [available[i] for i in np.random.choice(len(available), k, replace=False)]
                        x_masked = self._mask(x, mask_idxs)
                        x_masked_tf = self._mask(x_masked, [(t, f)])
                        out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        out_masked_tf = self.model(torch.tensor(x_masked_tf[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        mc.append(out_masked_tf - out_masked)
                    shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            x_all_masked = self._mask(x, [(t, f) for t in range(T) for f in range(F)])
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[BShap] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

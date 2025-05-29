"""
Support-Preserving SHAP Explainer

For every coalition to mask, the perturbed instance is replaced by a real sample from the dataset that matches the resulting support pattern. If no such sample exists, the coalition is skipped or flagged.

Parameters
----------
model : Any
    Model to be explained.
background : np.ndarray or torch.Tensor
    The full dataset of valid samples (N, T, F) for replacement.
skip_unmatched : bool
    If True, skip coalitions where no matching replacement is found (default: True).
device : str
    'cpu' or 'cuda'.
"""
import numpy as np
import torch
from ..base_explainer import BaseExplainer

class SupportPreservingSHAPExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        background,
        skip_unmatched=True,
        device=None
    ):
        super().__init__(model, background)
        self.background = background.detach().cpu().numpy() if hasattr(background, 'detach') else np.asarray(background)
        self.skip_unmatched = skip_unmatched
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Precompute support matrix for fast lookup
        self.bg_support = (self.background != 0)

    def _find_matching_sample(self, support_mask):
        # support_mask: (T, F) boolean
        # Returns index in background or None
        matches = np.all(self.bg_support == support_mask, axis=(1,2))
        idxs = np.where(matches)[0]
        if len(idxs) > 0:
            return np.random.choice(idxs)
        else:
            return None

    def shap_values(
        self,
        X,
        nsamples=100,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        SHAP with support-preserving perturbations.
        Only uses real data points as replacements for each masked coalition.
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
                        # Random coalition to mask
                        k = np.random.randint(1, len(available)+1)
                        mask_idxs = [available[i] for i in np.random.choice(len(available), k, replace=False)]
                        # Mask features by zeroing
                        x_masked = x.copy()
                        for (tt, ff) in mask_idxs:
                            x_masked[tt, ff] = 0
                        support_mask = (x_masked != 0)
                        idx = self._find_matching_sample(support_mask)
                        if idx is None:
                            if self.skip_unmatched:
                                continue  # skip this sample
                            else:
                                raise ValueError("No matching sample found for support pattern!")
                        x_replacement = self.background[idx]
                        # Now mask (t, f) as well
                        x_masked_tf = x_masked.copy()
                        x_masked_tf[t, f] = 0
                        support_mask_tf = (x_masked_tf != 0)
                        idx_tf = self._find_matching_sample(support_mask_tf)
                        if idx_tf is None:
                            if self.skip_unmatched:
                                continue
                            else:
                                raise ValueError("No matching sample found for tf-masked support pattern!")
                        x_replacement_tf = self.background[idx_tf]

                        # Model evaluations
                        out_masked = self.model(torch.tensor(x_replacement[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        out_masked_tf = self.model(torch.tensor(x_replacement_tf[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        mc.append(out_masked_tf - out_masked)
                    if len(mc) > 0:
                        shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            # Mask all, find replacement for all-zero support (optional, or skip normalization)
            all_masked = np.zeros_like(x)
            idx_all = self._find_matching_sample((all_masked != 0))
            if idx_all is not None:
                masked_pred = self.model(torch.tensor(self.background[idx_all][None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                shap_sum = shap_vals[b].sum()
                model_diff = orig_pred - masked_pred
                if shap_sum != 0:
                    shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[SupportPreservingSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

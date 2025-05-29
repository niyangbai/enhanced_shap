"""
Empirical Conditional SHAP for Discrete Data

For every coalition, imputes masked features by sampling from the empirical conditional distribution given the observed values of the unmasked features.
All imputations are valid, observed binary/one-hot patterns.

Parameters
----------
model : Any
    Model to be explained.
background : np.ndarray or torch.Tensor
    Full dataset (N, T, F) to estimate empirical conditionals.
skip_unmatched : bool
    If True, skip coalitions for which no match is found (default: True).
use_closest : bool
    If True and skip_unmatched is False, uses the most similar pattern (by Hamming distance).
device : str
    'cpu' or 'cuda'.
"""
import numpy as np
import torch
from ..base_explainer import BaseExplainer

class EmpiricalConditionalSHAPExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        background,
        skip_unmatched=True,
        use_closest=False,
        device=None
    ):
        super().__init__(model, background)
        self.background = background.detach().cpu().numpy() if hasattr(background, 'detach') else np.asarray(background)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.skip_unmatched = skip_unmatched
        self.use_closest = use_closest

    def _find_conditional_match(self, mask, x):
        """
        mask: boolean array (T, F). True means masked.
        x: (T, F) original sample.
        Returns index in background of matching instance, or None.
        """
        # Find all rows where unmasked values match exactly
        unmasked = ~mask
        match = np.all(self.background[:, unmasked] == x[unmasked], axis=1)
        idxs = np.where(match)[0]
        if len(idxs) > 0:
            return np.random.choice(idxs)
        elif self.use_closest and len(self.background) > 0:
            # Hamming distance in unmasked positions
            diffs = np.sum(self.background[:, unmasked] != x[unmasked], axis=1)
            idx = np.argmin(diffs)
            return idx
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
        SHAP with empirical conditional imputation, discrete data.
        All masked imputations are observed patterns.
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
            flat_x = x.flatten()
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        # Sample coalition to mask (excluding (t, f))
                        k = np.random.randint(1, len(available)+1)
                        mask_idxs = [available[i] for i in np.random.choice(len(available), k, replace=False)]
                        # Mask: build mask matrix (T, F) of True for masked positions
                        mask = np.zeros((T, F), dtype=bool)
                        for tt, ff in mask_idxs:
                            mask[tt, ff] = True
                        # Sample value for masked coalition from empirical conditional
                        idx_match = self._find_conditional_match(mask, x)
                        if idx_match is None:
                            if self.skip_unmatched:
                                continue
                            else:
                                # If not found and use_closest is off, skip
                                continue
                        x_masked = self.background[idx_match].copy()
                        # Now also mask (t, f)
                        mask2 = mask.copy()
                        mask2[t, f] = True
                        idx_match2 = self._find_conditional_match(mask2, x)
                        if idx_match2 is None:
                            if self.skip_unmatched:
                                continue
                            else:
                                continue
                        x_masked_tf = self.background[idx_match2].copy()
                        # Evaluate
                        out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        out_masked_tf = self.model(torch.tensor(x_masked_tf[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        mc.append(out_masked_tf - out_masked)
                    if len(mc) > 0:
                        shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            # Mask all (entire mask is True)
            mask_all = np.ones((T, F), dtype=bool)
            idx_all = self._find_conditional_match(mask_all, x)
            if idx_all is not None:
                masked_pred = self.model(torch.tensor(self.background[idx_all][None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                shap_sum = shap_vals[b].sum()
                model_diff = orig_pred - masked_pred
                if shap_sum != 0:
                    shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[EmpCondSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

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
from shap_enhanced.base_explainer import BaseExplainer

class SupportPreservingSHAPExplainer(BaseExplainer):
    """
    Support-Preserving SHAP Explainer.

    Uses real samples from the background matching the support pattern of the masked instance.
    If the data is not truly one-hot or binary sparse, falls back to mean-masking (SHAP style).

    Parameters
    ----------
    model : Any
        Model to be explained.
    background : np.ndarray or torch.Tensor
        The full dataset (N, T, F) or (N, F).
    skip_unmatched : bool
        If True, skip coalitions where no matching sample is found.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        background,
        skip_unmatched=True,
        device=None
    ):
        super().__init__(model, background)
        self.background = background.detach().cpu().numpy() if hasattr(background, 'detach') else np.asarray(background)
        if self.background.ndim == 2:
            self.background = self.background[:, None, :]  # (N, 1, F)
        self.skip_unmatched = skip_unmatched
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bg_support = (self.background != 0)
        data_flat = self.background.reshape(-1, self.background.shape[-1])
        is_binary = np.all((data_flat == 0) | (data_flat == 1))
        is_onehot = np.all(np.sum(data_flat, axis=1) == 1)
        self.is_onehot = bool(is_binary and is_onehot)
        if not self.is_onehot:
            print("[SupportPreservingSHAP] WARNING: Data is not one-hot. Will use classic mean-masking SHAP fallback.")
        self.mean_baseline = np.mean(self.background, axis=0)  # (T, F)

    def _find_matching_sample(self, support_mask):
        support_mask = support_mask[None, ...] if support_mask.ndim == 2 else support_mask
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
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        if X_in.ndim == 2:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        for b in range(B):
            x = X_in[b]
            if self.is_onehot:
                # Original support-preserving logic
                all_pos = [(t, f) for t in range(T) for f in range(F)]
                for t in range(T):
                    for f in range(F):
                        mc = []
                        available = [idx for idx in all_pos if idx != (t, f)]
                        for _ in range(nsamples):
                            k = np.random.randint(1, len(available)+1)
                            mask_idxs = [available[i] for i in np.random.choice(len(available), k, replace=False)]
                            x_masked = x.copy()
                            for (tt, ff) in mask_idxs:
                                x_masked[tt, ff] = 0
                            support_mask = (x_masked != 0)
                            idx = self._find_matching_sample(support_mask)
                            if idx is None:
                                if self.skip_unmatched:
                                    continue
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
                            out_masked = self.model(torch.tensor(x_replacement[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                            out_masked_tf = self.model(torch.tensor(x_replacement_tf[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                            mc.append(out_masked_tf - out_masked)
                        if len(mc) > 0:
                            shap_vals[b, t, f] = np.mean(mc)
            else:
                # Classic SHAP fallback (mean masking): for each feature, mask just that feature
                for t in range(T):
                    for f in range(F):
                        x_masked = x.copy()
                        x_masked[t, f] = self.mean_baseline[t, f]
                        out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        out_orig = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        shap_vals[b, t, f] = out_orig - out_masked

            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            if self.is_onehot:
                all_masked = np.zeros_like(x)
                idx_all = self._find_matching_sample((all_masked != 0))
                if idx_all is not None:
                    masked_pred = self.model(torch.tensor(self.background[idx_all][None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                else:
                    masked_pred = self.model(torch.tensor(self.mean_baseline[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            else:
                masked_pred = self.model(torch.tensor(self.mean_baseline[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if X_in.shape[0] == 1 else shap_vals
        if check_additivity:
            print(f"[SupportPreservingSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

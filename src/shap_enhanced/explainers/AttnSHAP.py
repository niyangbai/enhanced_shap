"""
Attention-Guided SHAP (AttnSHAP) Explainer.

This explainer samples feature (time-step, feature) coalitions for perturbation, using either uniform or attention-weighted probabilities based on the model's attention mechanism.
"""

import numpy as np
import torch
from .base_explainer import BaseExplainer

class AttnSHAPExplainer(BaseExplainer):
    """
    Attention-Guided SHAP Explainer (AttnSHAP).

    Parameters
    ----------
    model : Any
        Must provide an `get_attention_weights(x)` method returning attention weights of shape (T,) or (T, F).
    background : np.ndarray or torch.Tensor
        Background data for mean imputation, shape (N, T, F).
    use_attention : bool
        If True, sample coalitions weighted by attention; if False, sample uniformly.
    mask_strategy : str
        'mean' (default) or 'zero'.
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(self, model, background, use_attention=True, mask_strategy="mean", device=None):
        super().__init__(model, background)
        self.use_attention = use_attention
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if mask_strategy == "mean":
            self._mean = background.mean(axis=0)
        else:
            self._mean = None

    def _get_attention_weights(self, x):
        # x: (T, F) or (1, T, F)
        # Assumes model exposes .get_attention_weights(x) method (returns (T,) or (T, F))
        if hasattr(self.model, "get_attention_weights"):
            with torch.no_grad():
                x_in = torch.tensor(x[None], dtype=torch.float32, device=self.device) if x.ndim == 2 else torch.tensor(x, dtype=torch.float32, device=self.device)
                attn = self.model.get_attention_weights(x_in)  # (T,) or (T, F)
                return attn.squeeze().detach().cpu().numpy()
        else:
            raise RuntimeError("Model must implement get_attention_weights(x).")

    def _impute(self, X, idxs):
        X_imp = X.copy()
        for (t, f) in idxs:
            if self.mask_strategy == "zero":
                X_imp[t, f] = 0.0
            elif self.mask_strategy == "mean":
                X_imp[t, f] = self._mean[t, f]
            else:
                raise ValueError(f"Unknown mask_strategy: {self.mask_strategy}")
        return X_imp

    def shap_values(self, X, nsamples=100, coalition_size=None, check_additivity=True, random_seed=42, **kwargs):
        """
        Estimate SHAP values by sampling coalitions, attention-guided or uniform.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            Input sequence(s), shape (T, F) or (B, T, F).
        nsamples : int
            Number of coalitions per (t, f).
        coalition_size : int or None
            Number of positions to mask per coalition. Random if None.
        check_additivity : bool
            If True, rescale for additivity.
        random_seed : int
            For reproducibility.
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        shape = X_in.shape
        if len(shape) == 2:
            X_in = X_in[None, ...]
            single = True
        else:
            single = False
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x_orig = X_in[b]
            attn = self._get_attention_weights(x_orig) if self.use_attention else None
            # Flatten attention weights for (t, f)
            if attn is not None:
                if attn.ndim == 1:
                    attn_flat = np.repeat(attn, F)
                else:
                    attn_flat = attn.flatten()
                attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
            all_pos = [(t, f) for t in range(T) for f in range(F)]

            for t in range(T):
                for f in range(F):
                    contribs = []
                    available_pos = [idx for idx in all_pos if idx != (t, f)]
                    idx_map = {idx: i for i, idx in enumerate(available_pos)}
                    for _ in range(nsamples):
                        # Sample coalition (excluding (t, f))
                        if coalition_size is not None:
                            k = coalition_size
                        else:
                            k = np.random.randint(1, len(available_pos) + 1)
                        if self.use_attention:
                            # Sample indices weighted by attention
                            attn_weights = np.array([attn_flat[idx_map[idx]] for idx in available_pos])
                            attn_weights /= (attn_weights.sum() + 1e-8)
                            C_idxs = np.random.choice(len(available_pos), size=k, replace=False, p=attn_weights)
                        else:
                            C_idxs = np.random.choice(len(available_pos), size=k, replace=False)
                        C_idxs = [available_pos[idx] for idx in C_idxs]

                        x_C = self._impute(x_orig, C_idxs)
                        x_C_tf = self._impute(x_C, [(t, f)])
                        out_C = self.model(torch.tensor(x_C[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        out_C_tf = self.model(torch.tensor(x_C_tf[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        contribs.append(out_C_tf - out_C)
                    shap_vals[b, t, f] = np.mean(contribs)

            # Additivity normalization per sample
            orig_pred = self.model(torch.tensor(x_orig[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            x_all_masked = self._impute(x_orig, all_pos)
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[AttnSHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")
        return shap_vals

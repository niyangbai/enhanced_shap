"""
AttnSHAPExplainer: Attention-Guided SHAP with General Proxy Attention

- Uses model's attention if available (model.get_attention_weights)
- Otherwise, computes a proxy attention: gradient, input, or perturbation-based
- Used for biasing coalition sampling in SHAP for sequential models

Parameters
----------
model : Any
    Model to be explained (should support batch input).
background : np.ndarray or torch.Tensor
    Background data (N, T, F).
use_attention : bool
    If True, use attention or proxy; if False, use uniform sampling.
proxy_attention : str
    If no real attention, use 'gradient' (default), 'input', or 'perturb'.
device : str
    'cpu' or 'cuda'.
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class AttnSHAPExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        background,
        use_attention=True,
        proxy_attention="gradient",  # 'gradient', 'input', 'perturb'
        device=None
    ):
        super().__init__(model, background)
        self.use_attention = use_attention
        self.proxy_attention = proxy_attention
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _get_attention_weights(self, x):
        """
        Returns attention weights for input x (T, F).
        If model has get_attention_weights(x), uses that;
        else, uses proxy specified by self.proxy_attention.
        Output: (T,) or (T, F)
        """
        # Try to use model's attention method if exists
        if hasattr(self.model, "get_attention_weights"):
            with torch.no_grad():
                x_in = torch.tensor(x[None], dtype=torch.float32, device=self.device)
                attn = self.model.get_attention_weights(x_in)
            attn = attn.squeeze().detach().cpu().numpy()
            # If (T, 1), squeeze
            if attn.ndim == 2 and attn.shape[1] == 1:
                attn = attn[:, 0]
            return attn
        # Else, use proxy method
        if self.proxy_attention == "gradient":
            x_tensor = torch.tensor(x[None], dtype=torch.float32, device=self.device, requires_grad=True)
            output = self.model(x_tensor)
            out_scalar = output.view(-1)[0]
            out_scalar.backward()
            attn = x_tensor.grad.abs().detach().cpu().numpy()[0]  # (T, F)
            attn_norm = attn / (attn.sum() + 1e-8)
            # Optionally, sum over features to get (T,)
            attn_time = attn_norm.sum(axis=-1)
            return attn_time
        elif self.proxy_attention == "input":
            attn = np.abs(x).sum(axis=-1)  # (T,)
            attn = attn / (attn.sum() + 1e-8)
            return attn
        elif self.proxy_attention == "perturb":
            base_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            T, F = x.shape
            attn = np.zeros(T)
            for t in range(T):
                x_masked = x.copy()
                x_masked[t, :] = 0
                pred = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                attn[t] = abs(base_pred - pred)
            attn = attn / (attn.sum() + 1e-8)
            return attn
        else:
            raise RuntimeError("No attention or proxy_attention available.")

    def shap_values(
        self,
        X,
        nsamples=100,
        coalition_size=3,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        SHAP with attention/proxy-guided coalition sampling.
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
        for b in range(B):
            x_orig = X_in[b]
            attn = self._get_attention_weights(x_orig) if self.use_attention else None
            if attn is not None:
                attn_flat = attn.flatten() if attn.ndim == 1 else attn.sum(axis=1)
                attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        # Attention-guided sampling if available
                        if attn is not None:
                            attn_prob = np.array([attn[t_] if attn.ndim == 1 else attn[t_, f_] for (t_, f_) in available])
                            attn_prob = attn_prob / (attn_prob.sum() + 1e-8)
                            sel_idxs = np.random.choice(len(available), coalition_size, replace=False, p=attn_prob)
                        else:
                            sel_idxs = np.random.choice(len(available), coalition_size, replace=False)
                        mask_idxs = [available[i] for i in sel_idxs]
                        x_masked = x_orig.copy()
                        for (tt, ff) in mask_idxs:
                            x_masked[tt, ff] = 0
                        # Also mask (t, f)
                        x_masked_tf = x_masked.copy()
                        x_masked_tf[t, f] = 0
                        out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        out_masked_tf = self.model(torch.tensor(x_masked_tf[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
                        mc.append(out_masked_tf - out_masked)
                    shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x_orig[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            x_all_masked = np.zeros_like(x_orig)
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).detach().cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[AttnSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

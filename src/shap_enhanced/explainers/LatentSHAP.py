"""
LatentSHAP: SHAP in Latent Space Explainer

Runs SHAP attribution in the latent space (from an encoder), then maps the latent attributions back to the original input space using a decoder.

References:
- https://arxiv.org/abs/2006.06015 (Latent Attribution for Deep Models)
"""

import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class LatentSHAPExplainer(BaseExplainer):
    """
    LatentSHAP: SHAP Explainer in latent (encoded) space.

    Parameters
    ----------
    model : Any
        The model to be explained (expects latent input if using separate decoder).
    encoder : Callable
        Encoder module mapping input (T, F) -> latent.
    decoder : Callable
        Decoder mapping latent -> input space.
    base_explainer_class : class
        SHAP-style explainer to use in latent space (e.g. KernelExplainer, DeepExplainer).
    background : np.ndarray or torch.Tensor
        Background data (in input space).
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        encoder,
        decoder,
        base_explainer_class,
        background,
        device=None,
        base_explainer_kwargs=None,
    ):
        super().__init__(model, background)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_explainer_kwargs = base_explainer_kwargs or {}

        # Precompute latent background for SHAP
        print("[LatentSHAP] Encoding background set...")
        self.latent_bg = self._to_latent(background)
        self.base_explainer = base_explainer_class(
            model=self.model_in_latent,
            background=self.latent_bg,
            **self.base_explainer_kwargs
        )
        print("[LatentSHAP] Base explainer initialized.")

    def _to_latent(self, X):
        # X: (N, T, F) np or torch
        was_np = False
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
            was_np = True
        X_latent = self.encoder(X)
        return X_latent.detach().cpu().numpy() if was_np else X_latent

    def _latent_to_input_jacobian(self, x_input):
        """
        Computes decoder Jacobian d(input)/d(latent) for linear projection.
        For local explanation: use decoder's local linearization (per sample).
        """
        x_input = torch.tensor(x_input, dtype=torch.float32, device=self.device).unsqueeze(0).requires_grad_()
        latent = self.encoder(x_input)
        latent_dim = latent.shape[-1]
        T, F = x_input.shape[1:]
        jacobian = []
        for i in range(latent_dim):
            grad_out = torch.zeros_like(latent)
            grad_out[..., i] = 1
            decoder_out = self.decoder(latent)
            decoder_out.backward(grad_out, retain_graph=True)
            jacobian.append(x_input.grad.detach().cpu().numpy().copy())
            x_input.grad.zero_()
        jac = np.stack(jacobian, axis=-1)  # shape (1, T, F, latent_dim)
        return jac[0]  # (T, F, latent_dim)

    def model_in_latent(self, Z):
        # Given latent Z, run through decoder (if needed) then model
        if isinstance(Z, np.ndarray):
            Z = torch.tensor(Z, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            x_decoded = self.decoder(Z)
            out = self.model(x_decoded)
            return out.cpu().numpy() if hasattr(out, "cpu") else out

    def shap_values(
        self,
        X,
        **kwargs
    ):
        """
        Steps:
        1. Encode X into latent space.
        2. Compute SHAP values in latent space.
        3. Project latent SHAP attributions to input space using decoder's local Jacobian.
        """
        is_torch = hasattr(X, 'detach')
        X_np = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        if len(X_np.shape) == 2:
            X_np = X_np[None, ...]
            single = True
        else:
            single = False

        Z = self._to_latent(X_np)
        latent_shap = self.base_explainer.shap_values(Z, **kwargs)  # shape (B, latent_dim) or (latent_dim,) or similar

        # Project back to input space
        input_shap = []
        for i in range(X_np.shape[0]):
            # Local Jacobian at this input
            jac = self._latent_to_input_jacobian(X_np[i])
            # If latent_shap is (latent_dim,), make it (latent_dim,)
            latent_attr = latent_shap[i] if latent_shap.ndim > 1 else latent_shap
            # Project: sum_j J_ij * phi_j  (chain rule)
            phi_input = np.tensordot(jac, latent_attr, axes=[-1, 0])  # (T, F)
            input_shap.append(phi_input)
        input_shap = np.stack(input_shap, axis=0)
        return input_shap[0] if single else input_shap

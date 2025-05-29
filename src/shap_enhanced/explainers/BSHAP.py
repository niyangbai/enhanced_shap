"""
BShap: Distribution-Free SHAP Explainer

Estimates feature (timestep, feature) contributions by perturbing with uniform or marginal (per-feature) sampling, without relying on the empirical distribution or any background dataset.

Reference:
  - https://www.tandfonline.com/doi/full/10.1080/02331888.2025.2487853
"""

import numpy as np
import torch
from ..base_explainer import BaseExplainer

class BShapExplainer(BaseExplainer):
    """
    BShap: Distribution-Free SHAP Explainer

    Parameters
    ----------
    model : Any
        Model to be explained.
    X_domain : np.ndarray
        (Optional) Array or tuple describing the domain of possible input values for uniform/marginal sampling,
        e.g., (T, F, 2) for per-feature min/max, or shape (N, T, F) for marginal sampling.
    n_samples : int
        Number of random perturbation samples per SHAP value.
    sampling : str
        'uniform' (default, draws from feature min/max) or 'marginal' (draws from empirical marginal per feature).
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        X_domain=None,
        n_samples=100,
        sampling="uniform",
        device=None
    ):
        super().__init__(model, background=None)
        self.X_domain = X_domain
        self.n_samples = n_samples
        self.sampling = sampling
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Precompute feature domains if available
        self.feature_min, self.feature_max, self.marginals = self._extract_domains(X_domain)

    def _extract_domains(self, X_domain):
        if X_domain is None:
            return 0.0, 1.0, None
        if self.sampling == "uniform":
            # X_domain is expected as (T, F, 2) with min/max or (N, T, F)
            if isinstance(X_domain, tuple) or isinstance(X_domain, list):
                feature_min, feature_max = X_domain
            elif isinstance(X_domain, np.ndarray) and X_domain.ndim == 4:
                # e.g. shape (T, F, 2)
                feature_min = X_domain[..., 0]
                feature_max = X_domain[..., 1]
            elif isinstance(X_domain, np.ndarray) and X_domain.ndim == 3:
                feature_min = X_domain.min(axis=0)
                feature_max = X_domain.max(axis=0)
            else:
                raise ValueError("Provide X_domain as (T, F, 2) or (N, T, F)")
            return feature_min, feature_max, None
        elif self.sampling == "marginal":
            # Store all samples as marginal empirical distribution for each (t, f)
            return None, None, X_domain  # X_domain: (N, T, F)
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

    def _sample_value(self, t, f):
        if self.sampling == "uniform":
            return np.random.uniform(self.feature_min[t, f], self.feature_max[t, f])
        elif self.sampling == "marginal":
            N = self.marginals.shape[0]
            idx = np.random.randint(0, N)
            return self.marginals[idx, t, f]
        else:
            raise ValueError(f"Unknown sampling method: {self.sampling}")

    def _perturb(self, x, mask):
        # For each (t, f) in mask, replace with random value
        x_perturbed = x.copy()
        for (t, f) in mask:
            x_perturbed[t, f] = self._sample_value(t, f)
        return x_perturbed

    def shap_values(
        self,
        X,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        Estimate SHAP values with distribution-free perturbations.

        Parameters
        ----------
        X : np.ndarray or torch.Tensor
            (T, F) or (B, T, F)
        Returns
        -------
        shap_vals : np.ndarray
            (T, F) or (B, T, F)
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
            x = X_in[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(self.n_samples):
                        # Sample random coalition (subset of available)
                        k = np.random.randint(1, len(available)+1)
                        mask = np.random.choice(len(available), size=k, replace=False)
                        mask_idxs = [available[i] for i in mask]
                        x_masked = self._perturb(x, mask_idxs)
                        x_masked_tf = self._perturb(x_masked, [(t, f)])
                        # Evaluate marginal contribution
                        out_masked = self.model(torch.tensor(x_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        out_masked_tf = self.model(torch.tensor(x_masked_tf[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                        mc.append(out_masked_tf - out_masked)
                    shap_vals[b, t, f] = np.mean(mc)

            # Additivity normalization per sample
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            x_all_perturbed = self._perturb(x, all_pos)
            perturbed_pred = self.model(torch.tensor(x_all_perturbed[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - perturbed_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[BShap Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - perturbed_pred):.4f}")
        return shap_vals

"""
Sparse Coalition SHAP Explainer

Only forms coalitions (feature subsets to mask) that produce valid sparse patterns, such as one-hot encodings or binary support. For one-hot sets, masking means setting the entire group to zero (no class selected), never producing invalid or fractional vectors.

Parameters
----------
model : Any
    Model to be explained.
background : np.ndarray or torch.Tensor
    Background for optional mean/mode imputation (not used for masking).
onehot_groups : list of lists
    For one-hot data, each sublist contains feature indices of a one-hot set, e.g. [[0,1,2],[3,4,5],...].
mask_strategy : str
    'zero' (default): mask sets features to zero (no class).
    'observed': use only observed masked patterns from the dataset (optional, not implemented here for brevity).
device : str
    'cpu' or 'cuda'.
"""
import numpy as np
import torch
from ..base_explainer import BaseExplainer

class SparseCoalitionSHAPExplainer(BaseExplainer):
    def __init__(
        self,
        model,
        background,
        onehot_groups=None,
        mask_strategy="zero",
        device=None
    ):
        super().__init__(model, background)
        self.onehot_groups = onehot_groups  # e.g., [[0,1,2],[3,4,5],...]
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _mask(self, x, groups_to_mask):
        # x: (T, F)
        x_masked = x.copy()
        if self.onehot_groups is not None:
            for group in groups_to_mask:
                for idx in group:
                    x_masked[:, idx] = 0
        else:
            # For general binary: simply set to zero
            for idx in groups_to_mask:
                t, f = idx
                x_masked[t, f] = 0
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
        SHAP with valid sparse coalitions: (B, T, F) or (T, F)
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
            if self.onehot_groups is not None:
                all_groups = self.onehot_groups
                group_flat = [idx for group in all_groups for idx in group]
                for group in all_groups:
                    for idx in group:
                        contribs = []
                        groups_others = [g for g in all_groups if g != group]
                        for _ in range(nsamples):
                            # Sample random subset of other groups to mask
                            k = np.random.randint(0, len(groups_others)+1)
                            C_idxs = np.random.choice(len(groups_others), size=k, replace=False)
                            mask_groups = [groups_others[i] for i in C_idxs]
                            # Mask C (other groups)
                            x_C = self._mask(x, mask_groups)
                            # Mask C + this group
                            x_C_g = self._mask(x_C, [group])
                            out_C = self.model(torch.tensor(x_C[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                            out_C_g = self.model(torch.tensor(x_C_g[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                            contribs.append(out_C - out_C_g)
                        # Assign SHAP value to all features in this group equally (or just to idx)
                        shap_vals[b, :, idx] = np.mean(contribs) / len(group)
            else:
                # General binary: per (t, f)
                all_pos = [(t, f) for t in range(T) for f in range(F)]
                for t in range(T):
                    for f in range(F):
                        contribs = []
                        available = [idx for idx in all_pos if idx != (t, f)]
                        for _ in range(nsamples):
                            # Mask random subset of others
                            k = np.random.randint(0, len(available)+1)
                            C_idxs = np.random.choice(len(available), size=k, replace=False)
                            mask_idxs = [available[i] for i in C_idxs]
                            x_C = self._mask(x, [mask_idxs])
                            x_C_tf = self._mask(x_C, [[(t, f)]])
                            out_C = self.model(torch.tensor(x_C[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                            out_C_tf = self.model(torch.tensor(x_C_tf[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
                            contribs.append(out_C - out_C_tf)
                        shap_vals[b, t, f] = np.mean(contribs)
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            # Mask all groups/positions
            if self.onehot_groups is not None:
                x_all_masked = self._mask(x, self.onehot_groups)
            else:
                all_pos = [(t, f) for t in range(T) for f in range(F)]
                x_all_masked = self._mask(x, [all_pos])
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum

        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[SparseCoalitionSHAP] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")
        return shap_vals

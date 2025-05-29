"""
Hierarchical SHAP (h-SHAP) Explainer

Computes SHAP values hierarchically using structured feature grouping.
Supports recursive estimation at group and subgroup levels.

Reference:
  - https://arxiv.org/abs/2104.06164
"""

import numpy as np
import torch
from ..base_explainer import BaseExplainer

class HShapExplainer(BaseExplainer):
    """
    h-SHAP: Hierarchical SHAP Explainer

    Parameters
    ----------
    model : Any
        Model to be explained.
    background : np.ndarray or torch.Tensor
        (N, T, F), for mean imputation.
    hierarchy : list
        Feature hierarchy as nested lists of (t, f) tuples (see below).
        Example: [[(0,0), (0,1)], [(1,0), (1,1)], ...] for block-of-2 time steps.
    mask_strategy : str
        'mean' or 'zero' (default: 'mean').
    device : str
        'cpu' or 'cuda'.
    """
    def __init__(
        self,
        model,
        background,
        hierarchy,
        mask_strategy="mean",
        device=None
    ):
        super().__init__(model, background)
        self.hierarchy = hierarchy  # List of groups, or nested list
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if mask_strategy == "mean":
            self._mean = background.mean(axis=0)
        else:
            self._mean = None

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

    def _shap_group(self, x, group_idxs, rest_idxs, nsamples=50):
        # Estimate marginal contribution of 'group' vs. 'rest'
        contribs = []
        all_idxs = group_idxs + rest_idxs
        for _ in range(nsamples):
            # Sample subset of rest to mask
            k = np.random.randint(0, len(rest_idxs) + 1)
            rest_sample = np.random.choice(rest_idxs, size=k, replace=False) if k > 0 else []
            # Mask: (rest_sample only), then (rest_sample + group)
            x_rest = self._impute(x, rest_sample)
            x_both = self._impute(x_rest, group_idxs)
            out_rest = self.model(torch.tensor(x_rest[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            out_both = self.model(torch.tensor(x_both[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            contribs.append(out_rest - out_both)
        return np.mean(contribs)

    def _explain_recursive(self, x, groups, nsamples=50, attributions=None):
        """
        Recursively explain at group and subgroup levels.
        groups: list of lists or tuples
        Returns: attribution dict {(t, f): value}
        """
        if attributions is None:
            attributions = {}
        group_indices = []
        for group in groups:
            if isinstance(group[0], (tuple, list)):
                # group is a nested group, recurse
                sub_attr = self._explain_recursive(x, group, nsamples, attributions)
                # group_indices += flatten(group)
                group_indices += [idx for g in group for idx in (g if isinstance(g[0], (tuple, list)) else [g])]
            else:
                group_indices += [group]

        # At this hierarchy level, estimate group SHAP for each group
        for group in groups:
            if isinstance(group[0], (tuple, list)):
                flat_group = [idx for g in group for idx in (g if isinstance(g[0], (tuple, list)) else [g])]
            else:
                flat_group = [group]
            rest = [idx for idx in group_indices if idx not in flat_group]
            phi = self._shap_group(x, flat_group, rest, nsamples=nsamples)
            # Split SHAP value equally among group members
            for idx in flat_group:
                attributions[idx] = attributions.get(idx, 0.0) + phi / len(flat_group)
        return attributions

    def shap_values(
        self,
        X,
        nsamples=50,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        """
        Hierarchical SHAP: computes SHAP values for each (t, f) via recursive group explanation.
        Returns: (T, F) or (B, T, F)
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
            attr = self._explain_recursive(x, self.hierarchy, nsamples=nsamples)
            for (t, f), v in attr.items():
                shap_vals[b, t, f] = v
            # Additivity normalization
            orig_pred = self.model(torch.tensor(x[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            x_all_masked = self._impute(x, all_pos)
            masked_pred = self.model(torch.tensor(x_all_masked[None], dtype=torch.float32, device=self.device)).cpu().numpy().squeeze()
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[h-SHAP Additivity] sum(SHAP)={shap_vals.sum():.4f} | Model diff={float(orig_pred - masked_pred):.4f}")
        return shap_vals

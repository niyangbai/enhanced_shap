"""
Support-Preserving SHAP Explainer
=================================

Theoretical Explanation
-----------------------

Support-Preserving SHAP is a specialized feature attribution method tailored for **sparse** or **structured discrete data**,  
such as one-hot encodings or binary presence/absence features. Unlike traditional SHAP variants that create  
synthetic masked inputs (often resulting in out-of-distribution samples), this explainer **only evaluates inputs that have  
been observed in the dataset** and match the support pattern induced by masking.

For each coalition (subset of features to mask), the method attempts to find a real background sample  
with the **same binary support** (nonzero positions) as the masked instance. If no such sample exists, the coalition  
is skipped or flagged—ensuring that only valid, realistic inputs are used for estimating SHAP values.

For continuous or dense data, the method gracefully falls back to **mean-masking** (standard SHAP behavior).

Key Concepts
^^^^^^^^^^^^

- **Support Pattern Matching**:  
    Masked inputs are replaced with real background examples that match the nonzero pattern (support)  
    of the masked input. This maintains validity and avoids generating unrealistic inputs.

- **One-Hot / Binary Support**:  
    Especially effective for categorical features encoded as one-hot vectors or binary indicators.  
    Masking respects group structures and ensures feasible combinations.

- **Graceful Fallback**:  
    When applied to continuous or dense data, the explainer defaults to mean-masking to retain applicability.

- **Additivity Normalization**:  
    Final attributions are scaled such that their total equals the model output difference between  
    the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, device context, and configuration for skipping or flagging unmatched patterns.

2. **Support-Preserving Masking**:
    - For each sampled coalition of masked features:
        - Create a masked version of the input.
        - Find a background example with the same binary support (nonzero positions).
        - If no match is found, either skip or raise an exception based on configuration.
        - For non-sparse (dense) inputs, fallback to mean-masking.

3. **SHAP Value Estimation**:
    - For each feature:
        - Repeatedly sample coalitions of other features.
        - For each:
            - Mask the coalition and find a matching background sample.
            - Mask the coalition plus the feature of interest and find another match.
            - Compute the model output difference.
        - Average these differences to estimate the feature’s marginal contribution.

4. **Normalization**:
    - Scale the final attributions so their sum matches the model output difference  
        between the unmasked and fully-masked input.

Use Case
--------

Ideal for:
    - One-hot encoded categorical features.
    - Binary indicators (presence/absence).
    - Sparse high-dimensional data where only valid observed patterns should be used for attribution.
"""



import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer

class SupportPreservingSHAPExplainer(BaseExplainer):
    r"""
    SupportPreservingSHAPExplainer: Real-Pattern-Constrained SHAP Estimator

    This explainer approximates SHAP values by generating only masked inputs that match real examples 
    in the dataset—preserving the discrete or sparse structure of the input space. It avoids 
    out-of-distribution perturbations by requiring coalitions (masked variants) to have binary 
    support patterns that exist in the original data.

    If the data is not sparse (e.g., continuous), the method falls back to mean-masking, 
    akin to standard SHAP explainers.

    :param model: Predictive model to explain.
    :type model: Any
    :param background: Dataset used to match support patterns (shape: (N, T, F) or (N, F)).
    :type background: np.ndarray or torch.Tensor
    :param bool skip_unmatched: If True, coalitions without support-matching background samples are skipped.
    :param str device: Device to evaluate model on ('cpu' or 'cuda').
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
        r"""
        Compute SHAP values by evaluating only valid support-preserving perturbations.

        For sparse inputs (e.g., one-hot or binary):
        
            - For each feature, sample coalitions of other features.
            - Construct masked inputs and locate matching background samples with same nonzero support.
            - Evaluate model differences with and without the feature of interest.
            - Average differences to estimate SHAP values.

        For dense inputs:
        - Fallback to standard mean-based masking for each feature individually.

        .. math::
            \phi_i = \mathbb{E}_{S \subseteq N \setminus \{i\}} \left[
                f(x_{S \cup \{i\}}) - f(x_S)
            \right]

        Final attributions are normalized such that:

        .. math::
            \sum_i \phi_i = f(x) - f(x_{\text{masked}})

        :param X: Input sample or batch of shape (T, F) or (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of coalition samples per feature.
        :param bool check_additivity: If True, prints sum of SHAP vs model output difference.
        :param int random_seed: Seed for reproducibility.
        :return: SHAP attributions with same shape as input.
        :rtype: np.ndarray
        """
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

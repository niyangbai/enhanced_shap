r"""
BShapExplainer: Distribution-Free SHAP for Sequential Models
============================================================

Theoretical Explanation
-----------------------

BShap is a distribution-free variant of the SHAP framework, specifically designed for sequential models such as LSTMs.
Unlike classical SHAP methods that rely on empirical data (e.g., using mean or sample values as feature baselines),
BShap masks features using uninformative replacements such as uniform noise, Gaussian noise, or zeros.
This makes it particularly suitable when the underlying data distribution is unknown, unreliable, or intentionally ignored.

By avoiding assumptions about the data, BShap enables a cleaner interpretation of how a model behaves under
entirely synthetic perturbations—revealing how features contribute even when removed from their contextual correlations.

Key Concepts
^^^^^^^^^^^^

- **Distribution-Free Masking**: Masked features are replaced with independently sampled values that do not rely on the input data distribution.
- **Masking Strategies**:
  - `'random'`: Sample feature values uniformly at random from a defined range (default).
  - `'noise'`: Add Gaussian noise to masked feature values.
  - `'zero'`: Set masked features to zero.
- **No Data Assumptions**: All masking is performed without drawing from empirical feature distributions.
- **Additivity Normalization**: Feature attributions are normalized so that their sum equals the change in model output
  between the original and fully-masked input.

Algorithm
---------

1. **Initialization**:
   - Accepts a model, a value range for feature sampling, the number of samples, masking strategy (`'random'`, `'noise'`, `'zero'`), and device context.

2. **Masking**:
    - For each coalition (a subset of features to mask), masked values are replaced by:
        - Random values (uniform), or
        - Gaussian noise, or
        - Zeros, depending on the selected strategy.

3. **SHAP Value Estimation**:
    - For each feature:
        - Randomly select a subset of other features to mask.
        - Compute model output for:
            - The input with the coalition masked.
            - The input with the coalition plus the current feature masked.
        - Record and average the difference in outputs as the estimated contribution.

4. **Normalization**:
    - Scale the attributions so that their sum equals the difference between the original and fully-masked model outputs.

References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**
  [SHAP foundation—coalitional feature attribution framework]

- **Molnar, “Interpretable Machine Learning” (2022), SHAP chapter**
  [Discusses various SHAP variants, including noise or zero baselines vs. data-driven baselines]

- **Chen, Lundberg & Lee (2022), “Explaining a Series of Models by Propagating Shapley Values” (Generalized DeepSHAP)**
  [Highlights how interpretation depends on baseline distributions and how multiple or synthetic baselines influence attribution] :contentReference[oaicite:2]{index=2}
"""

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class BShapExplainer(BaseExplainer):
    r"""
    BShap: Distribution-Free SHAP Explainer for Sequential Models

    Implements a SHAP-based attribution method that avoids empirical data distribution assumptions
    by applying synthetic masking strategies (e.g., uniform noise, Gaussian noise, or zero).
    This is useful for evaluating model robustness or interpretability in data-agnostic contexts.

    :param model: Sequence model to explain.
    :param input_range: Tuple of (min, max) or arrays defining per-feature value bounds. Used for random masking.
    :type input_range: tuple or (np.ndarray, np.ndarray)
    :param int n_samples: Number of coalitions sampled per feature.
    :param str mask_strategy: Masking strategy: 'random', 'noise', or 'zero'.
    :param str device: Device identifier, e.g., 'cpu' or 'cuda'.
    """

    def __init__(
        self, model, input_range=None, n_samples=50, mask_strategy="random", device=None
    ):
        super().__init__(model, background=None)
        self.input_range = input_range
        self.n_samples = n_samples
        self.mask_strategy = mask_strategy
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _mask(self, x, mask_idxs):
        r"""
        Apply masking to the input at specified (t, f) indices.

        Masked values are replaced using the selected masking strategy:
        - 'random': Sampled uniformly from input_range or [-1, 1].
        - 'noise': Original value plus Gaussian noise.
        - 'zero': Set to zero.

        :param x: Input sample of shape (T, F).
        :type x: np.ndarray
        :param mask_idxs: List of (t, f) index pairs to mask.
        :type mask_idxs: list[tuple[int, int]]
        :return: Masked input.
        :rtype: np.ndarray
        """
        x_masked = x.copy()
        T, F = x.shape
        for t, f in mask_idxs:
            if self.mask_strategy == "random":
                # Per-feature min/max or fallback
                if self.input_range is not None:
                    mn, mx = self.input_range
                    if isinstance(mn, np.ndarray):
                        x_masked[t, f] = np.random.uniform(mn[f], mx[f])
                    else:
                        x_masked[t, f] = np.random.uniform(mn, mx)
                else:
                    x_masked[t, f] = np.random.uniform(-1, 1)
            elif self.mask_strategy == "noise":
                x_masked[t, f] = x[t, f] + np.random.normal(0, 0.5)
            elif self.mask_strategy == "zero":
                x_masked[t, f] = 0.0
            else:
                raise ValueError("Unknown mask_strategy")
        return x_masked

    def shap_values(
        self, X, nsamples=None, check_additivity=True, random_seed=42, **kwargs
    ):
        r"""
        Compute SHAP values using distribution-free perturbations.

        Estimates marginal feature contributions by averaging differences between model outputs
        under masked coalitions. Uses synthetic masking based on the configured strategy
        without any reliance on background data statistics.

        Final attributions are normalized to satisfy the SHAP additivity constraint:

        .. math::
            \sum_{t=1}^T \sum_{f=1}^F \phi_{t,f} \approx f(x) - f(x_{\text{masked}})

        :param X: Input of shape (T, F) or (B, T, F)
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of coalition samples per feature (defaults to self.n_samples).
        :param bool check_additivity: Print diagnostic message for SHAP sum vs. model delta.
        :param int random_seed: Seed for reproducibility.
        :return: SHAP values of shape (T, F) or (B, T, F)
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        if nsamples is None:
            nsamples = self.n_samples
        is_torch = hasattr(X, "detach")
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = len(X_in.shape) == 2
        if single:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        all_pos = [(t, f) for t in range(T) for f in range(F)]
        for b in range(B):
            x = X_in[b]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        k = np.random.randint(1, len(available) + 1)
                        mask_idxs = [
                            available[i]
                            for i in np.random.choice(len(available), k, replace=False)
                        ]
                        x_masked = self._mask(x, mask_idxs)
                        x_masked_tf = self._mask(x_masked, [(t, f)])
                        out_masked = (
                            self.model(
                                torch.tensor(
                                    x_masked[None],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        out_masked_tf = (
                            self.model(
                                torch.tensor(
                                    x_masked_tf[None],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        out_masked = float(np.ravel(out_masked)[0])  # ensure scalar
                        out_masked_tf = float(np.ravel(out_masked_tf)[0])
                        mc.append(out_masked_tf - out_masked)
                    shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = float(
                np.ravel(
                    self.model(
                        torch.tensor(x[None], dtype=torch.float32, device=self.device)
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )[0]
            )
            x_all_masked = self._mask(x, all_pos)
            masked_pred = float(
                np.ravel(
                    self.model(
                        torch.tensor(
                            x_all_masked[None], dtype=torch.float32, device=self.device
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )[0]
            )
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if np.abs(shap_sum) > 1e-8:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[BShap] sum(SHAP)={shap_vals.sum():.4f} (should match model diff)")
        return shap_vals

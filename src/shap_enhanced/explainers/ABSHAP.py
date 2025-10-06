r"""
Adaptive Baseline SHAP (Sparse)
===============================

Theoretical Explanation
-----------------------

Adaptive Baseline SHAP (ABSHAP) is a feature attribution method built upon the SHAP framework.
It is specifically designed to yield valid, interpretable explanations for both dense (e.g., continuous or tabular)
and sparse (e.g., categorical or one-hot encoded) input data.

Unlike traditional SHAP methods that use static baselines (such as zeros or means), ABSHAP dynamically samples baselines
for masked features from real observed background samples. This helps avoid out-of-distribution perturbations,
which is particularly critical for sparse or categorical data where unrealistic combinations can easily arise.

Key Concepts
^^^^^^^^^^^^

- **Adaptive Masking**: Each feature’s masking method is chosen based on its distribution:
    - Dense/continuous features use the mean value from the background dataset.
    - Sparse/categorical features (e.g., those with >90% zeros) are replaced using values from real background examples.
- **Strategy Selection**: The masking approach can be assigned automatically per feature or manually specified by the user.
- **Valid Perturbations**: All masked samples are guaranteed to lie within the original data distribution, preventing unrealistic inputs.

Algorithm
---------

1. **Initialization**:
    - Accepts a model, background dataset, number of baselines, masking strategy, and device.
    - Automatically determines a masking strategy per feature or uses a user-specified configuration.
    - Computes mean values for dense-feature masking.

2. **Masking**:
    - For each coalition (a selected subset of features to mask), masked values are replaced with:
        - The feature-wise mean (dense features), or
        - A sampled value from a real background example (sparse features).

3. **SHAP Value Estimation**:
    - For each feature:
        - Randomly sample subsets of other features to mask.
        - For each sampled baseline:
            - Compute model outputs on:
                - Input with selected features masked.
                - Input with selected features plus the current feature masked.
            - Calculate the difference in model outputs.
        - Average these differences to estimate the marginal contribution of the feature.
    - Normalize the resulting attributions so their sum equals the difference between the original and fully-masked model outputs.

References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**
  [SHAP foundation]

- **Merrick & Taly (2020), “Keep it Real: Towards Realistic and Efficient Shapley Value Explanations”**
  [Proposes adaptive masking based on feature type and using real data to avoid out-of-distribution perturbations]

- **Molnar, “Interpretable Machine Learning” (2022), SHAP chapter**
  [Summarizes best practices and practical warnings about feature masking in SHAP for different data types]

"""

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class AdaptiveBaselineSHAPExplainer(BaseExplainer):
    r"""
    Adaptive Baseline SHAP (ABSHAP) Explainer for Dense and Sparse Features.

    Implements a SHAP explainer that adaptively masks features based on their data distribution:
    using mean-based masking for continuous features and sample-based masking for sparse or
    categorical features. This ensures valid perturbations and avoids out-of-distribution artifacts.

    .. note::
        Feature masking strategy can be determined automatically or manually specified.

    .. warning::
        Adaptive masking requires background data and introduces computational overhead.

    :param Callable model: Model to be explained. Should accept PyTorch tensors as input.
    :param Union[np.ndarray, torch.Tensor] background: Background dataset for baseline sampling.
        Shape: (N, F) or (N, T, F).
    :param int n_baselines: Number of baselines to sample per explanation. Default is 10.
    :param Union[str, Sequence[str]] mask_strategy: Either "auto" for detection or list per feature.
    :param str device: PyTorch device identifier, e.g., "cpu" or "cuda". Defaults to auto-detection.
    """

    def __init__(
        self,
        model: Any,
        background: np.ndarray | torch.Tensor,
        n_baselines: int = 10,
        mask_strategy: str | Sequence[str] = "auto",
        device: str = None,
    ):
        super().__init__(model, background)
        bg = (
            background.detach().cpu().numpy()
            if hasattr(background, "detach")
            else np.asarray(background)
        )
        self.background = bg if bg.ndim == 3 else bg[:, None, :]  # (N, T, F)
        self.N, self.T, self.F = self.background.shape
        self.n_baselines = n_baselines
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # Determine masking strategy per feature
        if mask_strategy == "auto":
            # For each feature, if >90% zeros in background, use 'adaptive' masking, else 'mean'
            self.feature_strategies = []
            for f in range(self.F):
                bg_feat = self.background[..., f].flatten()
                zero_frac = np.mean(bg_feat == 0)
                self.feature_strategies.append(
                    "adaptive" if zero_frac > 0.9 else "mean"
                )
        elif isinstance(mask_strategy, list | tuple | np.ndarray):
            assert len(mask_strategy) == self.F
            self.feature_strategies = list(mask_strategy)
        elif isinstance(mask_strategy, str):
            # All features use the same
            self.feature_strategies = [mask_strategy] * self.F
        else:
            raise ValueError(f"Invalid mask_strategy: {mask_strategy}")

        self.mean_baseline = np.mean(self.background, axis=0)  # (T, F)

    def _select_baselines(self, x: np.ndarray, n: int) -> np.ndarray:
        r"""
        Randomly selects `n` baseline samples from the background dataset.

        :param np.ndarray x: Input instance for context (unused directly).
        :param int n: Number of baselines to sample.
        :return np.ndarray: Sampled baseline instances of shape (n, T, F).
        """
        idx = np.random.choice(self.N, n, replace=True)
        return self.background[idx]  # (n, T, F)

    def _mask_input(
        self, x: np.ndarray, baseline: np.ndarray, mask: list
    ) -> np.ndarray:
        r"""
        Applies feature-wise masking to the input sample `x`.

        If the feature is continuous, replaces with the mean; if categorical/sparse, uses baseline sample.

        :param np.ndarray x: Original input of shape (T, F).
        :param np.ndarray baseline: Baseline sample used for masking.
        :param list mask: List of (t, f) tuples indicating which positions to mask.
        :return np.ndarray: Masked input of shape (T, F).
        """
        x_masked = x.copy()
        for t, f in mask:
            if self.feature_strategies[f] == "mean":
                x_masked[t, f] = self.mean_baseline[t, f]
            else:  # "adaptive"
                x_masked[t, f] = baseline[t, f]
        return x_masked

    def _full_mask(self, x: np.ndarray) -> np.ndarray:
        r"""
        Returns a fully-masked version of the input `x` for normalization purposes.

        Continuous features are mean-masked, sparse features use a single baseline sample.

        :param np.ndarray x: Input to be masked, of shape (T, F).
        :return np.ndarray: Fully-masked version of `x`.
        """
        x_masked = x.copy()
        baseline = self._select_baselines(x, 1)[0]
        for t in range(self.T):
            for f in range(self.F):
                if self.feature_strategies[f] == "mean":
                    x_masked[t, f] = self.mean_baseline[t, f]
                else:
                    x_masked[t, f] = baseline[t, f]
        return x_masked

    def shap_values(
        self,
        X: np.ndarray | torch.Tensor,
        nsamples: int = 100,
        random_seed: int = 42,
        **kwargs,
    ) -> np.ndarray:
        r"""
        Estimates SHAP values for the given input `X` using the ABSHAP algorithm.

        For each feature (t, f), estimates its marginal contribution by comparing model
        outputs with and without the feature masked, averaging over sampled coalitions and baselines.

        .. math::
            \phi_{i} = \mathbb{E}_{S \subseteq N \setminus \{i\}} \left[ f(x_{S \cup \{i\}}) - f(x_S) \right]

        The attributions are then normalized to match the output difference between the original and
        fully-masked prediction.

        :param Union[np.ndarray, torch.Tensor] X: Input samples, shape (B, T, F) or (T, F).
        :param int nsamples: Number of masking combinations per feature. Default is 100.
        :param int random_seed: Seed for reproducibility. Default is 42.
        :return np.ndarray: SHAP values of shape (T, F) or (B, T, F).
        """
        np.random.seed(random_seed)
        X = X.detach().cpu().numpy() if hasattr(X, "detach") else np.asarray(X)
        single = X.ndim == 2
        X = X[None, ...] if single else X  # (B, T, F)
        B, T, F = X.shape
        out = np.zeros((B, T, F), dtype=np.float32)

        for b in range(B):
            x = X[b]
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    vals = []
                    other_pos = [p for p in all_pos if p != (t, f)]
                    for _ in range(nsamples):
                        k = np.random.randint(1, len(other_pos) + 1)
                        mask_idxs = [
                            other_pos[i]
                            for i in np.random.choice(len(other_pos), k, replace=False)
                        ]
                        for baseline in self._select_baselines(x, self.n_baselines):
                            x_masked = self._mask_input(x, baseline, mask_idxs)
                            x_masked_plus = self._mask_input(
                                x_masked, baseline, [(t, f)]
                            )
                            pred_masked = float(
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
                            pred_masked_plus = float(
                                self.model(
                                    torch.tensor(
                                        x_masked_plus[None],
                                        dtype=torch.float32,
                                        device=self.device,
                                    )
                                )
                                .detach()
                                .cpu()
                                .numpy()
                                .squeeze()
                            )
                            vals.append(pred_masked_plus - pred_masked)
                    out[b, t, f] = np.mean(vals)

            # Normalize attributions to match output difference (SHAP style)
            orig_pred = float(
                self.model(
                    torch.tensor(x[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            full_masked = self._full_mask(x)
            full_pred = float(
                self.model(
                    torch.tensor(
                        full_masked[None], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            diff = orig_pred - full_pred
            summ = out[b].sum()
            if np.abs(summ) > 1e-8:
                out[b] *= diff / summ

        return out[0] if single else out

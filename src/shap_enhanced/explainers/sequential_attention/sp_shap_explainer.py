"""Sequential Perturbation SHAP (SP-SHAP) Explainer (Robust Version).

This module implements a robust variant of SP-SHAP for sequential models,
incorporating context-aware masking, gradient-based boosting, and safeguards
against unstable gradients and masking artifacts.

The explainer approximates the Shapley value for each timestep in a sequence by:

.. math::

    \phi_t = \mathbb{E}_{S \subseteq T \setminus \{t\}} \left[ f(S \cup \{t\}) - f(S) \right]

For sequence models like LSTMs, this is extended to:

.. math::

    \phi_t = \mathbb{E}_{S \subseteq T \setminus C(t)} \left[ f(S \cup C(t)) - f(S) \right]

where :math:`C(t)` denotes a temporal window around timestep :math:`t`.

"""

import torch
import random
import numpy as np
from typing import Any, Optional, List
from shap_enhanced.explainers.base import BaseExplainer
from scipy.signal import savgol_filter


def sample_coalitions(T: int, nsamples: int, p: float = 0.5) -> List[List[int]]:
    """
    Sample random timestep subsets as coalitions.

    :param T: Total number of timesteps.
    :param nsamples: Number of coalitions to generate.
    :param p: Expected subset fraction.
    :return: List of subsets (each a list of timestep indices).
    """
    subsets = []
    for _ in range(nsamples):
        size = max(1, int(p * T))
        subset = random.sample(range(T), size)
        subsets.append(subset)
    return subsets


def mask_timesteps_robust(x: torch.Tensor, timesteps: List[int]) -> torch.Tensor:
    """
    Mask timesteps via context-aware mean imputation.

    For each timestep `t` in `timesteps`, its value is replaced with the average
    of its neighbors (or fallback values at sequence edges).

    :param x: Input tensor of shape (1, time, features).
    :param timesteps: List of timesteps to mask.
    :return: Masked tensor.
    """
    x_masked = x.clone()
    B, T, F = x.shape
    for t in timesteps:
        if 0 < t < T - 1:
            x_masked[:, t] = 0.5 * (x[:, t - 1] + x[:, t + 1])
        elif t == 0:
            x_masked[:, t] = x[:, 1]
        elif t == T - 1:
            x_masked[:, t] = x[:, T - 2]
        else:
            x_masked[:, t] = 0.0
    return x_masked


class SPShapExplainer(BaseExplainer):
    """
    Robust Sequential Perturbation SHAP Explainer.

    Approximates per-timestep feature attribution using masked perturbations and
    optional gradient-based reweighting. Suitable for RNNs, LSTMs, and other temporal models.

    :param model: Trained sequence model.
    :param nsamples: Number of coalitions to sample for each instance.
    :param window_size: Temporal context window for each perturbed timestep.
    :param target_index: Output index to explain (for multi-output models).
    :param random_seed: Optional seed for reproducibility.
    :param mode: 'default' (coalition masking) or 'simple' (feature-wise masking).
    :param smooth: Whether to apply Savitzky–Golay smoothing.
    :param gradient_boost: Whether to reweight SHAP values using gradient sensitivity.
    """

    def __init__(self, model: Any, nsamples: int = 100, window_size: int = 1,
                 target_index: Optional[int] = 0, random_seed: Optional[int] = None,
                 mode: str = "default", smooth: bool = True,
                 gradient_boost: bool = True) -> None:
        super().__init__(model)
        self.nsamples = nsamples
        self.window_size = window_size
        self.target_index = target_index
        self.mode = mode.lower()
        self.smooth = smooth
        self.gradient_boost = gradient_boost

        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        if self.mode not in ["default", "simple"]:
            raise ValueError(f"Invalid mode '{mode}'.")

    def explain(self, X: torch.Tensor) -> torch.Tensor:
        """
        Generate SP-SHAP attributions for a batch of inputs.

        :param X: Tensor of shape (batch, time, features).
        :return: Attributions of shape (batch, time), normalized to [0, 1].
        """
        B, T, F = X.shape
        attributions = torch.zeros((B, T), device=X.device)

        for i in range(B):
            x = X[i:i+1]
            try:
                if self.mode == "simple":
                    contrib = self._simple_shap(x)
                else:
                    contrib = self._default_shap(x)

                if self.gradient_boost:
                    contrib = self._boost_by_gradient(x, contrib)

                if self.smooth:
                    contrib = torch.tensor(
                        savgol_filter(contrib.cpu().numpy(), 5, 2, mode='nearest'),
                        device=x.device
                    )

                contrib = torch.clamp(contrib, -1.0, 1.0)
                contrib = (contrib - contrib.min()) / (contrib.max() - contrib.min() + 1e-6)

            except Exception as e:
                print(f"[Warning] Skipped sample {i}: {e}")
                contrib = torch.zeros(T, device=X.device)

            attributions[i] = contrib

        return self._format_output(attributions)

    def _default_shap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Full SP-SHAP with context masking.

        For each sampled subset S and timestep t, compute:

        .. math::

            \phi_t \approx \frac{1}{|S|} \sum_S \left[ f(S \cup C(t)) - f(S) \right]

        :param x: Single input sequence (1, time, features).
        :return: SHAP values (time,).
        """
        T = x.shape[1]
        contribs = torch.zeros(T, device=x.device)
        subsets = sample_coalitions(T, self.nsamples)

        for S in subsets:
            try:
                x_S = mask_timesteps_robust(x, list(set(range(T)) - set(S)))
                out_S = self._predict(x_S).item()
            except:
                out_S = self._predict(x).item()

            for t in range(T):
                if t in S:
                    continue
                context = range(max(0, t - self.window_size), min(T, t + self.window_size + 1))
                S_plus = list(set(S + list(context)))

                try:
                    x_St = mask_timesteps_robust(x, list(set(range(T)) - set(S_plus)))
                    out_St = self._predict(x_St).item()
                except:
                    out_St = out_S

                contribs[t] += out_St - out_S

        contribs /= self.nsamples
        return contribs

    def _simple_shap(self, x: torch.Tensor) -> torch.Tensor:
        """
        Simpler variant using (time, feature) masking only.

        :param x: Input sequence (1, time, features).
        :return: SHAP values (time,).
        """
        _, T, F = x.shape
        shap_values = torch.zeros(T, device=x.device)
        pred_x = self._predict(x).item()

        for t in range(T):
            for f in range(F):
                x_mod = x.clone()
                x_mod[:, t, f] = 0.0
                try:
                    pred_mod = self._predict(x_mod).item()
                except:
                    pred_mod = pred_x
                shap_values[t] += pred_x - pred_mod
            shap_values[t] /= F

        return shap_values

    def _boost_by_gradient(self, x: torch.Tensor, contribs: torch.Tensor) -> torch.Tensor:
        """
        Reweight attributions using gradient magnitude at each timestep.

        .. math::

            \hat{\phi}_t = \phi_t \cdot \left( 1 + \alpha \cdot \frac{||\nabla_t f(x)||}{\max ||\nabla_t f(x)||} \right)

        :param x: Input tensor (1, T, F).
        :param contribs: SHAP values (T,).
        :return: Boosted SHAP values (T,).
        """
        x = x.requires_grad_(True)
        try:
            pred = self._predict(x).sum()
            grad = torch.autograd.grad(pred, x, retain_graph=False)[0]  # [1, T, F]
            grad_norm = grad.abs().sum(dim=-1).squeeze()  # [T]
            grad_norm /= (grad_norm.max() + 1e-6)

            contribs = 0.7 * contribs + 0.3 * contribs * grad_norm
        except Exception as e:
            print(f"[Warning] Gradient boost skipped: {e}")
        return contribs

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run model and select correct output index.

        :param x: Input tensor (1, T, F).
        :return: Output tensor (1,) or scalar.
        """
        with torch.no_grad():
            output = self.model(x)
        if output.ndim > 1 and output.shape[1] > 1 and self.target_index is not None:
            return output[:, self.target_index]
        return output.squeeze()

from typing import Any, Optional
import torch
import torch.nn.functional as F
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.perturbation import (
    mask_timesteps,
    perturb_sequence_with_noise
)


class RecurrentExplainer(BaseExplainer):
    """Recurrent Explainer for sequential models using perturbation.

    Estimates feature importance by masking or injecting noise at each timestep.

    .. math::

        \\text{Attribution}_t = f(x) - f(x_{\\setminus t})
    """

    def __init__(self,
                 model: Any,
                 mode: str = "mask",              # 'mask' or 'noise'
                 mask_value: float = 0.0,
                 noise_level: float = 0.05,
                 window_size: int = 1,
                 nsamples: int = 10,
                 target_index: Optional[int] = 0,
                 use_absolute: bool = False,
                 smooth_kernel: int = 3,
                 clip_negative: bool = False
                 ) -> None:
        super().__init__(model)
        self.mode = mode
        self.mask_value = mask_value
        self.noise_level = noise_level
        self.window_size = window_size
        self.nsamples = nsamples
        self.target_index = target_index
        self.use_absolute = use_absolute
        self.smooth_kernel = smooth_kernel
        self.clip_negative = clip_negative

    def _get_output(self, X: torch.Tensor) -> torch.Tensor:
        """Extracts scalar output for each sample in batch."""
        with torch.no_grad():
            out = self.model(X)
            if out.ndim == 2:
                return out[:, self.target_index or 0]
            return out.view(-1)

    def _smooth_attributions(self, attr: torch.Tensor) -> torch.Tensor:
        """Apply moving average smoothing along time axis."""
        if self.smooth_kernel <= 1:
            return attr
        kernel = torch.ones(1, 1, self.smooth_kernel, device=attr.device) / self.smooth_kernel
        attr_unsq = attr.unsqueeze(1)  # [B, 1, T]
        padded = F.pad(attr_unsq, (self.smooth_kernel // 2, self.smooth_kernel // 2), mode='reflect')
        smoothed = F.conv1d(padded, kernel).squeeze(1)  # [B, T]
        return smoothed

    def explain(self, X: torch.Tensor) -> torch.Tensor:
        """Compute per-timestep importance via perturbation.

        :param X: Input tensor of shape [batch, time, features]
        :return: Attributions tensor [batch, time]
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input must be a torch.Tensor")

        batch_size, time_steps, _ = X.shape
        attributions = torch.zeros((batch_size, time_steps), device=X.device)
        base_outputs = self._get_output(X)

        for _ in range(self.nsamples):
            for t in range(time_steps):
                if self.mode == "mask":
                    # Centered window around t
                    start = max(0, t - self.window_size // 2)
                    end = min(time_steps, start + self.window_size)
                    mask_indices = list(range(start, end))
                    perturbed_X = mask_timesteps(X, mask_indices, self.mask_value)

                elif self.mode == "noise":
                    perturbed_X = perturb_sequence_with_noise(X, timesteps=[t], noise_level=self.noise_level)

                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

                perturbed_outputs = self._get_output(perturbed_X)

                delta = base_outputs - perturbed_outputs
                if self.use_absolute:
                    delta = delta.abs()

                attributions[:, t] += delta

        attributions /= self.nsamples

        if self.clip_negative:
            attributions = torch.clamp(attributions, min=0.0)

        attributions = self._smooth_attributions(attributions)
        return self._format_output(attributions)
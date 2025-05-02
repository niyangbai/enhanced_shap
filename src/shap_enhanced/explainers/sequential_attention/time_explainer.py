"""Time Explainer for sequential models."""

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.perturbation import mask_timesteps, mask_time_window, perturb_sequence_with_noise
from shap_enhanced.algorithms.sampling import sample_timesteps

class TimeExplainer(BaseExplainer):
    """Time Explainer for sequential models.

    Perturbs specific timesteps and measures impact on model output:

    .. math::

        \\text{Attribution}_t = f(x) - f(x_{\\setminus t})
    """

    def __init__(self, model: Any, mode: str = "mask", mask_value: float = 0.0,
                 noise_level: float = 0.1, window_size: int = 1, nsamples: int = 10,
                 strategy: str = "sequential", target_index: Optional[int] = 0) -> None:
        """Initialize the Time Explainer.

        :param Any model: A sequential model.
        :param str mode: 'mask' or 'noise'.
        :param float mask_value: Value for masking.
        :param float noise_level: Std of noise added.
        :param int window_size: Size of masking window.
        :param int nsamples: Number of perturbations per sample.
        :param str strategy: 'sequential' (perturb t one-by-one) or 'random' (sampled timesteps).
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.mode = mode
        self.mask_value = mask_value
        self.noise_level = noise_level
        self.window_size = window_size
        self.nsamples = nsamples
        self.strategy = strategy
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate feature attributions over time.

        :param Any X: Input tensor [batch, time, features].
        :return Any: Time importance (torch.Tensor).
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for TimeExplainer.")

        batch_size, time_steps, _ = X.shape
        attributions = torch.zeros((batch_size, time_steps), device=X.device)

        base_outputs = self.model(X)

        if base_outputs.ndim > 1 and base_outputs.shape[1] > 1 and self.target_index is not None:
            base_outputs = base_outputs[:, self.target_index]
        elif base_outputs.ndim > 1 and base_outputs.shape[1] > 1:
            base_outputs = base_outputs[:, 0]

        for _ in range(self.nsamples):
            if self.strategy == "sequential":
                timesteps_to_perturb = list(range(time_steps))
            elif self.strategy == "random":
                timesteps_to_perturb = sample_timesteps(time_steps, nsamples=time_steps)
            else:
                raise ValueError(f"Unsupported perturbation strategy: {self.strategy}")

            for t in timesteps_to_perturb:
                if self.mode == "mask":
                    if self.window_size > 1:
                        perturbed_X = mask_time_window(X, window_size=self.window_size, mask_value=self.mask_value)
                    else:
                        perturbed_X = mask_timesteps(X, [t], mask_value=self.mask_value)
                elif self.mode == "noise":
                    perturbed_X = perturb_sequence_with_noise(X, noise_level=self.noise_level)
                else:
                    raise ValueError(f"Unsupported mode: {self.mode}")

                perturbed_outputs = self.model(perturbed_X)

                if perturbed_outputs.ndim > 1 and perturbed_outputs.shape[1] > 1 and self.target_index is not None:
                    perturbed_outputs = perturbed_outputs[:, self.target_index]
                elif perturbed_outputs.ndim > 1 and perturbed_outputs.shape[1] > 1:
                    perturbed_outputs = perturbed_outputs[:, 0]

                delta = (base_outputs - perturbed_outputs)
                attributions[:, t] += delta

        attributions /= self.nsamples

        return self._format_output(attributions)
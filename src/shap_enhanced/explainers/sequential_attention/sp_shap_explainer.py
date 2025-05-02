"""Sequential Perturbation SHAP (SP-SHAP) Explainer."""

from typing import Any, Optional
import torch
import random
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_subsets
from shap_enhanced.algorithms.perturbation import mask_timesteps, mask_time_window
from shap_enhanced.algorithms.approximation import monte_carlo_expectation

class SPShapExplainer(BaseExplainer):
    """Sequential Perturbation SHAP (SP-SHAP) Explainer.

    Approximates Shapley values for timesteps by perturbing subsets or (time, feature) pairs.

    .. math::

        \\phi_t = \\mathbb{E}_{S \\subseteq T \\setminus \\{t\\}} \\left[ f(S \\cup \\{t\\}) - f(S) \\right]
    """

    def __init__(self, model: Any, mask_value: float = 0.0, nsamples: int = 100,
                 window_size: int = 1, random_seed: Optional[int] = None,
                 target_index: Optional[int] = 0, mode: str = "default") -> None:
        """Initialize the SP-SHAP Explainer.

        :param Any model: Time series model.
        :param float mask_value: Value to mask timesteps.
        :param int nsamples: Number of subset samples.
        :param int window_size: Size of masked time window.
        :param Optional[int] random_seed: Random seed for reproducibility.
        :param Optional[int] target_index: Output index to explain.
        :param str mode: 'default' for full SP-SHAP, 'simple' for (time, feature) perturbation.
        """
        super().__init__(model)
        self.mask_value = mask_value
        self.nsamples = nsamples
        self.window_size = window_size
        self.random_seed = random_seed
        self.target_index = target_index
        self.mode = mode.lower()

        if self.mode not in ["default", "simple"]:
            raise ValueError(f"Invalid mode {self.mode}. Choose 'default' or 'simple'.")
        
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

    def explain(self, X: Any) -> Any:
        """Generate SHAP values using sequential perturbations.

        :param Any X: Input data (torch.Tensor) [batch, time, features].
        :return Any: SHAP attributions (torch.Tensor) [batch, time].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for SPShapExplainer.")

        batch_size, time_steps, _ = X.shape
        attributions = torch.zeros((batch_size, time_steps), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]  # (1, time, features)
            baseline_output = self._predict(x)

            if self.mode == "simple":
                contribs = self._simple_shap(x, time_steps)
            else:
                contribs = self._default_shap(x, time_steps)

            attributions[i] = contribs

        return self._format_output(attributions)

    def _simple_shap(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Simple (time, feature) replacement SHAP approximation.

        :param torch.Tensor x: Single input (1, time, features).
        :return torch.Tensor: SHAP values (time, features).
        """
        shap_values = torch.zeros((time_steps, x.shape[2]), device=x.device)

        try:
            pred_x = self._model_forward(x)
        except Exception as e:
            raise Exception(f"Model prediction failed on x: {e}")

        for t in range(time_steps):
            for f in range(x.shape[2]):
                modified_x = x.clone()
                random_indices = torch.randint(0, self.background.shape[0], (self.nsamples,), device=x.device)
                modified_x[:, t, f] = self.background[random_indices, t, f]

                preds_modified = self._model_forward(modified_x)

                mean_pred_modified = preds_modified.mean()
                shap_values[t, f] = pred_x - mean_pred_modified

        return shap_values

    def _default_shap(self, x: torch.Tensor, time_steps: int) -> torch.Tensor:
        """Default SP-SHAP sequential perturbation using subsets.

        :param torch.Tensor x: Single input (1, time, features).
        :return torch.Tensor: SHAP values (time, features).
        """
        subsets = sample_subsets(time_steps, self.nsamples)
        contribs = torch.zeros((time_steps,), device=x.device)

        baseline_output = self._predict(x)

        for subset in subsets:
            S = list(subset)
            x_S = self._mask_timesteps_except(x, S)
            output_S = self._predict(x_S)

            for t in range(time_steps):
                if t in S:
                    continue

                S_with_t = S + [t]
                x_St = self._mask_timesteps_except(x, S_with_t)
                output_St = self._predict(x_St)

                marginal_contribution = (output_St - output_S).squeeze(0)
                contribs[t] += marginal_contribution

        contribs /= self.nsamples
        return contribs

    def _predict(self, x: torch.Tensor) -> torch.Tensor:
        """Run model and select correct output index."""
        output = self.model(x)
        if output.ndim > 1 and output.shape[1] > 1 and self.target_index is not None:
            output = output[:, self.target_index]
        elif output.ndim > 1 and output.shape[1] > 1:
            output = output[:, 0]
        return output

    def _mask_timesteps_except(self, x: torch.Tensor, keep_indices: list) -> torch.Tensor:
        """Mask all timesteps except specified ones."""
        time_steps = x.shape[1]
        mask = torch.zeros((time_steps,), device=x.device)
        mask[keep_indices] = 1
        mask = mask.unsqueeze(0).unsqueeze(-1)  # (1, time, 1)

        x_masked = x.clone()
        x_masked = x_masked * mask + (1 - mask) * self.mask_value

        return x_masked
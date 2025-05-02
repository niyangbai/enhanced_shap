""" SHAP Interaction Explainer"""

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.sampling import sample_subsets

class SHAPInteractionExplainer(BaseExplainer):
    """SHAP Interaction Explainer for second-order feature interactions.

    Computes SHAP interaction values:

    .. math::

        \\phi_{i,j} = \\frac{1}{2} \\left( f(S \\cup \\{i,j\\}) - f(S \\cup \\{i\\}) - f(S \\cup \\{j\\}) + f(S) \\right)
    """

    def __init__(self, model: Any, nsamples: int = 100, mask_value: float = 0.0,
                 target_index: Optional[int] = 0) -> None:
        """Initialize the SHAP Interaction Explainer.

        :param Any model: Black-box model.
        :param int nsamples: Number of random subsets.
        :param float mask_value: Value for masking.
        :param Optional[int] target_index: Output index to explain.
        """
        super().__init__(model)
        self.nsamples = nsamples
        self.mask_value = mask_value
        self.target_index = target_index

    def explain(self, X: Any) -> Any:
        """Generate SHAP interaction matrices for each input.

        :param Any X: Input tensor (torch.Tensor) [batch, features].
        :return Any: Interaction matrices (torch.Tensor) [batch, features, features].
        """
        if not isinstance(X, torch.Tensor):
            raise ValueError("Input X must be a torch.Tensor for SHAPInteractionExplainer.")

        batch_size, num_features = X.shape
        interactions = torch.zeros((batch_size, num_features, num_features), device=X.device)

        for i in range(batch_size):
            x = X[i:i+1]
            interaction_matrix = torch.zeros((num_features, num_features), device=X.device)

            subsets = sample_subsets(num_features, self.nsamples)

            for S in subsets:
                S = list(S)

                # Build masked versions
                inputs_to_eval = []
                masks = []

                # S
                mask_S = self._build_mask(S, num_features)
                inputs_to_eval.append(self._apply_mask(x, mask_S))
                masks.append(("S", None, None))

                for i1 in range(num_features):
                    if i1 in S:
                        continue
                    # S + i
                    mask_Si = self._build_mask(S + [i1], num_features)
                    inputs_to_eval.append(self._apply_mask(x, mask_Si))
                    masks.append(("S+i", i1, None))

                    for i2 in range(i1, num_features):
                        if i2 in S or i2 == i1:
                            continue
                        # S + j
                        mask_Sj = self._build_mask(S + [i2], num_features)
                        inputs_to_eval.append(self._apply_mask(x, mask_Sj))
                        masks.append(("S+j", i2, None))

                        # S + i + j
                        mask_Sij = self._build_mask(S + [i1, i2], num_features)
                        inputs_to_eval.append(self._apply_mask(x, mask_Sij))
                        masks.append(("S+ij", i1, i2))

                # Batch evaluate
                inputs_to_eval = torch.cat(inputs_to_eval, dim=0)  # (N_eval, features)
                outputs = self.model(inputs_to_eval)

                if outputs.ndim > 1 and outputs.shape[1] > 1 and self.target_index is not None:
                    outputs = outputs[:, self.target_index]
                elif outputs.ndim > 1 and outputs.shape[1] > 1:
                    outputs = outputs[:, 0]

                # Map outputs
                output_map = {}
                for idx, (kind, i1, i2) in enumerate(masks):
                    output_map[(kind, i1, i2)] = outputs[idx]

                f_S = output_map[("S", None, None)]

                for i1 in range(num_features):
                    if i1 in S:
                        continue
                    f_Si = output_map.get(("S+i", i1, None), None)

                    for i2 in range(i1, num_features):
                        if i2 in S or i2 == i1:
                            continue

                        f_Sj = output_map.get(("S+j", i2, None), None)
                        f_Sij = output_map.get(("S+ij", i1, i2), None)

                        if f_Si is None or f_Sj is None or f_Sij is None:
                            continue

                        interaction = 0.5 * (f_Sij - f_Si - f_Sj + f_S).squeeze(0)
                        interaction_matrix[i1, i2] += interaction
                        interaction_matrix[i2, i1] += interaction  # symmetry

            interaction_matrix /= self.nsamples
            interactions[i] = interaction_matrix

        return self._format_output(interactions)

    def _build_mask(self, present_features: list, num_features: int) -> torch.Tensor:
        """Create a binary mask for features.

        :param list present_features: Indices to keep.
        :param int num_features: Total features.
        :return torch.Tensor: Binary mask.
        """
        mask = torch.zeros((1, num_features), device="cpu")
        mask[:, present_features] = 1
        return mask

    def _apply_mask(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Apply mask to input.

        :param torch.Tensor x: Input tensor (1, features).
        :param torch.Tensor mask: Binary mask (1, features).
        :return torch.Tensor: Masked input.
        """
        x_masked = x.clone()
        x_masked[mask == 0] = self.mask_value
        return x_masked
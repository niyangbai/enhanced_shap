"""Attention Explainer for models with attention mechanisms."""

from typing import Any, Optional
import torch
from shap_enhanced.explainers.base import BaseExplainer
from shap_enhanced.algorithms.attention import normalize_attention_weights, guided_attention_masking, compute_attention_flow

class AttentionExplainer(BaseExplainer):
    """Attention Explainer for models with attention mechanisms.

    Uses raw or aggregated attention weights to explain feature importance:

    .. math::

        \\text{Attribution}_t = \\text{AttentionScore}(t)
    """

    def __init__(self, model: Any, normalize: bool = True, norm_type: str = "l1",
                 use_attention_flow: bool = False, mask_threshold: Optional[float] = None) -> None:
        """Initialize the Attention Explainer.

        :param Any model: Model exposing attention weights.
        :param bool normalize: Whether to normalize attention weights.
        :param str norm_type: 'l1' or 'l2' normalization.
        :param bool use_attention_flow: Aggregate attention across layers.
        :param Optional[float] mask_threshold: Threshold to mask low-attention tokens.
        """
        super().__init__(model)
        self.normalize = normalize
        self.norm_type = norm_type
        self.use_attention_flow = use_attention_flow
        self.mask_threshold = mask_threshold

    def explain(self, X: Any, attention_maps: Any) -> Any:
        """Generate attributions based on attention scores.

        :param Any X: Input data (torch.Tensor) [batch, time, features].
        :param Any attention_maps: Attention weights (list of tensors or single tensor).
        :return Any: Attention-based feature attributions (torch.Tensor).
        """
        if isinstance(attention_maps, list):
            if self.use_attention_flow:
                attention = compute_attention_flow(attention_maps)
            else:
                attention = attention_maps[-1]  # Use last layer attention
        else:
            attention = attention_maps  # Assume single attention map

        if self.normalize:
            attention = normalize_attention_weights(attention, norm_type=self.norm_type)

        # Aggregate heads if needed
        if attention.ndim == 4:
            attention = attention.mean(dim=1)  # (batch, time, time)

        # Only use diagonal self-attention
        attention_scores = attention.diagonal(dim1=-2, dim2=-1)

        if self.mask_threshold is not None:
            X = guided_attention_masking(X, attention_scores, threshold=self.mask_threshold)

        return self._format_output(attention_scores)
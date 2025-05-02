"""Basic explainer that assigns random or constant feature attributions."""

from typing import Any
import numpy as np
from shap_enhanced.explainers.base import BaseExplainer

class BasicExplainer(BaseExplainer):
    """Basic explainer that assigns random or constant feature attributions.

    This explainer serves as a simple baseline without using model internals.
    """

    def __init__(self, model: Any, mode: str = "constant", constant_value: float = 0.1) -> None:
        """Initialize the basic explainer.

        :param Any model: Model object (not used actively in this explainer).
        :param str mode: Attribution mode, either 'constant' or 'random'.
        :param float constant_value: Value assigned for 'constant' mode.
        """
        super().__init__(model)
        self.mode = mode
        self.constant_value = constant_value

    def explain(self, X: Any) -> Any:
        """Generate basic feature attributions for the given input X.

        :param Any X: Input data to explain (numpy array or tensor).
        :return Any: Feature attributions with the same shape as input features.
        """
        if hasattr(X, "shape"):
            shape = X.shape
        else:
            raise ValueError("Input X must have a 'shape' attribute (numpy array, tensor, etc.).")

        if self.mode == "constant":
            attributions = np.full(shape, self.constant_value)
        elif self.mode == "random":
            attributions = np.random.rand(*shape)
        else:
            raise ValueError(f"Unsupported mode '{self.mode}'. Choose 'constant' or 'random'.")

        return self._format_output(attributions)
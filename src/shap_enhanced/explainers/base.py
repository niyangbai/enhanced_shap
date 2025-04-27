from abc import ABC, abstractmethod
from typing import Any

class BaseExplainer(ABC):
    """Abstract base class for all explainers.

    Every explainer must inherit from this class and implement
    the `explain` method to compute feature attributions.
    """

    def __init__(self, model: Any) -> None:
        """Initialize the explainer.

        :param Any model: A model object with a predict function.
        """
        self.model = model

    @abstractmethod
    def explain(self, X: Any) -> Any:
        """Compute feature attributions for the given input X.

        :param Any X: Input data to be explained.
        :return Any: Feature attributions corresponding to X.
        """
        pass

    def _validate_input(self, X: Any) -> None:
        """Validate the input X.

        This method checks if the input format is acceptable.
        Subclasses may override for specific input types.

        :param Any X: Input data to validate.
        """
        pass

    def _format_output(self, attributions: Any) -> Any:
        """Format the output attributions into a standardized format.

        :param Any attributions: Raw attributions.
        :return Any: Formatted attributions.
        """
        return attributions


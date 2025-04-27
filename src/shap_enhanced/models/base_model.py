from abc import ABC, abstractmethod
from typing import Any

class BaseModel(ABC):
    """Abstract base class for all models.

    Every model must inherit from this class and implement
    the `fit` and `predict` methods.
    """

    @abstractmethod
    def fit(self, X: Any, y: Any) -> None:
        """Train the model on input data X and target labels y.

        :param Any X: Input features for training.
        :param Any y: Target labels for training.
        :return None: This method does not return anything.
        """
        pass

    @abstractmethod
    def predict(self, X: Any) -> Any:
        """Predict outputs for the given input X.

        :param Any X: Input features for prediction.
        :return Any: Predicted outputs corresponding to X.
        """
        pass

    def evaluate(self, X: Any, y: Any) -> float:
        """Evaluate the model on input data X and labels y.

        Computes a default evaluation score such as accuracy
        or mean squared error, depending on the task.

        :param Any X: Input features for evaluation.
        :param Any y: True labels for evaluation.
        :return float: Evaluation score.
        """
        raise NotImplementedError("Evaluation method must be implemented in subclasses if needed.")

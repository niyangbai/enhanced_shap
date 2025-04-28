import torch

def trapezoidal_integrate(values: torch.Tensor) -> torch.Tensor:
    """Approximate integral using trapezoidal rule.

    :param torch.Tensor values: Tensor of shape (steps+1, ...)
    :return torch.Tensor: Integrated value across steps.
    """

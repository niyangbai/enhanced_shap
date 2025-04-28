import torch

def linear_interpolation(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    """Generate linearly interpolated tensors.

    :param torch.Tensor start: Start tensor.
    :param torch.Tensor end: End tensor.
    :param int steps: Number of steps.
    :return torch.Tensor: Interpolated tensors.
    """

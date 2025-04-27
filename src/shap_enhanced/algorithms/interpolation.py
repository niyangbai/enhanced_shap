import torch

def linear_interpolation(start: torch.Tensor, end: torch.Tensor, steps: int) -> torch.Tensor:
    """Generate linearly interpolated tensors.

    :param torch.Tensor start: Start tensor.
    :param torch.Tensor end: End tensor.
    :param int steps: Number of steps.
    :return torch.Tensor: Interpolated tensors.
    """
    alphas = torch.linspace(0, 1, steps + 1, device=start.device).view(-1, *([1] * (start.ndim)))
    return start + alphas * (end - start)

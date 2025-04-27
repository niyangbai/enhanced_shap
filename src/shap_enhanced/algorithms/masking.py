import torch
from typing import List

def apply_binary_mask(X: torch.Tensor, mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """Apply a binary mask to a tensor.

    :param torch.Tensor X: Input tensor.
    :param torch.Tensor mask: 1 or 0 mask tensor, same shape.
    :param float mask_value: Value used where mask is 0.
    :return torch.Tensor: Masked tensor.
    """
    return X * mask + (1 - mask) * mask_value

def apply_group_masking(X: torch.Tensor, groups: List[List[int]], group_mask: torch.Tensor, mask_value: float = 0.0) -> torch.Tensor:
    """Apply group-based masking.

    :param torch.Tensor X: Input tensor (batch, features).
    :param List[List[int]] groups: Feature groups.
    :param torch.Tensor group_mask: (batch, n_groups) binary mask.
    :param float mask_value: Mask value.
    :return torch.Tensor: Masked tensor.
    """
    X_masked = X.clone()
    batch_size = X.shape[0]
    for i in range(batch_size):
        for j, group in enumerate(groups):
            if group_mask[i, j] == 0:
                X_masked[i, group] = mask_value
    return X_masked
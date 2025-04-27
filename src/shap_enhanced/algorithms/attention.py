import torch

def guided_relu_backward_hook():
    """Return a backward hook for Guided Backpropagation."""
    def hook(module, grad_input, grad_output):
        return tuple(torch.clamp(g, min=0.0) for g in grad_input)
    return hook

def normalize_attention_weights(attention: torch.Tensor, norm_type: str = "l1") -> torch.Tensor:
    """Normalize attention weights across dimensions.

    :param torch.Tensor attention: Raw attention tensor (batch, heads, time, time)
    :param str norm_type: Normalization ('l1' or 'l2')
    :return torch.Tensor: Normalized attention tensor
    """
    if norm_type == "l1":
        norm = attention.sum(dim=-1, keepdim=True) + 1e-8
    elif norm_type == "l2":
        norm = torch.norm(attention, p=2, dim=-1, keepdim=True) + 1e-8
    else:
        raise ValueError(f"Unsupported norm type: {norm_type}")
    return attention / norm

def compute_attention_flow(attentions: list) -> torch.Tensor:
    """Compute cumulative attention flow across layers.

    :param list attentions: List of attention matrices per layer.
    :return torch.Tensor: Attention flow matrix.
    """
    result = attentions[0]
    for attn in attentions[1:]:
        result = torch.bmm(attn, result)
    return result

def guided_attention_masking(X: torch.Tensor, attention_map: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
    """Mask input based on attention scores.

    :param torch.Tensor X: Input tensor (batch, time, features)
    :param torch.Tensor attention_map: Attention score tensor (batch, time)
    :param float threshold: Attention threshold for masking
    :return torch.Tensor: Masked input
    """
    mask = (attention_map >= threshold).float().unsqueeze(-1)  # (batch, time, 1)
    return X * mask

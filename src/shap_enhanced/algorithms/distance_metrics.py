import torch

def euclidean_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Euclidean distance."""
    return torch.norm(x - y, dim=-1)

def cosine_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute cosine similarity."""
    x_norm = torch.nn.functional.normalize(x, dim=-1)
    y_norm = torch.nn.functional.normalize(y, dim=-1)
    return (x_norm * y_norm).sum(dim=-1)

def dynamic_time_warping_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Dynamic Time Warping (DTW) distance between sequences.

    :param torch.Tensor x: (time_x, features)
    :param torch.Tensor y: (time_y, features)
    :return torch.Tensor: DTW distance
    """
    time_x, _ = x.shape
    time_y, _ = y.shape

    dist = torch.zeros((time_x + 1, time_y + 1), device=x.device) + float('inf')
    dist[0, 0] = 0

    for i in range(1, time_x + 1):
        for j in range(1, time_y + 1):
            cost = torch.norm(x[i-1] - y[j-1])
            dist[i, j] = cost + torch.min(dist[i-1, j], dist[i, j-1], dist[i-1, j-1])

    return dist[time_x, time_y]

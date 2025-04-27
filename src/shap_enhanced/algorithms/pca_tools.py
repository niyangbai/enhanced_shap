import torch
from typing import Any
from sklearn.decomposition import PCA, IncrementalPCA

def fit_pca(X: torch.Tensor, n_components: int, batch_size: int = None) -> Any:
    """Fit PCA or IncrementalPCA depending on batch size.

    :param torch.Tensor X: Input tensor.
    :param int n_components: Components to keep.
    :param int batch_size: If set, uses IncrementalPCA.
    :return Any: Fitted PCA object.
    """
    X_np = X.detach().cpu().numpy()
    if batch_size is None:
        pca = PCA(n_components=n_components)
        pca.fit(X_np)
    else:
        pca = IncrementalPCA(n_components=n_components, batch_size=batch_size)
        pca.fit(X_np)
    return pca

def transform_pca(X: torch.Tensor, pca: Any) -> torch.Tensor:
    """Project tensor into PCA space."""
    X_np = X.detach().cpu().numpy()
    Z = pca.transform(X_np)
    return torch.tensor(Z, device=X.device, dtype=X.dtype)

def inverse_pca(Z: torch.Tensor, pca: Any) -> torch.Tensor:
    """Inverse transform PCA projection."""
    Z_np = Z.detach().cpu().numpy()
    X_recon = pca.inverse_transform(Z_np)
    return torch.tensor(X_recon, device=Z.device, dtype=Z.dtype)

def select_principal_components(pca: Any, variance_threshold: float = 0.95) -> int:
    """Select minimum number of principal components covering given variance.

    :param Any pca: Fitted PCA object.
    :param float variance_threshold: Minimum cumulative variance explained.
    :return int: Number of components to keep.
    """
    cumulative_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (cumulative_variance < variance_threshold).sum() + 1
    return n_components
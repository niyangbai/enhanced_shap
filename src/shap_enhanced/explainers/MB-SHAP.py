"""
Multi-Baseline SHAP (MB-SHAP) Explainer.

Computes SHAP values using multiple baseline references (e.g., random samples, mean, or k-means centroids), and averages results to reduce dependence on a single baseline.

Parameters
----------
base_explainer_class : class
    Any SHAP-style explainer (must implement .shap_values()).
model : Any
    Model to be explained.
background : np.ndarray
    Candidate baselines to select from (N, T, F).
n_baselines : int
    Number of baselines to use.
baseline_strategy : str
    'random', 'mean', 'kmeans', or 'user'.
user_baselines : np.ndarray or None
    Optional. If supplied, these baselines are used directly.
base_explainer_kwargs : dict
    Arguments for base_explainer_class.
"""
import numpy as np
from sklearn.cluster import KMeans

class MultiBaselineSHAPExplainer:
    def __init__(
        self,
        base_explainer_class,
        model,
        background,
        n_baselines=5,
        baseline_strategy="random",
        user_baselines=None,
        base_explainer_kwargs=None,
        random_seed=42
    ):
        self.base_explainer_class = base_explainer_class
        self.model = model
        self.background = np.asarray(background)
        self.n_baselines = n_baselines
        self.strategy = baseline_strategy
        self.user_baselines = user_baselines
        self.base_explainer_kwargs = base_explainer_kwargs or {}
        self.random_seed = random_seed

        self.baselines = self._select_baselines()

    def _select_baselines(self):
        np.random.seed(self.random_seed)
        if self.user_baselines is not None:
            return np.asarray(self.user_baselines)
        if self.strategy == "random":
            idx = np.random.choice(len(self.background), self.n_baselines, replace=False)
            return self.background[idx]
        elif self.strategy == "mean":
            mean_baseline = self.background.mean(axis=0, keepdims=True)
            return np.repeat(mean_baseline, self.n_baselines, axis=0)
        elif self.strategy == "kmeans":
            # Flatten each sequence for clustering
            bg_flat = self.background.reshape(self.background.shape[0], -1)
            km = KMeans(n_clusters=self.n_baselines, random_state=self.random_seed)
            km.fit(bg_flat)
            centroids = km.cluster_centers_.reshape(self.n_baselines, *self.background.shape[1:])
            return centroids
        else:
            raise ValueError(f"Unknown baseline strategy: {self.strategy}")

    def shap_values(self, X, **kwargs):
        all_attributions = []
        for i, baseline in enumerate(self.baselines):
            # Baseline must be in batch form for most SHAP explainers
            expl = self.base_explainer_class(
                model=self.model,
                background=baseline[None] if baseline.ndim == 2 else baseline,
                **self.base_explainer_kwargs
            )
            attr = expl.shap_values(X, **kwargs)
            all_attributions.append(attr)
        all_attributions = np.stack(all_attributions, axis=0)  # (n_baselines, ...)
        return np.mean(all_attributions, axis=0)

"""
Ensemble SHAP with Noise (ES-SHAP) Wrapper.

Runs any SHAP-style explainer multiple times with noise injected into the input or baseline,
and aggregates the resulting attributions to reduce variance and improve robustness.

Parameters
----------
explainer_class : class
    SHAP-style explainer class (must implement .shap_values()).
n_runs : int
    Number of noisy ensemble runs.
noise_level : float
    Standard deviation of Gaussian noise to inject (relative to input std).
noise_target : str
    'input' (default) or 'background': add noise to test input or explainer's background.
aggregation : str
    'mean' (default) or 'median': how to aggregate ensemble attributions.
explainer_kwargs : dict
    Arguments passed to explainer_class constructor.
"""

import numpy as np

class EnsembleSHAPWithNoise:
    def __init__(
        self,
        explainer_class,
        model,
        background=None,
        n_runs=10,
        noise_level=0.05,
        noise_target="input",
        aggregation="mean",
        explainer_kwargs=None,
    ):
        self.explainer_class = explainer_class
        self.model = model
        self.background = background
        self.n_runs = n_runs
        self.noise_level = noise_level
        self.noise_target = noise_target  # "input" or "background"
        self.aggregation = aggregation    # "mean" or "median"
        self.explainer_kwargs = explainer_kwargs or {}

    def _add_noise(self, X, level):
        # Gaussian noise, scale by input std
        std = np.std(X)
        return X + np.random.normal(0, level * std, size=X.shape)

    def shap_values(self, X, **kwargs):
        attributions = []
        for run in range(self.n_runs):
            # Prepare (optionally) noisy inputs/background
            if self.noise_target == "input":
                X_noisy = self._add_noise(np.asarray(X), self.noise_level)
                background = self.background
            elif self.noise_target == "background" and self.background is not None:
                background_noisy = self._add_noise(np.asarray(self.background), self.noise_level)
                X_noisy = X
                background = background_noisy
            else:
                X_noisy = X
                background = self.background

            # Instantiate a new explainer each run
            expl = self.explainer_class(
                model=self.model,
                background=background,
                **self.explainer_kwargs
            )
            attr = expl.shap_values(X_noisy, **kwargs)
            attributions.append(attr)

        attributions = np.stack(attributions, axis=0)
        if self.aggregation == "mean":
            agg_attr = np.mean(attributions, axis=0)
        elif self.aggregation == "median":
            agg_attr = np.median(attributions, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation}")
        return agg_attr

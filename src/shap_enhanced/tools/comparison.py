import numpy as np
from scipy.stats import pearsonr

class Comparison:
    def __init__(self, ground_truth, shap_models):
        """
        Args:
            ground_truth (np.ndarray): SHAP ground truth values (seq_len, n_features)
            shap_models (dict): Dict of {explainer_name: attribution array}
        """
        self.ground_truth = ground_truth
        self.shap_models = shap_models
        self.results = {}
        self.pearson_results = {}

    def calculate_kpis(self):
        """Calculate MSE and Pearson for each explainer."""
        for name, arr in self.shap_models.items():
            mse = np.mean((arr - self.ground_truth) ** 2)
            gt_flat = self.ground_truth.flatten()
            arr_flat = arr.flatten()
            try:
                pearson, _ = pearsonr(gt_flat, arr_flat)
            except Exception:
                pearson = np.nan
            self.results[name] = mse
            self.pearson_results[name] = pearson
        return self.results, self.pearson_results

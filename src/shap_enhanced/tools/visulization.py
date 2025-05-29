import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_mse_pearson(results, pearson_results, title_prefix="SHAP Explainers vs Monte Carlo"):
    plt.figure(figsize=(8, 4))
    plt.subplot(1,2,1)
    plt.bar(list(results.keys()), list(results.values()))
    plt.ylabel("MSE")
    plt.title(f"{title_prefix} Ground Truth")
    plt.subplot(1,2,2)
    plt.bar(list(pearson_results.keys()), list(pearson_results.values()))
    plt.ylabel("Pearson Correlation")
    plt.title("Pearson Correlation (Explainer vs Ground Truth)")
    plt.tight_layout()
    plt.show()

def plot_3d_bars(shap_gt, shap_models, seq_len, n_features):
    import numpy as np
    fig = plt.figure(figsize=(12, 3 * len(shap_models)))
    for i, (name, arr) in enumerate(shap_models.items()):
        _x = np.arange(n_features)
        _y = np.arange(seq_len)
        xx, yy = np.meshgrid(_x, _y)
        xpos, ypos = xx.flatten(), yy.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.8

        # Ground Truth
        ax_gt = fig.add_subplot(len(shap_models), 2, 2 * i + 1, projection='3d')
        dz_gt = shap_gt.flatten()
        ax_gt.bar3d(xpos, ypos, zpos, dx, dy, dz_gt)
        ax_gt.set_title(f"{name}: Ground Truth")
        ax_gt.set_xlabel('Feature')
        ax_gt.set_ylabel('Time Step')

        # Explainer Output
        ax_ex = fig.add_subplot(len(shap_models), 2, 2 * i + 2, projection='3d')
        dz_ex = arr.flatten()
        ax_ex.bar3d(xpos, ypos, zpos, dx, dy, dz_ex)
        ax_ex.set_title(f"{name}: SHAP Explanation")
        ax_ex.set_xlabel('Feature')
        ax_ex.set_ylabel('Time Step')
    plt.tight_layout()
    plt.show()

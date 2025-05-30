import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

def plot_mse_pearson(
    results,
    pearson_results,
    save=None,
    bar_threshold=4
):
    """
    Plot MSE and Pearson correlation for SHAP explainers vs ground truth.

    Args:
        results (dict): {explainer: mse}
        pearson_results (dict): {explainer: pearson_corr}
        save (str or None): If not None, file path to save figure (pdf, png, etc).
        bar_threshold (int): Number of bars after which to switch to horizontal layout.
    """
    n_bars = max(len(results), len(pearson_results))
    orientation = "vertical"
    if n_bars > bar_threshold:
        orientation = "horizontal"

    plt.figure(figsize=(9, 6 if orientation == "horizontal" else 4))
    if orientation == "vertical":
        plt.subplot(1, 2, 1)
        plt.bar(list(results.keys()), list(results.values()))
        plt.ylabel("MSE")
        plt.title(f"MSE (Explainer vs Ground Truth)")
        plt.subplot(1, 2, 2)
        plt.bar(list(pearson_results.keys()), list(pearson_results.values()))
        plt.ylabel("Pearson Correlation")
        plt.title("Pearson Correlation (Explainer vs Ground Truth)")
    else:
        plt.subplot(1, 2, 1)
        plt.barh(list(results.keys()), list(results.values()))
        plt.xlabel("MSE")
        plt.title(f"MSE (Explainer vs Ground Truth)")
        plt.subplot(1, 2, 2)
        plt.barh(list(pearson_results.keys()), list(pearson_results.values()))
        plt.xlabel("Pearson Correlation")
        plt.title("Pearson Correlation (Explainer vs Ground Truth)")

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
        print(f"Figure saved to {save}")
    plt.show()


def plot_3d_bars(
    shap_gt,
    shap_models,
    seq_len,
    n_features,
    save=None,
    bar_alpha=0.85,
):
    """
    Plot ground truth and explainer SHAP values as paired 3D bar plots.

    Args:
        shap_gt (np.ndarray): Ground truth SHAP values, shape (seq_len, n_features)
        shap_models (dict): {name: shap values ndarray of shape (seq_len, n_features)}
        seq_len (int): Number of timesteps
        n_features (int): Number of features
        save (str or None): If not None, file path to save figure (pdf, png, etc).
        bar_alpha (float): Opacity of bars.
    """
    import numpy as np
    n_models = len(shap_models)
    fig = plt.figure(figsize=(12, 3 * n_models))

    for i, (name, arr) in enumerate(shap_models.items()):
        _x = np.arange(n_features)
        _y = np.arange(seq_len)
        xx, yy = np.meshgrid(_x, _y)
        xpos, ypos = xx.flatten(), yy.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.8

        # Ground Truth
        ax_gt = fig.add_subplot(n_models, 2, 2 * i + 1, projection='3d')
        dz_gt = shap_gt.flatten()
        ax_gt.bar3d(xpos, ypos, zpos, dx, dy, dz_gt, alpha=bar_alpha)
        ax_gt.set_title(f"{name}: Ground Truth")
        ax_gt.set_xlabel('Feature')
        ax_gt.set_ylabel('Time Step')
        ax_gt.set_zlabel('SHAP Value')

        # Explainer Output
        ax_ex = fig.add_subplot(n_models, 2, 2 * i + 2, projection='3d')
        dz_ex = arr.flatten()
        ax_ex.bar3d(xpos, ypos, zpos, dx, dy, dz_ex, alpha=bar_alpha)
        ax_ex.set_title(f"{name}: SHAP Explanation")
        ax_ex.set_xlabel('Feature')
        ax_ex.set_ylabel('Time Step')
        ax_ex.set_zlabel('SHAP Value')

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
        print(f"Figure saved to {save}")
    plt.show()


def plot_feature_comparison(
    shap_gt,
    shap_models,
    feature_names=None,
    save=None
):
    """
    Bar plot for feature attributions of all explainers vs ground truth (tabular/1D).

    Args:
        shap_gt (np.ndarray): Ground truth SHAP values, shape (n_features,)
        shap_models (dict): {name: shap values ndarray of shape (n_features,)}
        feature_names (list or None): Names for x-axis, optional.
        save (str or None): If not None, file path to save figure.
    """
    n_explainers = len(shap_models)
    n_features = len(shap_gt)
    features = np.arange(n_features)
    width = 0.35

    fig, axes = plt.subplots(n_explainers, 1, figsize=(8, 3 * n_explainers), sharex=True)
    if n_explainers == 1:
        axes = [axes]
    if feature_names is None:
        feature_names = [str(i) for i in range(n_features)]

    for ax, (name, vals) in zip(axes, shap_models.items()):
        ax.bar(features - width/2, shap_gt, width, label='Monte Carlo GT')
        ax.bar(features + width/2, vals, width, label=f'{name} SHAP')
        ax.set_title(f'{name} Explainer')
        ax.set_ylabel('Shapley value')
        ax.set_xticks(features)
        ax.set_xticklabels(feature_names)
        ax.legend()
    axes[-1].set_xlabel('Feature index')

    plt.tight_layout()
    if save:
        plt.savefig(save, bbox_inches="tight")
        print(f"Figure saved to {save}")
    plt.show()

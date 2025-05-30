import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

import matplotlib.pyplot as plt
import numpy as np

def plot_mse_pearson(
    results,
    pearson_results,
    save=None,
    bar_threshold=4,
    cmap="viridis",
    show=True,
):
    """
    Plot MSE and Pearson correlation for SHAP explainers vs ground truth.
    No colorbar, clean plot, minimal white margin when saving as pdf/png.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    n_bars = max(len(results), len(pearson_results))
    orientation = "vertical" if n_bars <= bar_threshold else "horizontal"

    mse_vals = np.array(list(results.values()))
    pearson_vals = np.array(list(pearson_results.values()))
    mse_names = list(results.keys())
    pearson_names = list(pearson_results.keys())

    vmax = max(np.abs(mse_vals).max(), np.abs(pearson_vals).max())
    vmin = min(np.abs(mse_vals).min(), np.abs(pearson_vals).min())
    norm = plt.Normalize(vmin, vmax)
    cmap_ = plt.get_cmap(cmap)

    figsize = (9, 4.5) if orientation == "vertical" else (11, 5.5)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    for ax, vals, names, title, ylabel in zip(
        axes,
        [mse_vals, pearson_vals],
        [mse_names, pearson_names],
        ["MSE (Explainer vs Ground Truth)", "Pearson Correlation (Explainer vs Ground Truth)"],
        ["MSE", "Pearson Correlation"]
    ):
        colors = cmap_(norm(np.abs(vals)))
        if orientation == "vertical":
            bars = ax.bar(names, vals, color=colors, edgecolor="#222", alpha=0.88, zorder=2)
            ax.set_ylabel(ylabel, fontsize=12)
            ax.set_xlabel("Explainer", fontsize=11)
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3g}',
                            xy=(bar.get_x() + bar.get_width() / 2, height),
                            xytext=(0, 4 if height >= 0 else -14),
                            textcoords="offset points",
                            ha='center', va='bottom' if height >= 0 else 'top',
                            fontsize=10, color='#222')
            ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=0)
        else:
            bars = ax.barh(names, vals, color=colors, edgecolor="#222", alpha=0.88, zorder=2)
            ax.set_xlabel(ylabel, fontsize=12)
            ax.set_ylabel("Explainer", fontsize=11)
            for bar in bars:
                width = bar.get_width()
                ax.annotate(f'{width:.3g}',
                            xy=(width, bar.get_y() + bar.get_height() / 2),
                            xytext=(4 if width >= 0 else -32, 0),
                            textcoords="offset points",
                            ha='left' if width >= 0 else 'right', va='center',
                            fontsize=10, color='#222')
            ax.grid(axis='x', linestyle='--', alpha=0.7, zorder=0)
        ax.set_title(title, fontsize=13, fontweight='bold', pad=6)
        ax.set_facecolor('#fafafa')
        ax.tick_params(axis='both', labelsize=9)

    plt.tight_layout(pad=1.2)

    # --- No colorbar block here! ---

    if save:
        plt.savefig(save, bbox_inches='tight', pad_inches=0.02, dpi=300)
        print(f"Figure saved to {save}")

    if show:
        plt.show()
    else:
        plt.close(fig)

def plot_3d_surface(
    shap_gt,
    shap_models,
    seq_len,
    n_features,
    save=None,
    cmap='viridis',
    vlim=None,
    show=True,
):
    """
    Plot ground truth and explainer SHAP values as paired 3D surface plots.
    Handles 1D/2D/3D input arrays for SHAP.
    """
    import matplotlib.pyplot as plt
    import numpy as np

    plt.style.use("seaborn-v0_8-whitegrid")
    n_models = len(shap_models)
    if vlim is None:
        vmax = max(np.abs(shap_gt).max(), max(np.abs(np.asarray(arr)).max() for arr in shap_models.values()))
        vlim = (-vmax, vmax)
    else:
        vmax = max(abs(vlim[0]), abs(vlim[1]))
        vlim = (-vmax, vmax)

    fig = plt.figure(figsize=(10, 3.8 * n_models))

    for i, (name, arr) in enumerate(shap_models.items()):
        arr = np.asarray(arr)
        gt = np.asarray(shap_gt)
        # If batched, squeeze out batch dim
        if arr.ndim == 3 and arr.shape[0] == 1:
            arr = arr[0]
        if gt.ndim == 3 and gt.shape[0] == 1:
            gt = gt[0]
        # If flattened
        if arr.ndim == 1:
            arr = arr.reshape(seq_len, n_features)
        if gt.ndim == 1:
            gt = gt.reshape(seq_len, n_features)

        _x = np.arange(n_features)
        _y = np.arange(seq_len)
        xx, yy = np.meshgrid(_x, _y)
        
        # 1. Ground Truth
        ax_gt = fig.add_subplot(n_models, 2, 2 * i + 1, projection='3d')
        surf_gt = ax_gt.plot_surface(xx, yy, gt, cmap=cmap, vmin=vlim[0], vmax=vlim[1], edgecolor='none', alpha=0.95, antialiased=True)
        ax_gt.set_title(f"{name} – Ground Truth", fontsize=14, fontweight='bold', pad=2)
        ax_gt.set_xlabel('Feature', fontsize=11, labelpad=6)
        ax_gt.set_ylabel('Time', fontsize=11, labelpad=6)
        ax_gt.set_zlabel('SHAP Value', fontsize=11, labelpad=8)
        ax_gt.view_init(elev=30, azim=130)
        ax_gt.set_facecolor('#fcfcfc')
        ax_gt.tick_params(axis='both', labelsize=9, pad=2)
        ax_gt.set_xticks(_x)
        ax_gt.set_yticks(_y)
        ax_gt.set_zlim(vlim[0]*1.1, vlim[1]*1.1)
        ax_gt.grid(False)

        # 2. Explainer Output
        ax_ex = fig.add_subplot(n_models, 2, 2 * i + 2, projection='3d')
        surf_ex = ax_ex.plot_surface(xx, yy, arr, cmap=cmap, vmin=vlim[0], vmax=vlim[1], edgecolor='none', alpha=0.95, antialiased=True)
        ax_ex.set_title(f"{name} – SHAP Explanation", fontsize=14, fontweight='bold', pad=2)
        ax_ex.set_xlabel('Feature', fontsize=11, labelpad=6)
        ax_ex.set_ylabel('Time', fontsize=11, labelpad=6)
        ax_ex.set_zlabel('SHAP Value', fontsize=11, labelpad=8)
        ax_ex.view_init(elev=30, azim=130)
        ax_ex.set_facecolor('#fcfcfc')
        ax_ex.tick_params(axis='both', labelsize=9, pad=2)
        ax_ex.set_xticks(_x)
        ax_ex.set_yticks(_y)
        ax_ex.set_zlim(vlim[0]*1.1, vlim[1]*1.1)
        ax_ex.grid(False)

    plt.subplots_adjust(wspace=0.18, hspace=0.18)
    plt.tight_layout(rect=[0, 0, 1, 1])

    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.01, dpi=300)
        print(f"Figure saved to {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_3d_bars(
    shap_gt,
    shap_models,
    seq_len,
    n_features,
    save=None,
    bar_alpha=0.88,
    bar_color='#3498db',  # Softer blue, can set e.g. "#43aa8b" for green
    show=True,
):
    """
    Plot ground truth and explainer SHAP values as paired 3D bar plots, visually appealing and publication ready.
    No colorbar, single color bars, minimal margin.
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    n_models = len(shap_models)
    vmax = max(np.abs(shap_gt).max(), max(np.abs(arr).max() for arr in shap_models.values()))
    vmin = -vmax

    fig = plt.figure(figsize=(11, 3.5 * n_models))
    for i, (name, arr) in enumerate(shap_models.items()):
        # Grid
        _x = np.arange(n_features)
        _y = np.arange(seq_len)
        xx, yy = np.meshgrid(_x, _y)
        xpos, ypos = xx.flatten(), yy.flatten()
        zpos = np.zeros_like(xpos)
        dx = dy = 0.75

        # 1. Ground Truth
        ax_gt = fig.add_subplot(n_models, 2, 2 * i + 1, projection='3d')
        dz_gt = shap_gt.flatten()
        # Use sign for color (optional): positive blue, negative orange/red
        bar_colors_gt = np.where(dz_gt >= 0, bar_color, "#f39c12")  # blue and orange
        ax_gt.bar3d(xpos, ypos, zpos, dx, dy, dz_gt, color=bar_colors_gt, edgecolor='k', alpha=bar_alpha, linewidth=0.2)
        ax_gt.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.08, zorder=0)
        ax_gt.set_title(f"{name} – Ground Truth", fontsize=14, fontweight='bold', pad=12)
        ax_gt.set_xlabel('Feature', fontsize=11, labelpad=6)
        ax_gt.set_ylabel('Time', fontsize=11, labelpad=6)
        ax_gt.set_zlabel('SHAP Value', fontsize=11, labelpad=8)
        ax_gt.view_init(elev=28, azim=120)
        ax_gt.grid(False)
        ax_gt.set_facecolor('#fcfcfc')
        ax_gt.tick_params(axis='both', labelsize=9, pad=2)
        ax_gt.set_xticks(_x)
        ax_gt.set_yticks(_y)
        # Tighter z axis
        ax_gt.set_zlim(vmin*1.1, vmax*1.1)

        # 2. Explainer Output
        ax_ex = fig.add_subplot(n_models, 2, 2 * i + 2, projection='3d')
        dz_ex = arr.flatten()
        bar_colors_ex = np.where(dz_ex >= 0, bar_color, "#f39c12")  # blue and orange
        ax_ex.bar3d(xpos, ypos, zpos, dx, dy, dz_ex, color=bar_colors_ex, edgecolor='k', alpha=bar_alpha, linewidth=0.2)
        ax_ex.plot_surface(xx, yy, np.zeros_like(xx), color='gray', alpha=0.08, zorder=0)
        ax_ex.set_title(f"{name} – SHAP Explanation", fontsize=14, fontweight='bold', pad=12)
        ax_ex.set_xlabel('Feature', fontsize=11, labelpad=6)
        ax_ex.set_ylabel('Time', fontsize=11, labelpad=6)
        ax_ex.set_zlabel('SHAP Value', fontsize=11, labelpad=8)
        ax_ex.view_init(elev=28, azim=120)
        ax_ex.grid(False)
        ax_ex.set_facecolor('#fcfcfc')
        ax_ex.tick_params(axis='both', labelsize=9, pad=2)
        ax_ex.set_xticks(_x)
        ax_ex.set_yticks(_y)
        ax_ex.set_zlim(vmin*1.1, vmax*1.1)

    plt.subplots_adjust(wspace=0.18, hspace=0.18)
    plt.tight_layout(rect=[0, 0, 1, 1])
    if save:
        plt.savefig(save, bbox_inches="tight", pad_inches=0.01, dpi=300)
        print(f"Figure saved to {save}")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_feature_comparison(
    shap_gt,
    shap_models,
    feature_names=None,
    save=None,
):
    """
    Bar plot for feature attributions of all explainers vs ground truth (tabular/1D).
    """
    plt.style.use("seaborn-v0_8-whitegrid")
    n_explainers = len(shap_models)
    n_features = len(shap_gt)
    features = np.arange(n_features)
    width = 0.35

    if feature_names is None:
        feature_names = [str(i) for i in range(n_features)]

    fig, axes = plt.subplots(n_explainers, 1, figsize=(10, 2.7 * n_explainers), sharex=True)
    if n_explainers == 1:
        axes = [axes]

    color_gt = "#4093c6"     # blue for GT
    color_exp = "#f99c2b"    # orange for Explainer

    for ax, (name, vals) in zip(axes, shap_models.items()):
        bars_gt = ax.bar(features - width/2, shap_gt, width, 
                        label='Monte Carlo GT', color=color_gt, edgecolor='#333', alpha=0.80)
        bars_exp = ax.bar(features + width/2, vals, width, 
                        label=f'{name} SHAP', color=color_exp, edgecolor='#333', alpha=0.80)

        ax.set_title(f"{name} Explainer", fontsize=13, fontweight='bold', pad=14)
        ax.set_ylabel('Shapley Value', fontsize=11)
        ax.set_xticks(features)
        ax.set_xticklabels(feature_names, fontsize=10)
        # Move legend outside plot for clarity, avoid overlap
        ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.20), ncol=2, fontsize=10, frameon=False)
        ax.grid(axis='y', linestyle='--', linewidth=0.8, alpha=0.7)
        ax.set_facecolor('#fafafa')
        ax.tick_params(axis='both', labelsize=9)

    axes[-1].set_xlabel('Feature Index', fontsize=11)

    plt.tight_layout(rect=[0, 0.08, 1, 1])  # Leave space for legend
    if save:
        plt.savefig(save, bbox_inches="tight", dpi=300)
        print(f"Figure saved to {save}")
    plt.show()

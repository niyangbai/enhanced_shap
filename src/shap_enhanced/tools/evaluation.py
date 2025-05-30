import numpy as np
import torch

def compute_shapley_gt_seq(model, x, baseline, nsamples=200, device="cpu"):
    """
    Estimate ground-truth Shapley values via Monte Carlo for a single sequence input.

    Args:
        model: Trained PyTorch model.
        x (np.ndarray): Input sample, shape (seq_len, n_features).
        baseline (np.ndarray): Baseline (reference) sample, shape (seq_len, n_features).
        nsamples (int): Number of Monte Carlo samples.
        device: Torch device.

    Returns:
        np.ndarray: Estimated Shapley values, shape (seq_len, n_features).
    """
    T, F = x.shape
    vals = np.zeros((T, F))
    model.eval()
    with torch.no_grad():
        for t in range(T):
            for f in range(F):
                diffs = []
                for _ in range(nsamples):
                    mask = np.random.rand(T, F) < 0.5
                    m_with = mask.copy(); m_with[t, f] = True
                    m_without = mask.copy(); m_without[t, f] = False

                    def apply_mask(m):
                        xm = baseline.copy()
                        xm[m] = x[m]
                        inp = torch.tensor(xm[None], dtype=torch.float32).to(device)
                        return model(inp).cpu().numpy().squeeze()

                    y0 = apply_mask(m_without)
                    y1 = apply_mask(m_with)
                    diffs.append(y1 - y0)
                vals[t, f] = np.mean(diffs)
    return vals

def compute_shapley_gt_tabular(model, x, baseline, nsamples=1000, device="cpu"):
    F = x.shape[0]
    shap_vals = np.zeros(F)
    model.eval()
    with torch.no_grad():
        for i in range(F):
            contribs = []
            for _ in range(nsamples):
                mask = np.random.rand(F) < 0.5
                mask_with = mask.copy(); mask_with[i] = True
                mask_without = mask.copy(); mask_without[i] = False
                def apply_mask(m):
                    x_mask = baseline.copy()
                    x_mask[m] = x[m]
                    inp = torch.tensor(x_mask[None], dtype=torch.float32).to(device)
                    return model(inp).cpu().numpy().squeeze()
                contribs.append(apply_mask(mask_with) - apply_mask(mask_without))
            shap_vals[i] = np.mean(contribs)
    return shap_vals
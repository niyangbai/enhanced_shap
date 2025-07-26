r"""
AttnSHAPExplainer: Attention-Guided SHAP with General Proxy Attention
=====================================================================

Theoretical Explanation
-----------------------

AttnSHAP is a feature attribution method that enhances the traditional SHAP framework by leveraging attention mechanisms
to guide the sampling of feature coalitions. This is especially effective for sequential or structured data,
where model-provided or proxy attention scores highlight important feature positions.

By biasing the coalition selection process with attention, AttnSHAP prioritizes the masking of less informative features
and isolates the contributions of more relevant ones more effectively than uniform sampling.

Key Concepts
^^^^^^^^^^^^

- **Attention-Guided Sampling**: When available, model attention weights (via `get_attention_weights`) are used
    to bias coalition sampling toward informative features.
- **Proxy Attention**: If direct attention is unavailable, attention scores can be approximated using:
    - **Gradient-based**: Magnitude of input gradients.
    - **Input-based**: Magnitude of input values.
    - **Perturbation-based**: Change in model output due to masking each individual feature.
- **Uniform Sampling**: Falls back to classical SHAP's uniform random sampling when attention is not used.
- **Additivity Normalization**: Attribution values are scaled such that their sum equals the model output difference
    between the original and fully-masked inputs.

Algorithm
---------

1. **Initialization**:
    - Takes a model, background dataset, a flag for using attention, a proxy attention strategy, and device context.

2. **Attention/Proxy Computation**:
    - For each input:
        - Retrieve model attention weights if available.
        - Otherwise, compute proxy attention based on the configured method.

3. **Coalition Sampling**:
    - For each feature:
        - Repeatedly sample a subset (coalition) of other features, with probability weighted by attention (if applicable).
        - Compute model output after masking the coalition.
        - Compute model output after masking the coalition plus the target feature.
        - Record the difference to estimate the marginal contribution.

4. **Normalization**:
    - Normalize feature attributions so that their sum matches the model output difference
        between the unmasked input and a fully-masked input baseline.
        
References
----------

- **Lundberg & Lee (2017), “A Unified Approach to Interpreting Model Predictions”**  
  [SHAP foundation]

- **Serrano & Smith (2019), “Is Attention Interpretable?”**  
  [Examines the interpretability and limitations of attention weights]

- **Jain & Wallace (2019), “Attention is not Explanation”**  
  [Argues that attention alone is not a reliable explanation mechanism]

- **Chefer, Gur, & Wolf (2021), “Transformer Interpretability Beyond Attention Visualization”**  
  [Shows advanced uses of attention and gradients for model interpretation]

- **Sundararajan et al. (2017), “Axiomatic Attribution for Deep Networks”**  
  [Introduces integrated gradients, a gradient-based attribution method relevant for proxy attention]

- **Janzing et al. (2020), "Explaining Classifiers by Removing Input Features"**  
  [Discusses alternative SHAP sampling strategies and implications for non-uniform sampling]
"""

import numpy as np
import torch

from shap_enhanced.base_explainer import BaseExplainer


class AttnSHAPExplainer(BaseExplainer):
    r"""
    Attention-Guided SHAP Explainer for structured/sequential data.

    This class implements an extension to the SHAP framework that leverages attention mechanisms
    (either native to the model or via proxy strategies) to guide the coalition sampling process,
    focusing attribution on informative feature regions.

    :param model: PyTorch model to be explained.
    :param background: Background dataset used for SHAP estimation.
    :param bool use_attention: If True, uses attention weights (or proxy) for guiding feature masking.
    :param str proxy_attention: Strategy to approximate attention when model does not provide it.
                                Options: "gradient", "input", "perturb".
    :param device: Computation device ('cuda' or 'cpu').
    """

    def __init__(
        self,
        model,
        background,
        use_attention=True,
        proxy_attention="gradient",  # 'gradient', 'input', 'perturb'
        device=None,
    ):
        super().__init__(model, background)
        self.use_attention = use_attention
        self.proxy_attention = proxy_attention
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def _get_attention_weights(self, x):
        r"""
        Computes attention weights for a single input.

        Uses native model attention if available. Otherwise, calculates proxy attention scores
        based on gradients, input magnitude, or perturbation sensitivity.

        .. math::
            \text{Gradient proxy: } \alpha_t = \frac{\sum_{f=1}^F |\frac{\partial y}{\partial x_{t,f}}|}{\sum_{t'=1}^T \sum_{f=1}^F |\frac{\partial y}{\partial x_{t',f}}| + \epsilon}

            \text{Input proxy: } \alpha_t = \frac{\sum_{f=1}^F |x_{t,f}|}{\sum_{t'=1}^T \sum_{f=1}^F |x_{t',f}| + \epsilon}

            \text{Perturb proxy: } \alpha_t = \frac{|y - y_{(-t)}|}{\sum_{t'=1}^T |y - y_{(-t')}| + \epsilon}

        :param x: Input array of shape (T, F)
        :return: Attention weights as a numpy array of shape (T,) or (T, F)
        :rtype: np.ndarray
        """
        # Try to use model's attention method if exists
        if hasattr(self.model, "get_attention_weights"):
            with torch.no_grad():
                x_in = torch.tensor(x[None], dtype=torch.float32, device=self.device)
                attn = self.model.get_attention_weights(x_in)
            attn = attn.squeeze().detach().cpu().numpy()
            # If (T, 1), squeeze
            if attn.ndim == 2 and attn.shape[1] == 1:
                attn = attn[:, 0]
            return attn
        # Else, use proxy method
        if self.proxy_attention == "gradient":
            x_tensor = torch.tensor(
                x[None], dtype=torch.float32, device=self.device, requires_grad=True
            )
            output = self.model(x_tensor)
            out_scalar = output.view(-1)[0]
            out_scalar.backward()
            attn = x_tensor.grad.abs().detach().cpu().numpy()[0]  # (T, F)
            attn_norm = attn / (attn.sum() + 1e-8)
            # Optionally, sum over features to get (T,)
            attn_time = attn_norm.sum(axis=-1)
            return attn_time
        elif self.proxy_attention == "input":
            attn = np.abs(x).sum(axis=-1)  # (T,)
            attn = attn / (attn.sum() + 1e-8)
            return attn
        elif self.proxy_attention == "perturb":
            base_pred = (
                self.model(
                    torch.tensor(x[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            T, F = x.shape
            attn = np.zeros(T)
            for t in range(T):
                x_masked = x.copy()
                x_masked[t, :] = 0
                pred = (
                    self.model(
                        torch.tensor(
                            x_masked[None], dtype=torch.float32, device=self.device
                        )
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .squeeze()
                )
                attn[t] = abs(base_pred - pred)
            attn = attn / (attn.sum() + 1e-8)
            return attn
        else:
            raise RuntimeError("No attention or proxy_attention available.")

    def shap_values(
        self,
        X,
        nsamples=100,
        coalition_size=3,
        check_additivity=True,
        random_seed=42,
        **kwargs,
    ):
        r"""
        Compute SHAP values using attention-guided or proxy-guided coalition sampling.

        For each feature at each time step, it estimates the marginal contribution by comparing
        model outputs when the feature is masked vs. when it is included in a masked coalition.
        Sampling is optionally biased using attention scores.

        The final attributions are normalized to satisfy SHAP's additivity constraint:

        .. math::
            \sum_{t=1}^T \sum_{f=1}^F \phi_{t,f} \approx f(x) - f(x_{masked})

        :param X: Input data of shape (B, T, F) or (T, F)
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of coalitions sampled per feature.
        :param int coalition_size: Number of features in each sampled coalition.
        :param bool check_additivity: Whether to print additivity check results.
        :param int random_seed: Seed for reproducible coalition sampling.
        :return: SHAP values of shape (T, F) for single input or (B, T, F) for batch.
        :rtype: np.ndarray
        """
        np.random.seed(random_seed)
        is_torch = hasattr(X, "detach")
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        single = len(X_in.shape) == 2
        if single:
            X_in = X_in[None, ...]
        B, T, F = X_in.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        for b in range(B):
            x_orig = X_in[b]
            attn = self._get_attention_weights(x_orig) if self.use_attention else None
            if attn is not None:
                attn_flat = attn.flatten() if attn.ndim == 1 else attn.sum(axis=1)
                attn_flat = attn_flat / (attn_flat.sum() + 1e-8)
            all_pos = [(t, f) for t in range(T) for f in range(F)]
            for t in range(T):
                for f in range(F):
                    mc = []
                    available = [idx for idx in all_pos if idx != (t, f)]
                    for _ in range(nsamples):
                        # Attention-guided sampling if available
                        if attn is not None:
                            attn_prob = np.array(
                                [
                                    attn[t_] if attn.ndim == 1 else attn[t_, f_]
                                    for (t_, f_) in available
                                ]
                            )
                            attn_prob = attn_prob / (attn_prob.sum() + 1e-8)
                            sel_idxs = np.random.choice(
                                len(available),
                                coalition_size,
                                replace=False,
                                p=attn_prob,
                            )
                        else:
                            sel_idxs = np.random.choice(
                                len(available), coalition_size, replace=False
                            )
                        mask_idxs = [available[i] for i in sel_idxs]
                        x_masked = x_orig.copy()
                        for tt, ff in mask_idxs:
                            x_masked[tt, ff] = 0
                        # Also mask (t, f)
                        x_masked_tf = x_masked.copy()
                        x_masked_tf[t, f] = 0
                        out_masked = (
                            self.model(
                                torch.tensor(
                                    x_masked[None],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        out_masked_tf = (
                            self.model(
                                torch.tensor(
                                    x_masked_tf[None],
                                    dtype=torch.float32,
                                    device=self.device,
                                )
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .squeeze()
                        )
                        mc.append(out_masked_tf - out_masked)
                    shap_vals[b, t, f] = np.mean(mc)
            # Additivity normalization
            orig_pred = (
                self.model(
                    torch.tensor(x_orig[None], dtype=torch.float32, device=self.device)
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            x_all_masked = np.zeros_like(x_orig)
            masked_pred = (
                self.model(
                    torch.tensor(
                        x_all_masked[None], dtype=torch.float32, device=self.device
                    )
                )
                .detach()
                .cpu()
                .numpy()
                .squeeze()
            )
            shap_sum = shap_vals[b].sum()
            model_diff = orig_pred - masked_pred
            if shap_sum != 0:
                shap_vals[b] *= model_diff / shap_sum
        shap_vals = shap_vals[0] if single else shap_vals
        if check_additivity:
            print(f"[AttnSHAP] sum(SHAP)={shap_vals.sum():.4f}")
        return shap_vals

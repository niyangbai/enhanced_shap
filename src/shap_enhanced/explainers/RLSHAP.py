"""
RL-SHAP: Reinforcement Learning SHAP Explainer
==============================================

Theoretical Explanation
-----------------------

RL-SHAP introduces a reinforcement learning–based approach to Shapley value estimation by learning  
a **masking policy** that selectively reveals feature–time pairs in an input sequence.  
Instead of enumerating random coalitions (as in classical SHAP), RL-SHAP trains a policy network  
to generate informative coalitions—those that most impact the model output when masked or revealed.

The policy is optimized using **policy gradient reinforcement learning**, where the reward  
is derived from the change in model prediction caused by a particular mask.  
By using the **Gumbel-Softmax trick**, RL-SHAP enables differentiable and efficient sampling  
of binary masks, making it feasible to train masking policies via gradient-based optimization.

Key Concepts
^^^^^^^^^^^^

- **Masking Policy Network**:  
    A neural network that learns to generate binary masks over the input features and time steps.  
    The masks decide which features to keep and which to replace with a baseline (e.g., mean).

- **Reinforcement Learning Objective**:  
    The policy network is trained to maximize the reward signal, defined as the absolute change  
    in model output caused by masking.

- **Gumbel-Softmax Sampling**:  
    Enables differentiable approximation of discrete binary masking, allowing gradient descent  
    to be used for training the policy network.

- **Mean Imputation**:  
    Masked features are imputed with their mean values, derived from a background dataset.

- **Attribution Estimation**:  
    After training, the policy is used to evaluate the marginal contribution of each feature  
    by comparing model predictions with and without the feature unmasked across various policy-generated masks.

Algorithm
---------

1. **Initialization**:
    - Accepts the model to be explained, background dataset for mean imputation, device context,  
        and policy network architecture or hyperparameters.

2. **Policy Training**:
    - For each training iteration:
        - Sample a batch of background inputs.
        - The policy network outputs logits for feature-wise masks.
        - Apply Gumbel-Softmax sampling to obtain soft (differentiable) or hard (binary) masks.
        - Construct masked inputs by replacing selected features with their background mean.
        - Compute model outputs on both original and masked inputs.
        - Calculate rewards as the absolute change in output.
        - Update the policy using policy gradient optimization to maximize expected reward.

3. **SHAP Value Estimation**:
    - For each input sample and each feature:
        - Sample multiple masks from the trained policy.
        - For each mask:
            - Evaluate model outputs with the feature masked vs. unmasked.
        - Estimate the SHAP value as the average difference in outputs due to unmasking the feature.

Use Case
--------

RL-SHAP is particularly beneficial when:
    - Exhaustive or random sampling of coalitions is infeasible due to input dimensionality.
    - The model is highly sensitive to feature interactions.
    - A learned masking strategy can offer better efficiency or accuracy than brute-force approaches.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shap_enhanced.base_explainer import BaseExplainer

class MaskingPolicy(nn.Module):
    r"""
    MaskingPolicy: Neural Mask Generator for Feature-Time Grids

    A simple fully connected neural network that produces logits for binary masks over a (T, F) input space.
    Designed to be used in RL-SHAP for learning masking policies via policy gradient methods.

    Architecture:
    - Input: Flattened (T, F) input sequence.
    - 2-layer MLP with ReLU activation.
    - Output reshaped to (B, T, F) for mask logits.

    :param int T: Number of time steps in the input.
    :param int F: Number of features per time step.
    :param int hidden_dim: Hidden layer size for MLP.
    """
    def __init__(self, T, F, hidden_dim=64):
        super().__init__()
        self.T, self.F = T, F
        self.fc1 = nn.Linear(T * F, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, T * F)

    def forward(self, x):
        r"""
        Forward pass to produce mask logits over each (t, f) entry.

        :param x: Input tensor of shape (B, T, F).
        :type x: torch.Tensor
        :return: Logits for masking, shape (B, T, F).
        :rtype: torch.Tensor
        """
        x_flat = x.view(x.shape[0], -1)
        logits = self.fc2(F.relu(self.fc1(x_flat)))  # (B, T*F)
        return logits.view(x.shape[0], self.T, self.F)  # (B, T, F)

class RLShapExplainer(BaseExplainer):
    r"""
    RLShapExplainer: Reinforcement Learning–based SHAP Explainer

    This explainer uses a policy network trained via reinforcement learning to 
    learn feature–time masking strategies that optimize attribution signal strength. 
    Instead of enumerating coalitions randomly, it learns where to mask for maximal 
    model impact and uses those masks to approximate SHAP values.

    .. math::
        \text{SHAP}(i) \approx \mathbb{E}_{\pi} \left[
            f(x \setminus i) - f(x)
        \right]

    :param model: The predictive model to be explained.
    :type model: torch.nn.Module
    :param background: Background dataset used for mean imputation.
    :type background: np.ndarray or torch.Tensor
    :param str device: Torch device, either 'cpu' or 'cuda'.
    :param int policy_hidden: Hidden layer size for the masking policy network.
    """

    def __init__(self, model, background, device=None, policy_hidden=64):
        super().__init__(model, background)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        T, F = background.shape[1:3]
        self.policy = MaskingPolicy(T, F, hidden_dim=policy_hidden).to(self.device)
        self.T, self.F = T, F
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def gumbel_sample(self, logits, tau=0.5):
        r"""
        Perform Gumbel-Softmax sampling over logits to generate differentiable binary-like masks.

        Adds Gumbel noise to logits and applies a sigmoid activation with temperature scaling
        to approximate binary sampling in a differentiable way.

        .. math::
            y = \sigma\left(\frac{\logits + G}{\tau}\right), \quad 
            G \sim \text{Gumbel}(0,1)

        :param logits: Raw logits over the (T, F) feature mask space.
        :type logits: torch.Tensor
        :param float tau: Temperature parameter controlling sharpness of output (lower = harder mask).
        :return: Differentiable soft mask tensor in [0, 1], same shape as logits.
        :rtype: torch.Tensor
        """
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return torch.sigmoid(y / tau)

    def train_policy(self, n_steps=500, batch_size=16, mask_frac=0.3):
        r"""
        Train the masking policy network using policy gradient optimization.

        The network is optimized to generate masks that maximize the absolute 
        change in the model's prediction when masking certain input features.

        .. note::
            Gumbel-Softmax is used to approximate discrete mask sampling for differentiability.

        :param int n_steps: Number of training iterations.
        :param int batch_size: Batch size sampled from the background at each step.
        :param float mask_frac: Fraction of features to mask in each sampled coalition.
        """
        print("[RL-SHAP] Training masking policy...")
        self.policy.train()
        background = torch.tensor(self.background, dtype=torch.float32).to(self.device)
        N = background.shape[0]
        for step in range(n_steps):
            idx = np.random.choice(N, batch_size, replace=True)
            x = background[idx]  # (B, T, F)
            logits = self.policy(x)
            masks = self.gumbel_sample(logits)  # (B, T, F), [0,1] soft mask

            # Select mask_frac of features: enforce average mask sum
            if mask_frac < 1.0:
                topk = int(mask_frac * self.T * self.F)
                masks_flat = masks.view(batch_size, -1)
                thresh = torch.topk(masks_flat, topk, dim=1)[0][:, -1].unsqueeze(1)
                hard_mask = (masks_flat >= thresh).float().view_as(masks)
            else:
                hard_mask = (masks > 0.5).float()

            # Masked input: replace masked positions with background mean
            x_masked = x.clone()
            mean_val = background.mean(dim=0)
            x_masked = hard_mask * x + (1 - hard_mask) * mean_val

            # Get model outputs
            y_orig = self._get_model_output(x)
            y_masked = self._get_model_output(x_masked)

            # Reward: absolute change in output (could be squared diff, etc.)
            reward = torch.abs(y_orig - y_masked)
            loss = -reward.mean()  # policy gradient: maximize reward

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if (step + 1) % 100 == 0:
                print(f"[RL-SHAP] Step {step+1}/{n_steps}, Avg Reward: {reward.mean().item():.4f}")

        print("[RL-SHAP] Policy training complete.")

    def _get_model_output(self, X):
        if isinstance(X, np.ndarray):
            X = torch.tensor(X, dtype=torch.float32, device=self.device)
        elif isinstance(X, torch.Tensor):
            X = X.to(self.device)
        out = self.model(X)
        return out.flatten() if out.ndim > 1 else out

    def shap_values(self, X, nsamples=50, mask_frac=0.3, tau=0.5, **kwargs):
        r"""
        Estimate SHAP values for input `X` using the trained masking policy.

        For each feature (t, f), multiple masks are sampled with the feature masked and unmasked. 
        The expected difference in model outputs estimates the marginal contribution of the feature.

        .. math::
            \phi_{t,f} = \mathbb{E}_{m \sim \pi} \left[
                f(x_{m \cup \{(t,f)\}}) - f(x_m)
            \right]

        :param X: Input to explain, shape (T, F) or (B, T, F).
        :type X: np.ndarray or torch.Tensor
        :param int nsamples: Number of mask samples to average over.
        :param float mask_frac: Fraction of features masked per sample.
        :param float tau: Gumbel-Softmax temperature.
        :param kwargs: Additional keyword arguments (not used).
        :return: Estimated SHAP values with same shape as input.
        :rtype: np.ndarray
        """
        self.policy.eval()
        is_torch = hasattr(X, 'detach')
        X_in = X.detach().cpu().numpy() if is_torch else np.asarray(X)
        shape = X_in.shape
        single = False
        if len(shape) == 2:  # (T, F)
            X_in = X_in[None, ...]
            single = True
        B, T, F = X_in.shape
        attributions = np.zeros((B, T, F), dtype=float)

        for b in range(B):
            x = torch.tensor(X_in[b][None], dtype=torch.float32, device=self.device)  # (1, T, F)
            for t in range(T):
                for f in range(F):
                    vals = []
                    for _ in range(nsamples):
                        # Sample a mask using learned policy
                        logits = self.policy(x)  # (1, T, F)
                        mask = self.gumbel_sample(logits, tau=tau)[0]  # (T, F)
                        # Mask (t, f) set to 0, others sampled by policy
                        mask_tf = mask.clone()
                        mask_tf[t, f] = 0.0
                        mean_val = torch.tensor(self.background.mean(axis=0), dtype=torch.float32, device=self.device)
                        x_masked = mask_tf * x[0] + (1 - mask_tf) * mean_val
                        # Attribution: output difference for unmasking (t, f)
                        out_C = self._get_model_output(x_masked[None])[0]
                        mask_tf[t, f] = 1.0  # unmask (t, f)
                        x_C_tf = mask_tf * x[0] + (1 - mask_tf) * mean_val
                        out_C_tf = self._get_model_output(x_C_tf[None])[0]
                        vals.append(out_C_tf.item() - out_C.item())
                    attributions[b, t, f] = np.mean(vals)
        return attributions[0] if single else attributions

"""
RL-SHAP: Reinforcement Learning SHAP Explainer

## Theoretical Explanation

RL-SHAP is a feature attribution method that learns a masking policy for selecting input feature–time subsets using a policy network trained with reinforcement learning (policy gradients). The policy network generates masks over the input, and the reward is the change in model output due to the selected mask. This approach enables the explainer to learn which subsets of features are most important for the model's prediction, rather than relying on random or exhaustive coalition sampling. RL-SHAP uses the Gumbel-Softmax trick for differentiable subset selection, allowing efficient training of the masking policy.

### Key Concepts

- **Masking Policy Network:** A neural network learns to generate masks over the input features/timesteps, selecting which features to keep or mask.
- **Reinforcement Learning:** The policy is trained using policy gradients, with the reward being the change in model output caused by masking.
- **Gumbel-Softmax Sampling:** Enables differentiable sampling of binary masks, making the policy network trainable via gradient descent.
- **Mean Imputation:** Masked features are replaced with the mean value from the background data.
- **Attribution Estimation:** After training, the policy is used to estimate the marginal contribution of each feature by measuring the effect of masking/unmasking it.

## Algorithm

1. **Initialization:**
    - Accepts a model, background data, device, and policy network parameters.
2. **Policy Training:**
    - For each batch of background samples:
        - The policy network generates mask logits.
        - Gumbel-Softmax is used to sample soft/hard masks.
        - Masked inputs are created by replacing masked features with the background mean.
        - The model is evaluated on both original and masked inputs.
        - The reward is the absolute change in model output.
        - The policy is updated via policy gradient to maximize the reward.
3. **SHAP Value Estimation:**
    - For each input and each feature:
        - Sample masks using the trained policy.
        - For each mask, compute the model output with and without the feature unmasked.
        - The average difference estimates the SHAP value for that feature.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from shap_enhanced.base_explainer import BaseExplainer

class MaskingPolicy(nn.Module):
    """
    Policy network for generating mask logits over (T, F).
    """
    def __init__(self, T, F, hidden_dim=64):
        super().__init__()
        self.T, self.F = T, F
        self.fc1 = nn.Linear(T * F, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, T * F)

    def forward(self, x):
        # x: (B, T, F)
        x_flat = x.view(x.shape[0], -1)
        logits = self.fc2(F.relu(self.fc1(x_flat)))  # (B, T*F)
        return logits.view(x.shape[0], self.T, self.F)  # (B, T, F)

class RLShapExplainer(BaseExplainer):
    """
    RL-SHAP: Masking Policy Gradient Explainer

    Parameters
    ----------
    model : Any
        Model to be explained.
    background : np.ndarray or torch.Tensor
        Background data for training.
    device : 'cpu' or 'cuda'
        Torch device.
    """

    def __init__(self, model, background, device=None, policy_hidden=64):
        super().__init__(model, background)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        T, F = background.shape[1:3]
        self.policy = MaskingPolicy(T, F, hidden_dim=policy_hidden).to(self.device)
        self.T, self.F = T, F
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=1e-3)

    def gumbel_sample(self, logits, tau=0.5):
        # logits: (B, T, F)
        gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
        y = logits + gumbel_noise
        return torch.sigmoid(y / tau)

    def train_policy(self, n_steps=500, batch_size=16, mask_frac=0.3):
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
        """
        Estimate SHAP values using learned masking policy.
        For each (t, f), sample maskings and measure expected output change.
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

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
"""


import numpy as np
import torch
from shap_enhanced.base_explainer import BaseExplainer
from shap_enhanced.algorithms.coalition_sampling import WeightedCoalitionSampler, RandomCoalitionSampler, create_all_positions
from shap_enhanced.algorithms.masking import ZeroMasker
from shap_enhanced.algorithms.model_evaluation import ModelEvaluator
from shap_enhanced.algorithms.normalization import AdditivityNormalizer
from shap_enhanced.algorithms.data_processing import process_inputs
from shap_enhanced.algorithms.attention import GradientAttention, InputMagnitudeAttention, PerturbationAttention

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
        device=None
    ):
        super().__init__(model, background)
        self.use_attention = use_attention
        self.proxy_attention = proxy_attention
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize common algorithm components
        self.masker = ZeroMasker()
        self.model_evaluator = ModelEvaluator(model, device)
        self.normalizer = AdditivityNormalizer(model, device)
        
        # Initialize attention computer based on proxy type
        if proxy_attention == "gradient":
            self.attention_computer = GradientAttention(device)
        elif proxy_attention == "input":
            self.attention_computer = InputMagnitudeAttention(aggregation="sum")
        elif proxy_attention == "perturb":
            self.attention_computer = PerturbationAttention(device=device)
        else:
            self.attention_computer = None

    def _get_attention_weights(self, x):
        r"""
        Computes attention weights for a single input.

        Uses native model attention if available. Otherwise, calculates proxy attention scores
        using the common attention algorithms.

        :param x: Input array of shape (T, F)
        :return: Attention weights as a numpy array of shape (T, F)
        :rtype: np.ndarray
        """
        # Try to use model's attention method if exists
        if hasattr(self.model, "get_attention_weights"):
            with torch.no_grad():
                x_in = torch.tensor(x[None], dtype=torch.float32, device=self.device)
                attn = self.model.get_attention_weights(x_in)
            attn = attn.squeeze().detach().cpu().numpy()
            # Ensure shape is (T, F)
            if attn.ndim == 1:  # (T,) -> (T, 1) -> (T, F)
                attn = np.broadcast_to(attn[:, None], x.shape)
            elif attn.ndim == 2 and attn.shape[1] == 1:  # (T, 1) -> (T, F)
                attn = np.broadcast_to(attn, x.shape)
            return attn
        
        # Use proxy method via common attention algorithms
        if self.attention_computer is not None:
            return self.attention_computer.compute_attention(x, self.model)
        else:
            raise RuntimeError("No attention or proxy_attention available.")

    def shap_values(
        self,
        X,
        nsamples=100,
        coalition_size=3,
        check_additivity=True,
        random_seed=42,
        **kwargs
    ):
        r"""
        Compute SHAP values using attention-guided or proxy-guided coalition sampling.

        For each feature at each time step, it estimates the marginal contribution by comparing
        model outputs when the feature is masked vs. when it is included in a masked coalition.
        Sampling is optionally biased using attention scores.

        :param X: Input data of shape (B, T, F) or (T, F)
        :param nsamples: Number of coalitions sampled per feature.
        :param coalition_size: Number of features in each sampled coalition.
        :param check_additivity: Whether to print additivity check results.
        :param random_seed: Seed for reproducible coalition sampling.
        :return: SHAP values of shape (T, F) for single input or (B, T, F) for batch.
        """
        np.random.seed(random_seed)
        
        # Process inputs using common algorithm
        X_processed, is_single, _ = process_inputs(X)
        B, T, F = X_processed.shape
        shap_vals = np.zeros((B, T, F), dtype=float)
        
        all_positions = create_all_positions(T, F)
        
        for b in range(B):
            x_orig = X_processed[b]
            
            # Get attention weights if enabled
            attn = self._get_attention_weights(x_orig) if self.use_attention else None
            
            # Setup coalition sampler based on attention availability
            if attn is not None:
                coalition_sampler = WeightedCoalitionSampler(attn)
            else:
                coalition_sampler = RandomCoalitionSampler()
            
            # Compute SHAP values for each position
            for t in range(T):
                for f in range(F):
                    marginal_contributions = []
                    target_position = (t, f)
                    
                    for _ in range(nsamples):
                        # Sample coalition excluding target feature
                        coalition = coalition_sampler.sample_coalition(
                            all_positions, exclude=target_position,
                            size_range=(coalition_size, coalition_size)
                        )
                        
                        # Evaluate model with coalition
                        x_coalition = self.masker.mask_features(x_orig, coalition)
                        pred_coalition = self.model_evaluator.evaluate_single(x_coalition)
                        
                        # Evaluate model with coalition + target feature
                        coalition_plus_target = coalition + [target_position]
                        x_coalition_plus = self.masker.mask_features(x_orig, coalition_plus_target)
                        pred_coalition_plus = self.model_evaluator.evaluate_single(x_coalition_plus)
                        
                        # Marginal contribution
                        contribution = pred_coalition_plus - pred_coalition
                        marginal_contributions.append(contribution)
                    
                    shap_vals[b, t, f] = np.mean(marginal_contributions)
            
            # Apply additivity normalization using common algorithm
            fully_masked = self.masker.mask_features(x_orig, all_positions)
            shap_vals[b] = self.normalizer.normalize_additive(
                shap_vals[b], x_orig, fully_masked
            )
        
        # Return in original format
        result = shap_vals[0] if is_single else shap_vals
        
        if check_additivity:
            print(f"[AttnSHAP] sum(SHAP)={result.sum():.4f}")
        
        return result

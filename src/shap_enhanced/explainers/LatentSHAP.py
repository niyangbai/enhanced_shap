"""
Latent SHAP with Autoencoding for Structured Time Series
=========================================================

Theoretical Explanation
-----------------------

This module implements a **Latent SHAP Explainer** that applies SHAP to the latent space of an autoencoder,  
specifically designed for sequential or multivariate time-series data.  
By operating in a compressed latent representation, the explainer benefits from reduced dimensionality  
and improved noise tolerance, while maintaining interpretability through a decoder Jacobian projection.

The workflow is based on encoding the input data, applying SHAP in latent space (e.g., via `GradientExplainer`),  
and then projecting the latent attributions back into the input space using the decoder's Jacobian.  
This approach is especially useful for high-dimensional or noisy input domains where direct SHAP attribution  
is unstable or computationally expensive.

Key Components
^^^^^^^^^^^^^^

- **Conv1dEncoder / Conv1dDecoder**: Lightweight 1D convolutional autoencoder architecture for time-series encoding/decoding.
- **LatentModelWrapper**: Wraps a downstream model and decoder into a latent-to-output computation graph for SHAP.
- **LatentSHAPExplainer**:
  - Runs SHAP (e.g., `GradientExplainer`) on the latent representation of input data.
  - Projects latent attributions to input space using decoder Jacobians (optionally path-integrated).
- **make_shap_explainer**: Utility that handles compatibility across SHAP versions and explainer signatures.
- **train_conv1d_autoencoder**: Simple MSE-based training function for the autoencoder.

Algorithm
---------

1. **Autoencoder Construction and Training**:
   - Build a `Conv1dEncoder` and `Conv1dDecoder` for the desired input shape.
   - Train them using standard MSE reconstruction loss.

2. **Latent Attribution via SHAP**:
   - Encode the input and background into latent space.
   - Run a base SHAP explainer (e.g., `GradientExplainer`) on the latent representations.
   - Estimate input attributions by projecting SHAP values using the decoder's Jacobian:
   
     .. math::
         \phi_{\text{input}} = J_{\text{decoder}}(z) \cdot \phi_{\text{latent}}

3. **Jacobian Projection Options**:
   - Compute Jacobian at a single point (`z`) or along a path from baseline to `z` for smoother attribution.

4. **Normalization and Output**:
   - Return attributions in the shape of the original input (e.g., `[T, F]` for time-series).

Use Case
--------

This setup is ideal when:
- Inputs are high-dimensional sequences or grids.
- The model is highly non-linear or unstable with direct input perturbations.
- Interpretability is needed at both latent and original input levels.

Example
-------

.. code-block:: python

    model = DummyLSTM(F)
    encoder = Conv1dEncoder(F, T, latent_dim=4)
    decoder = Conv1dDecoder(latent_dim=4, seq_len=T, output_features=F)

    latent_expl = LatentSHAPExplainer(
        model=model,
        encoder=encoder,
        decoder=decoder,
        base_explainer_class=shap.GradientExplainer,
        background=X[:32]
    )

    attr = latent_expl.shap_values(X[0])
"""


import numpy as np
import torch
import torch.nn as nn
import shap
import inspect
from shap_enhanced.base_explainer import BaseExplainer

class Conv1dEncoder(nn.Module):
    r"""
    Conv1dEncoder: Temporal Convolutional Encoder for Time-Series

    A lightweight 1D convolutional encoder that transforms multivariate time-series input 
    of shape (B, T, F) into a fixed-dimensional latent representation.

    Architecture:
    - 3 stacked Conv1d layers with ReLU activations.
    - Output is flattened and linearly projected to latent_dim.

    :param int input_features: Number of input features (F).
    :param int seq_len: Length of the input sequence (T).
    :param int latent_dim: Dimension of the latent space.
    """
    def __init__(self, input_features, seq_len, latent_dim):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(32 * seq_len, latent_dim)
        self.seq_len = seq_len
        self.input_features = input_features

    def forward(self, x):
        r"""
        Encode input into latent space.

        :param x: Input tensor of shape (B, T, F).
        :type x: torch.Tensor
        :return: Latent representation of shape (B, latent_dim).
        :rtype: torch.Tensor
        """
        x = x.permute(0, 2, 1)  # (B, F, T)
        x = self.conv(x)
        x = x.permute(0, 2, 1)  # (B, T, C)
        x = self.flatten(x)
        z = self.fc(x)
        return z

class Conv1dDecoder(nn.Module):
    r"""
    Conv1dDecoder: Temporal Convolutional Decoder for Time-Series

    A lightweight decoder that reconstructs time-series input from a latent representation.
    Mirrors the structure of Conv1dEncoder using Conv1d layers.

    :param int latent_dim: Dimensionality of the latent input.
    :param int seq_len: Desired output sequence length (T).
    :param int output_features: Number of output features (F).
    """
    def __init__(self, latent_dim, seq_len, output_features):
        super().__init__()
        self.fc = nn.Linear(latent_dim, 32 * seq_len)
        self.deconv = nn.Sequential(
            nn.Conv1d(32, 64, 3, padding=1), nn.ReLU(),
            nn.Conv1d(64, 32, 3, padding=1), nn.ReLU(),
            nn.Conv1d(32, output_features, 3, padding=1),
        )
        self.seq_len = seq_len
        self.output_features = output_features

    def forward(self, z):
        r"""
        Decode latent representation into reconstructed time-series.

        :param z: Latent input of shape (B, latent_dim).
        :type z: torch.Tensor
        :return: Reconstructed input of shape (B, T, F).
        :rtype: torch.Tensor
        """
        x = self.fc(z)
        x = x.view(-1, 32, self.seq_len)
        x = self.deconv(x)
        x = x.permute(0, 2, 1)
        return x  # (B, T, F)

class LatentModelWrapper(nn.Module):
    r"""
    LatentModelWrapper: Compose Decoder and Model for SHAP

    Utility wrapper that chains decoder and model together. Accepts latent input,
    decodes it to input space, and passes it through the downstream model.

    This is used to apply SHAP directly in the latent space.

    :param model: Predictive model accepting decoded input.
    :type model: torch.nn.Module
    :param decoder: Decoder mapping latent to input space.
    :type decoder: torch.nn.Module
    """
    def __init__(self, model, decoder):
        super().__init__()
        self.model = model
        self.decoder = decoder

    def forward(self, z):
        r"""
        Pass latent vector through decoder and then model.

        :param z: Latent input vector or batch.
        :type z: torch.Tensor
        :return: Model output for decoded input.
        :rtype: torch.Tensor
        """
        if z.dim() == 1:
            z = z.unsqueeze(0)
        x = self.decoder(z)
        if isinstance(x, (tuple, list)):
            x = x[0]
        return self.model(x)

def make_shap_explainer(explainer_class, model, background, **kwargs):
    r"""
    Instantiate a SHAP explainer object in a version-agnostic way.

    Handles differences in SHAP's API for older and newer versions, where the second argument
    might be named "data", "background", or another keyword.

    :param explainer_class: SHAP explainer class (e.g., GradientExplainer).
    :param model: The model to explain.
    :param background: Background dataset for SHAP.
    :param kwargs: Additional keyword arguments.
    :return: Instantiated SHAP explainer.
    """
    sig = inspect.signature(explainer_class.__init__)
    params = [p for p in sig.parameters if p != "self"]
    if len(params) == 1:
        return explainer_class(model, **kwargs)
    elif len(params) > 1:
        if background is not None:
            p2 = params[1].lower()
            if p2 in ("data", "background"):
                return explainer_class(model, background, **kwargs)
            else:
                return explainer_class(model, **{p2: background}, **kwargs)
        else:
            return explainer_class(model, **kwargs)
    else:
        raise RuntimeError("Can't infer how to call explainer_class")

def train_conv1d_autoencoder(encoder, decoder, X, epochs=80, lr=1e-3, device="cpu", batch_size=32):
    r"""
    Train a Conv1dEncoder and Conv1dDecoder on time-series data using MSE loss.

    This function fits an autoencoder to reconstruct its input using batched SGD.

    :param encoder: Instance of Conv1dEncoder.
    :type encoder: torch.nn.Module
    :param decoder: Instance of Conv1dDecoder.
    :type decoder: torch.nn.Module
    :param X: Input data of shape (N, T, F), either np.ndarray or torch.Tensor.
    :type X: np.ndarray or torch.Tensor
    :param int epochs: Number of training epochs.
    :param float lr: Learning rate for Adam optimizer.
    :param str device: Device to train on (e.g., 'cpu' or 'cuda').
    :param int batch_size: Mini-batch size.
    """
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()
    # Convert input to torch.Tensor
    if not torch.is_tensor(X):
        X_t = torch.tensor(X, dtype=torch.float32, device=device)
    else:
        X_t = X.to(device)
    N = X_t.shape[0]
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = torch.nn.MSELoss()
    for epoch in range(epochs):
        # Mini-batch training
        perm = torch.randperm(N)
        epoch_loss = 0.0
        for i in range(0, N, batch_size):
            idx = perm[i:i+batch_size]
            batch = X_t[idx]
            optimizer.zero_grad()
            z = encoder(batch)
            X_rec = decoder(z)
            loss = criterion(X_rec, batch)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * batch.size(0)
        epoch_loss /= N
        if (epoch+1) % max(1, (epochs // 8)) == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: Recon Loss = {epoch_loss:.5f}")
    encoder.eval()
    decoder.eval()


class LatentSHAPExplainer(BaseExplainer):
    r"""
    LatentSHAPExplainer: SHAP Attribution in Autoencoded Latent Space

    This class applies SHAP to the latent space of an autoencoder and projects the resulting attributions 
    back into input space using the decoder's Jacobian. It is especially useful for high-dimensional, 
    structured inputs (e.g., time series) where direct SHAP attribution is noisy or expensive.

    .. math::
        \phi_{\text{input}} = J_{\text{decoder}}(z) \cdot \phi_{\text{latent}}

    :param model: Downstream predictive model, operating in input space.
    :type model: torch.nn.Module
    :param encoder: Encoder network mapping input to latent space.
    :type encoder: torch.nn.Module
    :param decoder: Decoder network mapping latent to input space.
    :type decoder: torch.nn.Module
    :param base_explainer_class: SHAP explainer class (e.g., GradientExplainer).
    :param background: Background dataset (N, T, F) used for SHAP estimation.
    :type background: np.ndarray or torch.Tensor
    :param device: Device context (e.g., 'cpu' or 'cuda').
    :param base_explainer_kwargs: Optional dictionary of kwargs passed to SHAP explainer.
    :type base_explainer_kwargs: dict
    """
    def __init__(
        self,
        model,
        encoder,
        decoder,
        base_explainer_class,
        background,
        device=None,
        base_explainer_kwargs=None
    ):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = encoder.to(self.device)
        self.decoder = decoder.to(self.device)
        self.model = model.to(self.device)
        self.base_explainer_kwargs = base_explainer_kwargs or {}

        # Encode background (should be torch.Tensor)
        if not torch.is_tensor(background):
            background = torch.tensor(background, dtype=torch.float32, device=self.device)
        else:
            background = background.to(self.device)
        with torch.no_grad():
            latent_bg = encoder(background)
            if isinstance(latent_bg, (tuple, list)):
                latent_bg = latent_bg[0]

        # Wrap the model for latent input
        self.latent_model = LatentModelWrapper(self.model, self.decoder).to(self.device)
        # Construct SHAP explainer in latent space
        self.base_explainer = make_shap_explainer(
            base_explainer_class, self.latent_model, latent_bg, **self.base_explainer_kwargs
        )

        self.input_shape = background.shape[1:]
        self.latent_dim = latent_bg.shape[1]

    def _decoder_jacobian(self, latent_vec):
        r"""
        Compute the Jacobian of the decoder output with respect to a latent vector.

        .. math::
            J(z) = \frac{\partial \text{Decoder}(z)}{\partial z}

        :param latent_vec: Latent representation of a single sample.
        :type latent_vec: np.ndarray
        :return: Jacobian matrix of shape (input_dim, latent_dim).
        :rtype: np.ndarray
        """
        latent = torch.tensor(latent_vec, dtype=torch.float32, device=self.device, requires_grad=True)
        x_dec = self.decoder(latent.unsqueeze(0))
        if isinstance(x_dec, (tuple, list)):
            x_dec = x_dec[0]
        x_dec = x_dec.reshape(-1)
        jac = torch.autograd.functional.jacobian(
            lambda z: self.decoder(z.unsqueeze(0)).reshape(-1),
            latent, create_graph=False, vectorize=True
        )
        return jac.detach().cpu().numpy()

    def _pathwise_decoder_jacobian(self, z_base, z, n_steps=10):
        r"""
        Compute the average decoder Jacobian over interpolated path from baseline to latent vector.

        .. math::
            \bar{J} = \frac{1}{n} \sum_{i=1}^{n} J\left((1 - \alpha_i) z_{\text{base}} + \alpha_i z\right)

        :param z_base: Baseline latent vector (e.g., mean of background).
        :type z_base: np.ndarray
        :param z: Target latent vector for SHAP.
        :type z: np.ndarray
        :param int n_steps: Number of interpolation steps.
        :return: Averaged Jacobian matrix along the path.
        :rtype: np.ndarray
        """
        alphas = np.linspace(0, 1, n_steps)
        jacs = []
        for alpha in alphas:
            z_mix = (1 - alpha) * z_base + alpha * z
            jacs.append(self._decoder_jacobian(z_mix))
        return np.mean(jacs, axis=0)

    def shap_values(self, X, **kwargs):
        r"""
        Compute SHAP values in latent space and project them to input space.

        Steps:
        1. Encode input and background into latent space.
        2. Run SHAP (e.g., GradientExplainer) on latent input.
        3. Compute Jacobian from latent to input.
        4. Project latent SHAP values using the Jacobian.
        5. Return attributions with original input shape.

        .. math::
            \phi_{\text{input}} = J(z) \cdot \phi_{\text{latent}}

        :param X: Input sample(s), shape (T, F) or (B, T, F)
        :type X: np.ndarray or torch.Tensor
        :return: SHAP attributions in input space.
        :rtype: np.ndarray
        """
        single_input = False
        if isinstance(X, np.ndarray):
            X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
            if X.ndim == len(self.input_shape):
                X_t = X_t.unsqueeze(0)
                single_input = True
        elif torch.is_tensor(X):
            X_t = X.to(self.device)
            if X.dim() == len(self.input_shape):
                X_t = X_t.unsqueeze(0)
                single_input = True
        else:
            raise TypeError("Input X must be np.ndarray or torch.Tensor")

        with torch.no_grad():
            latent_X = self.encoder(X_t)
            if isinstance(latent_X, (tuple, list)):
                latent_X = latent_X[0]
        # SHAP expects a tensor on the correct device for GradientExplainer
        latent_shap = self.base_explainer.shap_values(latent_X, **kwargs)
        if isinstance(latent_shap, list):
            latent_shap = latent_shap[0]
        latent_X_np = latent_X.detach().cpu().numpy()
        latent_shap_np = latent_shap.detach().cpu().numpy() if torch.is_tensor(latent_shap) else np.asarray(latent_shap)
        B = latent_X_np.shape[0]
        # Get mean baseline in latent
        if hasattr(self.base_explainer, "data"):
            z_base = self.base_explainer.data.mean(dim=0).detach().cpu().numpy()
        else:
            z_base = np.zeros(self.latent_dim)
        input_attr = np.zeros((B, np.prod(self.input_shape)))
        for i in range(B):
            jac = self._pathwise_decoder_jacobian(z_base, latent_X_np[i], n_steps=10)
            phi_latent = latent_shap_np[i].squeeze()
            input_attr[i] = jac @ phi_latent
        input_attr = input_attr.reshape((B,) + self.input_shape)
        return input_attr[0] if single_input else input_attr

# ====================
# Usage Example:
# ====================
if __name__ == "__main__":
    # Generate data (T=10, F=3, N=100)
    T, F, N = 10, 3, 100
    X = np.random.randn(N, T, F)
    y = np.sum(X, axis=(1,2)) + np.random.randn(N) * 0.1

    # Dummy model
    class DummyLSTM(nn.Module):
        def __init__(self, input_dim):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, 16, batch_first=True)
            self.fc = nn.Linear(16, 1)
        def forward(self, x):
            out, _ = self.lstm(x)
            return self.fc(out[:, -1, :])

    model = DummyLSTM(F)
    encoder = Conv1dEncoder(F, T, latent_dim=4)
    decoder = Conv1dDecoder(latent_dim=4, seq_len=T, output_features=F)

    # Quick autoencoder train (optional)
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    for epoch in range(10):
        idx = np.random.choice(len(X), 32)
        Xb = torch.tensor(X[idx], dtype=torch.float32)
        z = encoder(Xb)
        X_rec = decoder(z)
        loss = ((X_rec - Xb) ** 2).mean()
        enc_opt.zero_grad()
        dec_opt.zero_grad()
        loss.backward()
        enc_opt.step()
        dec_opt.step()

    # LatentSHAPExplainer with GradientExplainer
    latent_expl = LatentSHAPExplainer(
        model=model,
        encoder=encoder,
        decoder=decoder,
        base_explainer_class=shap.GradientExplainer,
        background=X[:32],
        device="cpu"
    )

    x_test = X[0]
    attr = latent_expl.shap_values(x_test)
    print("Attr shape:", attr.shape)  # (T, F)

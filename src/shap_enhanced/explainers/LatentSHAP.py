"""
LatentSHAP: SHAP in Latent Space Explainer

Runs SHAP attribution in the latent space (from an encoder), then maps the latent attributions
back to the original input space using a decoder. Includes a general-purpose autoencoder.

References:
- https://arxiv.org/abs/2006.06015 (Latent Attribution for Deep Models)
"""

import numpy as np
import torch
import torch.nn as nn
import inspect

from shap_enhanced.base_explainer import BaseExplainer


# ---- General-purpose autoencoder modules ----

class DefaultEncoder(nn.Module):
    """
    General-purpose encoder for tabular or time series data.

    Parameters
    ----------
    input_dim : int
        Number of features per timestep (F).
    seq_len : int
        Number of timesteps (T).
    latent_dim : int
        Dimension of the latent space.
    """
    def __init__(self, input_dim, seq_len, latent_dim=8):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.flatten = nn.Flatten()
        self.encoder = nn.Sequential(
            nn.Linear(seq_len * input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, latent_dim),
        )

    def forward(self, x):
        # x: (B, T, F) or (T, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        x = self.flatten(x)  # (B, T*F)
        z = self.encoder(x)
        return z  # (B, latent_dim)

class DefaultDecoder(nn.Module):
    """
    General-purpose decoder for tabular or time series data.

    Parameters
    ----------
    latent_dim : int
        Dimension of the latent space.
    seq_len : int
        Number of timesteps (T).
    input_dim : int
        Number of features per timestep (F).
    """
    def __init__(self, latent_dim, seq_len, input_dim):
        super().__init__()
        self.seq_len = seq_len
        self.input_dim = input_dim
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, seq_len * input_dim),
        )

    def forward(self, z):
        # z: (B, latent_dim) or (latent_dim,)
        if z.dim() == 1:
            z = z.unsqueeze(0)
        x_flat = self.decoder(z)
        x_rec = x_flat.view(-1, self.seq_len, self.input_dim)
        return x_rec  # (B, T, F)
    
def train_default_autoencoder(encoder, decoder, data, n_epochs=100, lr=1e-3, device="cpu"):
    """
    Trains the encoder-decoder autoencoder on provided data.

    Args:
        encoder, decoder: torch.nn.Module
        data: np.ndarray or torch.Tensor, shape (N, T, F)
        n_epochs: int
        lr: float
        device: str
    """
    encoder.to(device)
    decoder.to(device)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    encoder.train()
    decoder.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        z = encoder(data)
        x_rec = decoder(z)
        loss = criterion(x_rec, data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Recon Loss: {loss.item():.4f}")
    encoder.eval()
    decoder.eval()

class LSTMEncoder(nn.Module):
    """
    Sequence encoder: (B, T, F) -> (B, latent_dim)
    """
    def __init__(self, input_dim, latent_dim=16, hidden_dim=32, num_layers=1, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional
        )
        self.fc = nn.Linear(hidden_dim * (2 if bidirectional else 1), latent_dim)

    def forward(self, x):
        # x: (B, T, F)
        if x.dim() == 2:
            x = x.unsqueeze(0)
        _, (h_n, _) = self.lstm(x)
        # Take last hidden state from all layers, concat if bidirectional
        h = h_n.transpose(0, 1).reshape(x.shape[0], -1)  # (B, H*num_layers*(2?))
        z = self.fc(h)
        return z  # (B, latent_dim)

class LSTMDecoder(nn.Module):
    def __init__(self, latent_dim, output_dim, seq_len, hidden_dim=32, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        self.init_fc = nn.Linear(latent_dim, hidden_dim * num_layers)
        self.out_fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        # z: (B, latent_dim)
        B = z.shape[0]
        num_layers = self.lstm.num_layers
        hidden_dim = self.lstm.hidden_size

        hidden = self.init_fc(z)
        assert hidden.shape[1] == num_layers * hidden_dim, \
            f"init_fc out: {hidden.shape}, need ({B}, {num_layers*hidden_dim})"

        hidden = hidden.view(B, num_layers, hidden_dim).permute(1, 0, 2).contiguous()
        cell = torch.zeros_like(hidden)
        x_in = torch.zeros(B, self.seq_len, self.lstm.input_size, device=z.device)
        out, _ = self.lstm(x_in, (hidden, cell))
        y = self.out_fc(out)
        return y

def train_LSTM_autoencoder(encoder, decoder, data, n_epochs=100, lr=1e-3, device="cpu"):
    encoder.to(device)
    decoder.to(device)
    data = torch.tensor(data, dtype=torch.float32, device=device)
    optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=lr)
    criterion = nn.MSELoss()
    encoder.train()
    decoder.train()
    for epoch in range(n_epochs):
        optimizer.zero_grad()
        z = encoder(data)
        x_rec = decoder(z)
        loss = criterion(x_rec, data)
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 20 == 0:
            print(f"Epoch {epoch+1}/{n_epochs}, Recon Loss: {loss.item():.4f}")
    encoder.eval()
    decoder.eval()


# ---- LatentSHAPExplainer ----

def ensure_float_numpy(arr):
    """Recursively converts input (tensor, list of tensors, object array, etc) to contiguous float32/64 numpy ndarray."""
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    elif isinstance(arr, (list, tuple)):
        arrs = [ensure_float_numpy(x) for x in arr]
        return np.stack(arrs)
    elif isinstance(arr, np.ndarray):
        if arr.dtype == object:
            arrs = [ensure_float_numpy(x) for x in arr]
            return np.stack(arrs)
        elif np.issubdtype(arr.dtype, np.floating):
            return arr
        else:
            return arr.astype(np.float32)
    else:
        return np.array(arr, dtype=np.float32)

def make_shap_explainer(explainer_class, model, background=None, **kwargs):
    """
    Instantiates a SHAP (or compatible) explainer, auto-handling 'background' vs 'data' positional/keyword.
    """
    sig = inspect.signature(explainer_class.__init__)
    params = list(sig.parameters.keys())
    params = [p for p in params if p != "self"]

    if len(params) == 1:
        return explainer_class(model, **kwargs)
    elif len(params) > 1:
        param2 = params[1].lower()
        if background is not None:
            if param2 in ("data", "background"):
                return explainer_class(model, background, **kwargs)
            else:
                return explainer_class(model, **{param2: background}, **kwargs)
        else:
            return explainer_class(model, **kwargs)
    else:
        raise RuntimeError("Cannot infer how to call explainer_class!")

def latent_model_wrapper(model, decoder, device):
    """
    Returns a function for SHAP: latent → decoder → model → numpy output.
    """
    def model_latent(z):
        """
        Robust wrapper: Accepts z as (latent_dim,), (B, latent_dim), (N, latent_dim), etc.
        Returns numpy array with shape (batch, ...) for SHAP KernelExplainer compatibility.
        """
        # Convert input to torch tensor on correct device
        if isinstance(z, np.ndarray):
            z_torch = torch.tensor(z, dtype=torch.float32, device=device)
        elif isinstance(z, torch.Tensor):
            z_torch = z.to(device)
        else:
            raise TypeError("Latent input must be numpy array or torch tensor")
        
        # Always flatten all but last dim to make (batch, latent_dim)
        z_flat = z_torch.reshape(-1, z_torch.shape[-1])

        # Decode
        x_dec = decoder(z_flat)
        if isinstance(x_dec, (tuple, list)):
            x_dec = x_dec[0]
        
        # Run through model
        out = model(x_dec)
        if isinstance(out, torch.Tensor):
            out = out.detach().cpu().numpy()
        elif not isinstance(out, np.ndarray):
            out = np.array(out)

        # SHAP compatibility: always at least 2D (batch, output)
        out = np.asarray(out)
        if out.ndim == 0:  # pure scalar
            out = out[None, None]
        elif out.ndim == 1:
            out = out[:, None]  # shape (batch, 1)
        elif out.ndim > 2:
            out = out.reshape(out.shape[0], -1)  # flatten everything after batch

        # (Optional) Debug: print output shape for first few calls
        # print(f"[LatentSHAP] model_latent: input z {z.shape if hasattr(z,'shape') else type(z)}, output {out.shape}")

        return out


    return model_latent

class LatentSHAPExplainer(BaseExplainer):
    """
    Latent SHAP Explainer.

    - Applies SHAP explainer in latent space.
    - Projects attributions to input space using decoder Jacobian.
    - Output attributions match input shape.
    """

    def __init__(
        self,
        model,
        encoder,
        decoder,
        base_explainer_class,
        background,
        device=None,
        base_explainer_kwargs=None,
    ):
        super().__init__(model, background)
        self.encoder = encoder
        self.decoder = decoder
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.base_explainer_kwargs = base_explainer_kwargs or {}

        # Encode background and convert to numpy (N, latent_dim)
        print("[LatentSHAP] Encoding background...")
        latent_bg = encoder(torch.tensor(background, dtype=torch.float32, device=self.device)) \
            if not isinstance(background, torch.Tensor) else encoder(background.to(self.device))
        if isinstance(latent_bg, (tuple, list)):
            latent_bg = latent_bg[0]
        latent_bg = ensure_float_numpy(latent_bg)

        print("latent_bg:", type(latent_bg), latent_bg.dtype, latent_bg.shape)

        # Make a model wrapper that takes latent space and returns output as numpy
        self.model_in_latent = latent_model_wrapper(model, decoder, self.device)

        # Universal background passing
        self.base_explainer = make_shap_explainer(
            base_explainer_class, self.model_in_latent, background=latent_bg, **self.base_explainer_kwargs
        )
        print("[LatentSHAP] Base explainer initialized.")

        # Save dimensions
        self.input_shape = background.shape[1:]
        self.latent_dim = latent_bg.shape[1]

    def _decoder_jacobian(self, latent_vec):
        """
        Returns the Jacobian (input_dim_flat, latent_dim) of decoder wrt latent_vec.
        """
        latent = torch.tensor(latent_vec, dtype=torch.float32, device=self.device, requires_grad=True)
        x_dec = self.decoder(latent.unsqueeze(0))
        if isinstance(x_dec, (tuple, list)):
            x_dec = x_dec[0]
        x_dec = x_dec.view(-1)  # flatten input
        jac = torch.autograd.functional.jacobian(
            lambda z: self.decoder(z.unsqueeze(0))[0].reshape(-1),
            latent,
            create_graph=False,
            vectorize=True
        )
        return jac.detach().cpu().numpy()  # (input_dim_flat, latent_dim)

    def _pathwise_decoder_jacobian(self, z_start, z_end, n_steps=10):
        """
        Returns the average Jacobian from z_start to z_end in latent space.
        """
        z_start = np.array(z_start)
        z_end = np.array(z_end)
        alphas = np.linspace(0, 1, n_steps)
        jacs = []
        for alpha in alphas:
            z = (1 - alpha) * z_start + alpha * z_end
            jacs.append(self._decoder_jacobian(z))
        return np.mean(jacs, axis=0)

    def shap_values(self, X, **kwargs):
        # Encode X to latent space
        if isinstance(X, np.ndarray):
            X_torch = torch.tensor(X, dtype=torch.float32, device=self.device)
        else:
            X_torch = X.to(self.device)
        latent_X = self.encoder(X_torch)
        if isinstance(latent_X, (tuple, list)):
            latent_X = latent_X[0]
        latent_X_np = ensure_float_numpy(latent_X)
        latent_shap = self.base_explainer.shap_values(latent_X_np, **kwargs)
        if isinstance(latent_shap, list):
            latent_shap = latent_shap[0]
        if latent_shap.shape[0] != latent_X_np.shape[0]:
            latent_shap = np.array(latent_shap)[None, ...]
        B = latent_X_np.shape[0]
        input_attr = np.zeros((B, np.prod(self.input_shape)))
        z_base = self.base_explainer.data.data.mean(axis=0)
        for i in range(B):
            jac = self._pathwise_decoder_jacobian(z_base, latent_X_np[i], n_steps=10)
            phi_latent = latent_shap[i].squeeze()
            # Optionally normalize
            phi_latent = phi_latent / (np.linalg.norm(phi_latent) + 1e-6)
            input_attr[i] = jac @ phi_latent
        input_attr = input_attr.reshape((B,) + self.input_shape)
        if input_attr.shape[0] == 1:
            return input_attr[0]
        return input_attr
"""Simple LSTM model for sequential data."""

import torch
import torch.nn as nn
from shap_enhanced.models.base_model import BaseModel

class SimpleLSTM(BaseModel):
    """Simple LSTM model for sequential data."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 1):
        """Initialize the Simple LSTM model.

        :param int input_size: Number of input features.
        :param int hidden_size: Number of features in the hidden state.
        :param int output_size: Number of output features.
        :param int num_layers: Number of LSTM layers (default is 1).
        """
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        """Train the LSTM model on input data X and target labels y."""
        self.lstm.train()
        criterion = nn.MSELoss()  # Assuming regression, adjust for classification
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Train loop
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_x = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # Forward pass
                output, (hn, cn) = self.lstm(batch_x)
                output = self.fc(output[:, -1, :])  # Use the last time step output

                # Compute loss and backpropagate
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input X."""
        self.lstm.eval()
        with torch.no_grad():
            output, (hn, cn) = self.lstm(X)
            output = self.fc(output[:, -1, :])  # Use the last time step output
        return output

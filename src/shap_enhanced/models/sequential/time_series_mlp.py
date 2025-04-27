# src/shap_enhanced/models/sequential/time_series_mlp.py

import torch
import torch.nn as nn
from shap_enhanced.models.base_model import BaseModel

class TimeSeriesMLP(BaseModel):
    """Multilayer Perceptron (MLP) for time-series data."""

    def __init__(self, input_size: int, hidden_size: int, output_size: int, num_layers: int = 2):
        """Initialize the Time Series MLP model.

        :param int input_size: Number of input features.
        :param int hidden_size: Number of hidden units in the layers.
        :param int output_size: Number of output features.
        :param int num_layers: Number of layers in the MLP (default is 2).
        """
        super().__init__()
        
        self.num_layers = num_layers
        self.hidden_size = hidden_size

        # List of fully connected layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())  # Apply ReLU activation

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, output_size))

        # Sequential container to hold the layers
        self.mlp = nn.Sequential(*layers)

    def fit(self, X: torch.Tensor, y: torch.Tensor, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        """Train the Time Series MLP model on input data X and target labels y."""
        self.mlp.train()
        criterion = nn.MSELoss()  # Assuming regression, adjust for classification
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Train loop
        for epoch in range(epochs):
            for i in range(0, len(X), batch_size):
                batch_x = X[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # Forward pass
                output = self.mlp(batch_x)

                # Compute loss and backpropagate
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input X."""
        self.mlp.eval()
        with torch.no_grad():
            output = self.mlp(X)
        return output

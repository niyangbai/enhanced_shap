"""Early Fusion Model for multimodal data."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from shap_enhanced.models.base_model import BaseModel

class EarlyFusionModel(BaseModel):
    """Early Fusion Model for multimodal data."""

    def __init__(self, tabular_input_size: int, image_input_channels: int, 
                 image_output_size: int, text_input_size: int, hidden_size: int, output_size: int):
        """Initialize the Early Fusion Model.

        :param int tabular_input_size: Number of tabular features.
        :param int image_input_channels: Number of input channels for image data.
        :param int image_output_size: Output size of the CNN model (flattened).
        :param int text_input_size: Number of words in the text input (vocabulary size).
        :param int hidden_size: Hidden layer size for text and tabular features.
        :param int output_size: Number of output features (depending on the task).
        """
        super().__init__()

        # Tabular MLP (for tabular data)
        self.tabular_mlp = nn.Sequential(
            nn.Linear(tabular_input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # CNN (for image data)
        self.cnn = nn.Sequential(
            nn.Conv2d(image_input_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Flatten()
        )
        self.image_fc = nn.Linear(128 * (image_output_size // 4) * (image_output_size // 4), hidden_size)

        # RNN (for text data)
        self.text_rnn = nn.LSTM(text_input_size, hidden_size, batch_first=True, bidirectional=True)
        self.text_fc = nn.Linear(hidden_size * 2, hidden_size)  # Bidirectional, so multiply by 2

        # Fully connected layer for the combined features
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * 3, hidden_size),  # MLP, CNN, and Text features
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def fit(self, X_tabular: torch.Tensor, X_images: torch.Tensor, X_text: torch.Tensor, y: torch.Tensor, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        """Train the Early Fusion model on tabular, image, and text data."""
        self.train()
        criterion = nn.MSELoss()  # Assuming regression; adjust for classification
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        # Train loop
        for epoch in range(epochs):
            for i in range(0, len(X_tabular), batch_size):
                batch_tabular = X_tabular[i:i+batch_size]
                batch_images = X_images[i:i+batch_size]
                batch_text = X_text[i:i+batch_size]
                batch_y = y[i:i+batch_size]

                # Forward pass for each modality
                tabular_features = self.tabular_mlp(batch_tabular)
                image_features = self.cnn(batch_images)
                image_features = self.image_fc(image_features)
                _, (text_features, _) = self.text_rnn(batch_text)
                text_features = self.text_fc(text_features[-1])  # Use last hidden state

                # Combine features from all modalities
                combined_features = torch.cat([tabular_features, image_features, text_features], dim=1)

                # Fully connected output
                output = self.fc(combined_features)

                # Compute loss and backpropagate
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict(self, X_tabular: torch.Tensor, X_images: torch.Tensor, X_text: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input (tabular, image, text)."""
        self.eval()
        with torch.no_grad():
            tabular_features = self.tabular_mlp(X_tabular)
            image_features = self.cnn(X_images)
            image_features = self.image_fc(image_features)
            _, (text_features, _) = self.text_rnn(X_text)
            text_features = self.text_fc(text_features[-1])  # Use last hidden state

            # Combine features from all modalities
            combined_features = torch.cat([tabular_features, image_features, text_features], dim=1)

            # Final output
            output = self.fc(combined_features)
        return output

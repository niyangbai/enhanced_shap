"""Simple CNN model for image classification or regression."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from shap_enhanced.models.base_model import BaseModel


class SimpleCNN(nn.Module, BaseModel):
    """Simple CNN model for image classification or regression."""

    def __init__(self, input_channels: int, num_classes: int):
        """
        Initialize the Simple CNN model.

        :param input_channels: Number of input channels for the image (e.g., 3 for RGB).
        :param num_classes: Number of output classes (for classification).
        """
        super().__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),  # Assuming input image size is 32x32
            nn.ReLU(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CNN model.

        :param x: Input tensor of shape (batch_size, input_channels, height, width).
        :return: Output tensor of shape (batch_size, num_classes).
        """
        x = self.feature_extractor(x)
        x = self.classifier(x)
        return x

    def fit(
        self,
        X: torch.Tensor,
        y: torch.Tensor,
        epochs: int = 20,
        batch_size: int = 32,
        lr: float = 0.001,
        validation_data: tuple = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        """
        Train the Simple CNN model on image data.

        :param X: Training data tensor.
        :param y: Training labels tensor.
        :param epochs: Number of training epochs.
        :param batch_size: Batch size for training.
        :param lr: Learning rate for the optimizer.
        :param validation_data: Tuple (X_val, y_val) for validation.
        :param device: Device to run the training on ('cuda' or 'cpu').
        """
        self.to(device)
        self.train()

        train_dataset = TensorDataset(X, y)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)

                # Forward pass
                output = self.forward(batch_x)
                loss = criterion(output, batch_y)

                # Backward pass and optimization
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {epoch_loss / len(train_loader):.4f}")

            # Validation step
            if validation_data:
                self.validate(*validation_data, device=device)

    def validate(self, X_val: torch.Tensor, y_val: torch.Tensor, device: str = "cpu"):
        """
        Validate the model on validation data.

        :param X_val: Validation data tensor.
        :param y_val: Validation labels tensor.
        :param device: Device to run the validation on ('cuda' or 'cpu').
        """
        self.eval()
        X_val, y_val = X_val.to(device), y_val.to(device)

        with torch.no_grad():
            output = self.forward(X_val)
            loss = nn.CrossEntropyLoss()(output, y_val)
            accuracy = (output.argmax(dim=1) == y_val).float().mean().item()

        print(f"Validation Loss: {loss.item():.4f}, Accuracy: {accuracy * 100:.2f}%")

    def predict(self, X: torch.Tensor, device: str = "cpu") -> torch.Tensor:
        """
        Predict outputs for the given input X.

        :param X: Input tensor.
        :param device: Device to run the prediction on ('cuda' or 'cpu').
        :return: Predicted outputs.
        """
        self.eval()
        X = X.to(device)

        with torch.no_grad():
            output = self.forward(X)

        return output

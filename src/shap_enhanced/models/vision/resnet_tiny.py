# src/shap_enhanced/models/vision/resnet_tiny.py

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import logging

logging.basicConfig(level=logging.INFO)

class ResNetTiny(nn.Module):
    """ResNet Tiny model for image classification."""

    def __init__(self, input_channels: int, num_classes: int, device: torch.device = None):
        """Initialize the ResNet Tiny model.

        :param int input_channels: Number of input channels for the image (e.g., 3 for RGB).
        :param int num_classes: Number of output classes (for classification).
        :param torch.device device: Device to run the model on (CPU or GPU).
        """
        super(ResNetTiny, self).__init__()

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet block
        self.layer1 = self._make_layer(64, 128, 2, downsample=True)
        self.layer2 = self._make_layer(128, 256, 2, downsample=True)

        self.fc = nn.Linear(256, num_classes)

        self.to(self.device)

    def _make_layer(self, in_channels: int, out_channels: int, num_blocks: int, downsample: bool = False) -> nn.Sequential:
        """Create a residual block.

        :param int in_channels: Number of input channels.
        :param int out_channels: Number of output channels.
        :param int num_blocks: Number of residual blocks.
        :param bool downsample: Whether to downsample the input.
        :return: A sequential container of residual blocks.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(self._residual_block(in_channels, out_channels, downsample=(i == 0 and downsample)))
            in_channels = out_channels
        return nn.Sequential(*layers)

    def _residual_block(self, in_channels: int, out_channels: int, downsample: bool = False) -> nn.Module:
        """Create a residual block with a skip connection.

        :param int in_channels: Number of input channels.
        :param int out_channels: Number of output channels.
        :param bool downsample: Whether to downsample the input.
        :return: A residual block.
        """
        stride = 2 if downsample else 1
        shortcut = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels)
        ) if downsample or in_channels != out_channels else nn.Identity()

        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            shortcut
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the ResNet Tiny model."""
        x = x.to(self.device)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = torch.flatten(x, 1)  # Flatten the output before feeding into the fully connected layer
        x = self.fc(x)
        return x

    def fit(self, train_data: torch.Tensor, train_labels: torch.Tensor, val_data: torch.Tensor = None,
            val_labels: torch.Tensor = None, epochs: int = 20, batch_size: int = 32, lr: float = 0.001):
        """Train the ResNet Tiny model on image data."""
        self.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        train_dataset = TensorDataset(train_data, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_loader = None
        if val_data is not None and val_labels is not None:
            val_dataset = TensorDataset(val_data, val_labels)
            val_loader = DataLoader(val_dataset, batch_size=batch_size)

        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, batch_y in train_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)

                # Forward pass
                output = self.forward(batch_x)

                # Compute loss and backpropagate
                loss = criterion(output, batch_y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            logging.info(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(train_loader):.4f}")

            if val_loader:
                self._validate(val_loader, criterion)

    def _validate(self, val_loader: DataLoader, criterion: nn.Module):
        """Validate the model on validation data."""
        self.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
                output = self.forward(batch_x)
                val_loss += criterion(output, batch_y).item()
                _, predicted = torch.max(output, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()

        logging.info(f"Validation Loss: {val_loss/len(val_loader):.4f}, Accuracy: {100 * correct / total:.2f}%")

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        """Predict outputs for the given input X."""
        self.eval()
        X = X.to(self.device)
        with torch.no_grad():
            output = self.forward(X)
        return output

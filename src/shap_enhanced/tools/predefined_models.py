"""
Neural Network Models for Tabular and Sequential Data
=====================================================

Overview
--------

This module defines two neural network architectures tailored for different data modalities:

- **RealisticLSTM**: A bidirectional LSTM model with attention over time, designed for sequential input (e.g., time series).
- **TabularMLP**: A simple multi-layer perceptron (MLP) for static, tabular input.

Both models are suitable for regression or binary classification tasks, and can be used in conjunction with  
feature attribution or SHAP-based interpretability methods.

Model Summaries
^^^^^^^^^^^^^^^^

- **RealisticLSTM**:
  - Bidirectional LSTM with configurable depth and dropout.
  - Temporal attention mechanism over LSTM outputs.
  - Feedforward head for final prediction.
  - Input shape: ``(batch_size, seq_len, input_dim)``

- **TabularMLP**:
  - Lightweight MLP with one hidden layer and ReLU activation.
  - Input shape: ``(batch_size, input_dim)``

Use Case
--------

These models are intended for:
- Interpretable machine learning pipelines.
- SHAP explanation experiments on sequential and tabular benchmarks.
- General-purpose regression or binary classification.

Example
-------

.. code-block:: python

    model_seq = RealisticLSTM(input_dim=5, hidden_dim=32)
    model_tab = TabularMLP(input_dim=10)
    y_pred_seq = model_seq(x_seq)  # x_seq: shape (B, T, 5)
    y_pred_tab = model_tab(x_tab)  # x_tab: shape (B, 10)
"""


import torch
import torch.nn as nn

class RealisticLSTM(nn.Module):
    r"""
    Bidirectional LSTM model with temporal attention for sequential input.

    This model processes sequence data with a bidirectional LSTM and computes an attention-weighted  
    context vector across time. The resulting vector is passed through a feedforward head for final prediction.

    .. note::
        Designed for regression or binary classification tasks on temporal data such as time series.

    :param int input_dim: Number of input features at each timestep.
    :param int hidden_dim: Hidden size for LSTM and intermediate layers (default: 32).
    :param int num_layers: Number of LSTM layers (default: 2).
    :param int output_dim: Output dimension, e.g., 1 for scalar prediction (default: 1).
    :param float dropout: Dropout rate applied after attention context (default: 0.2).
    :return: Output tensor of shape (batch_size, output_dim).
    :rtype: torch.Tensor
    """
    def __init__(self, input_dim, hidden_dim=32, num_layers=2, output_dim=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.attn_fc = nn.Linear(hidden_dim * 2, 1)  # attention over time
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        out, _ = self.lstm(x)  # (B, T, 2*H)
        attn_weights = torch.softmax(self.attn_fc(out).squeeze(-1), dim=1)  # (B, T)
        context = torch.sum(out * attn_weights.unsqueeze(-1), dim=1)        # (B, 2*H)
        context = self.dropout(context)
        return self.head(context)


class TabularMLP(nn.Module):
    r"""
    Lightweight multilayer perceptron (MLP) for tabular regression or classification tasks.

    Consists of a single hidden layer with ReLU activation and a linear output layer.  
    Suitable for benchmarking SHAP explanations on tabular data.

    :param int input_dim: Number of input features.
    :param int hidden_dim: Number of hidden units (default: 16).
    :param int output_dim: Output dimension (default: 1).
    :return: Output tensor of shape (batch_size, output_dim).
    :rtype: torch.Tensor
    """
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
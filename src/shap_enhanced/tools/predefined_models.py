import torch
import torch.nn as nn

class RealisticLSTM(nn.Module):
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
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)
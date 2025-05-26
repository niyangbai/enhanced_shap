import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 1) SIMULATION
np.random.seed(0)
torch.manual_seed(0)

n_samples = 500
time_steps = 20
feat_dim = 1

# true importance: only timesteps 5 (weight=1) and 15 (weight=2)
true_imp = np.zeros(time_steps)
true_imp[5] = 1.0
true_imp[15] = 2.0

# generate X and y
X = np.random.randn(n_samples, time_steps, feat_dim).astype(np.float32)
y = X[:, 5, 0]*true_imp[5] + X[:, 15, 0]*true_imp[15] + 0.1*np.random.randn(n_samples)
y = y.astype(np.float32)

# split train/test
split = int(0.8 * n_samples)
X_train, X_test = X[:split], X[split:]
y_train, y_test = y[:split], y[split:]

# 2) LSTM REGRESSOR
class LSTMRegressor(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim=1):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hid_dim, batch_first=True)
        self.fc = nn.Linear(hid_dim, out_dim)
    def forward(self, x):
        _, (h, _) = self.lstm(x)
        return self.fc(h.squeeze(0))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMRegressor(feat_dim, hid_dim=16).to(device)
opt = optim.Adam(model.parameters(), lr=1e-2)
loss_fn = nn.MSELoss()

# training loop
model.train()
for epoch in range(300):
    x_batch = torch.from_numpy(X_train).to(device)
    y_batch = torch.from_numpy(y_train).unsqueeze(1).to(device)
    opt.zero_grad()
    pred = model(x_batch)
    loss = loss_fn(pred, y_batch)
    loss.backward()
    opt.step()
    if epoch % 100 == 0:
        print(f"Epoch {epoch}, loss {loss.item():.4f}")

# 3) REAL vs. EXPLAINED IMPORTANCE
from shap_enhanced.explainers.sequential_attention.recurrent_explainer import RecurrentExplainer

model.eval()
XT = torch.from_numpy(X_test).to(device)
# compute the explainer’s attribution
explainer = RecurrentExplainer(model, mode="mask", mask_value=0.0, nsamples=20)
attr = explainer.explain(XT)             # shape [batch, time]
est_imp = attr.mean(0).cpu().numpy()     # average over test batch

# normalize both for plotting
true_norm = true_imp / true_imp.max()
est_norm  = est_imp / est_imp.max()

# 4) PLOT
plt.plot(true_norm, label="True importance")
plt.plot(est_norm,  label="Estimated importance")
plt.xlabel("Time step")
plt.ylabel("Normalized importance")
plt.title("True vs. LSTM‐RecurrentExplainer Importance")
plt.legend()
plt.show()

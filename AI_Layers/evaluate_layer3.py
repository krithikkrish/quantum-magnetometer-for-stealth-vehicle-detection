# =====================================================================
# MAC STABILITY FIX
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error

# -------------------------------------------------------------
# 1. LOAD DATASET
# -------------------------------------------------------------
if not os.path.exists("layer3_kin.npz"):
    raise FileNotFoundError("layer3_kin.npz not found! Run 'python3 wake_data_generator.py' first.")

data = np.load("layer3_kin.npz")
X = data["X"]
Y = data["Y"]

MIN_BOUNDS = np.array([200.0, 100.0, 500.0, 0.0], dtype=np.float32)
MAX_BOUNDS = np.array([600.0, 5000.0, 5000.0, 2 * np.pi], dtype=np.float32)

np.random.seed(42)
n_samples = len(X)
n_train = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)
val_idx = indices[n_train:]

val_x = torch.tensor(X[val_idx], dtype=torch.float32).unsqueeze(1)
val_y_true = Y[val_idx]

# -------------------------------------------------------------
# 2. MODEL ARCHITECTURE
# -------------------------------------------------------------
class KinematicsCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.Conv1d(32, 64, kernel_size=9, stride=4, padding=4),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
        )
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        self.regressor = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()
        )

    def forward(self, x):
        feat = self.conv_stem(x)
        feat = feat.permute(0, 2, 1)
        lstm_out, _ = self.lstm(feat)
        pooled = torch.mean(lstm_out, dim=1)
        return self.regressor(pooled)

# -------------------------------------------------------------
# 3. EVALUATION
# -------------------------------------------------------------
print("Loading trained Layer 3 model weights ('layer3_kinematics.pth')...")
device = torch.device("cpu")
model = KinematicsCNNLSTM().to(device)

if not os.path.exists("layer3_kinematics.pth"):
    raise FileNotFoundError("Run 'python3 train_layer3.py' first to train and save 'layer3_kinematics.pth'!")

model.load_state_dict(torch.load("layer3_kinematics.pth", map_location=device))
model.eval()

print(f"Evaluating on {len(val_idx)} unseen test signals...")
with torch.no_grad():
    preds_norm = model(val_x).numpy()

# Denormalize to physical units
val_y_pred = preds_norm * (MAX_BOUNDS - MIN_BOUNDS) + MIN_BOUNDS

param_names = ["Velocity (m/s)", "Altitude (m)", "CPA Distance (m)", "Heading (rad)"]
units = ["m/s", "m", "m", "rad"]

print("\n" + "=" * 62)
print("            LAYER 3: KINEMATICS ESTIMATION REPORT")
print("=" * 62)
for i in range(4):
    rmse = np.sqrt(mean_squared_error(val_y_true[:, i], val_y_pred[:, i]))
    r2 = r2_score(val_y_true[:, i], val_y_pred[:, i])
    print(f"  {param_names[i]:<20} | RMSE: {rmse:>8.2f} {units[i]:<4} | R² Score: {r2:>6.4f}")
print("=" * 62)

# -------------------------------------------------------------
# 4. PLOT 4-PANEL REGRESSION FIGURE
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

for i in range(4):
    ax = axes[i]
    true_vals = val_y_true[:, i]
    pred_vals = val_y_pred[:, i]
    r2 = r2_score(true_vals, pred_vals)
    rmse = np.sqrt(mean_squared_error(true_vals, pred_vals))
    
    # Scatter plot of predictions vs ground truth
    ax.scatter(true_vals, pred_vals, alpha=0.35, color=colors[i], edgecolors="none", s=25)
    
    # Ideal y = x line
    min_v, max_v = MIN_BOUNDS[i], MAX_BOUNDS[i]
    ax.plot([min_v, max_v], [min_v, max_v], 'k--', lw=1.8, label="Ideal Estimate ($y=x$)")
    
    ax.set_xlim([min_v, max_v])
    ax.set_ylim([min_v, max_v])
    ax.set_xlabel(f"True {param_names[i]}", fontsize=11)
    ax.set_ylabel(f"Estimated {param_names[i]}", fontsize=11)
    ax.set_title(f"{param_names[i]} (R² = {r2:.3f}, RMSE = {rmse:.1f} {units[i]})", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

plt.suptitle("Layer 3: CNN-LSTM Kinematics Regression with PINN Loss", fontsize=14, fontweight="bold")
plt.tight_layout()
output_fig = "Layer3_Kinematics_Regression.png"
plt.savefig(output_fig, dpi=300)
print(f"\nPlots successfully saved to '{output_fig}'!")
plt.show()

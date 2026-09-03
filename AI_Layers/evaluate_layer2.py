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
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report

# -------------------------------------------------------------
# 1. LOAD DATASET (Validation Split)
# -------------------------------------------------------------
if not os.path.exists("layer2_auth.npz"):
    raise FileNotFoundError("layer2_auth.npz not found!")

data = np.load("layer2_auth.npz")
X = data["X"]
y = data["y"]

# Same random split seed used during training
np.random.seed(42)
n_samples = len(X)
n_train = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)
val_idx = indices[n_train:]

val_x = torch.tensor(X[val_idx], dtype=torch.float32).unsqueeze(1)
val_y = y[val_idx]

# -------------------------------------------------------------
# 2. MODEL ARCHITECTURE
# -------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(channels)
        self.act = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size=7, padding=3)
        self.bn2 = nn.BatchNorm1d(channels)
        
    def forward(self, x):
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.act(out + residual)

class WakeAuthenticator1DCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=15, stride=2, padding=7),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2)
        )
        self.res1 = ResidualBlock1D(32)
        
        self.stage2 = nn.Sequential(
            nn.Conv1d(32, 64, kernel_size=11, stride=2, padding=5),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool1d(2)
        )
        self.res2 = ResidualBlock1D(64)
        
        self.stage3 = nn.Sequential(
            nn.Conv1d(64, 128, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.stage2(x)
        x = self.res2(x)
        x = self.stage3(x)
        return self.head(x)

# -------------------------------------------------------------
# 3. RUN EVALUATION
# -------------------------------------------------------------
print("Loading trained model weights ('layer2_authenticator.pth')...")
device = torch.device("cpu")
model = WakeAuthenticator1DCNN().to(device)

if not os.path.exists("layer2_authenticator.pth"):
    raise FileNotFoundError("Please let 'python3 train_layer2.py' finish first to create 'layer2_authenticator.pth'.")

model.load_state_dict(torch.load("layer2_authenticator.pth", map_location=device))
model.eval()

print("Evaluating on 800 unseen test signals...")
with torch.no_grad():
    logits = model(val_x)
    probs = torch.sigmoid(logits).numpy().flatten()
    preds = (probs >= 0.5).astype(int)

# -------------------------------------------------------------
# 4. PRINT METRICS & GENERATE PLOTS
# -------------------------------------------------------------
print("\n" + "="*55)
print("       LAYER 2: WAKE AUTHENTICATOR TEST REPORT")
print("="*55)
print(classification_report(val_y, preds, target_names=["Decoy/Spoof/Noise", "Genuine Wake"]))

fpr, tpr, _ = roc_curve(val_y, probs)
roc_auc = auc(fpr, tpr)
print(f"Area Under ROC Curve (AUC): {roc_auc:.4f}\n")

# Plot ROC Curve and Confusion Matrix
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# ROC Curve
ax1.plot(fpr, tpr, color="crimson", lw=2.5, label=f"1D Res-CNN (AUC = {roc_auc:.4f})")
ax1.plot([0, 1], [0, 1], color="navy", lw=1.5, linestyle="--", label="Random Chance")
ax1.set_xlim([0.0, 1.0])
ax1.set_ylim([0.0, 1.05])
ax1.set_xlabel("False Alarm Rate (False Positives)", fontsize=11)
ax1.set_ylabel("Detection Probability (True Positives)", fontsize=11)
ax1.set_title("Layer 2 ROC Curve (Stealth Wake Detection)", fontsize=12)
ax1.legend(loc="lower right")
ax1.grid(True, alpha=0.3)

# Confusion Matrix
cm = confusion_matrix(val_y, preds)
im = ax2.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax2)
ax2.set_title("Confusion Matrix (800 Test Samples)", fontsize=12)
tick_marks = np.arange(2)
ax2.set_xticks(tick_marks)
ax2.set_xticklabels(["Spoof/Noise", "Genuine Wake"])
ax2.set_yticks(tick_marks)
ax2.set_yticklabels(["Spoof/Noise", "Genuine Wake"])

for i in range(2):
    for j in range(2):
        ax2.text(j, i, format(cm[i, j], 'd'),
                 ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black",
                 fontsize=14, fontweight="bold")

ax2.set_ylabel("True Ground Truth", fontsize=11)
ax2.set_xlabel("AI Prediction", fontsize=11)

plt.tight_layout()
plt.savefig("Layer2_ROC_ConfusionMatrix.png", dpi=300)
print("Successfully generated and saved figure to 'Layer2_ROC_ConfusionMatrix.png'!")
plt.show()
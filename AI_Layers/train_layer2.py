# =====================================================================
# MAC STABILITY FIX (Must be at the very top before any other imports)
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# ---------------------------------------------------------------------
# 1. LOAD DATASET
# ---------------------------------------------------------------------
print("Loading Layer 2 dataset...")
if not os.path.exists("layer2_auth.npz"):
    raise FileNotFoundError("layer2_auth.npz not found! Run 'python3 wake_data_generator.py' first.")

data = np.load("layer2_auth.npz")
X = data["X"]  # Shape: (4000, 4000)
y = data["y"]  # Shape: (4000,), 1 = Genuine Wake, 0 = Spoof/Decoy/Noise

# PyTorch Conv1D format: (Batch_Size, Channels=1, Time_Steps=4000)
X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)

# 80% Train / 20% Validation Split
np.random.seed(42)
torch.manual_seed(42)

n_samples = len(X)
n_train = int(0.8 * n_samples)
indices = np.random.permutation(n_samples)

train_idx, val_idx = indices[:n_train], indices[n_train:]

train_loader = DataLoader(
    TensorDataset(X_tensor[train_idx], y_tensor[train_idx]),
    batch_size=64,
    shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_tensor[val_idx], y_tensor[val_idx]),
    batch_size=64,
    shuffle=False
)

print(f"Data ready: {len(train_idx)} training samples, {len(val_idx)} validation samples.")

# ---------------------------------------------------------------------
# 2. DEFINE 1D RESIDUAL CNN ARCHITECTURE
# ---------------------------------------------------------------------
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

# ---------------------------------------------------------------------
# 3. TRAINING LOOP
# ---------------------------------------------------------------------
device = torch.device("cpu")  # CPU is rock-solid and takes only ~3 seconds
print(f"Training on: {device}")

model = WakeAuthenticator1DCNN().to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15)

epochs = 15
print("\nStarting Training...")

for epoch in range(1, epochs + 1):
    model.train()
    train_loss = 0.0
    correct = 0
    total = 0
    
    for batch_x, batch_y in train_loader:
        batch_x, batch_y = batch_x.to(device), batch_y.to(device)
        
        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * len(batch_y)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += len(batch_y)
        
    scheduler.step()
    train_acc = (correct / total) * 100
    train_loss = train_loss / total
    
    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for batch_x, batch_y in val_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            val_loss += loss.item() * len(batch_y)
            preds = (torch.sigmoid(logits) >= 0.5).float()
            val_correct += (preds == batch_y).sum().item()
            val_total += len(batch_y)
            
    val_acc = (val_correct / val_total) * 100
    val_loss = val_loss / val_total
    
    print(f"Epoch [{epoch:02d}/{epochs:02d}] "
          f"| Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
          f"| Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")

# Save model checkpoint
torch.save(model.state_dict(), "layer2_authenticator.pth")
print("\nSuccess: Model weights saved to 'layer2_authenticator.pth'!")
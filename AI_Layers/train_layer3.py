# =====================================================================
# MAC STABILITY FIX (Must be at the very top)
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader

# Physics normalization bounds (scales all parameters to [0, 1] for stable training)
MIN_BOUNDS = np.array([200.0, 100.0, 500.0, 0.0], dtype=np.float32)
MAX_BOUNDS = np.array([600.0, 5000.0, 5000.0, 2 * np.pi], dtype=np.float32)

# -------------------------------------------------------------
# 1. LOAD & NORMALIZE DATASET
# -------------------------------------------------------------
def load_layer3_data(data_path="layer3_kin.npz"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found! Run 'python3 wake_data_generator.py' first.")

    data = np.load(data_path)
    X = data["X"]  # Shape: (4000, 4000)
    Y = data["Y"]  # Shape: (4000, 4) -> [v, alt, CPA, heading]

    Y_norm = (Y - MIN_BOUNDS) / (MAX_BOUNDS - MIN_BOUNDS)

    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)
    Y_tensor = torch.tensor(Y_norm, dtype=torch.float32)

    # 80/20 Train/Val Split
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = len(X)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)

    train_idx, val_idx = indices[:n_train], indices[n_train:]
    return X_tensor, Y_tensor, train_idx, val_idx

# -------------------------------------------------------------
# 2. HYBRID CNN-LSTM ARCHITECTURE
# -------------------------------------------------------------
class KinematicsCNNLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        # 1D CNN Front-End: Extracts local high-frequency vortex shedding features
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
        # Sequence length is downsampled: 4000 -> 1000 -> 250 -> 125
        
        # Bi-directional LSTM Backbone: Tracks the temporal envelope slope
        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # Regression Head for [v, alt, CPA, heading]
        self.regressor = nn.Sequential(
            nn.Linear(64 * 2, 64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),
            nn.Linear(64, 4),
            nn.Sigmoid()  # Constrains predictions strictly within [0, 1] normalized bounds
        )

    def forward(self, x):
        feat = self.conv_stem(x)         # Shape: (B, 128, 125)
        feat = feat.permute(0, 2, 1)     # Shape: (B, 125, 128) for LSTM
        lstm_out, _ = self.lstm(feat)    # Shape: (B, 125, 128)
        
        # Pool across time to summarize trajectory
        pooled = torch.mean(lstm_out, dim=1)
        out = self.regressor(pooled)
        return out

# -------------------------------------------------------------
# 3. PHYSICS-INFORMED LOSS FUNCTION
# -------------------------------------------------------------
class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_phys=0.1):
        super().__init__()
        self.mse = nn.SmoothL1Loss()
        self.lambda_phys = lambda_phys

    def forward(self, pred_norm, target_norm):
        # 1. Standard supervised loss on kinematics parameters
        data_loss = self.mse(pred_norm, target_norm)
        
        # 2. Physics consistency penalty:
        # Denormalize predictions to physical units
        min_b = torch.tensor(MIN_BOUNDS, device=pred_norm.device)
        max_b = torch.tensor(MAX_BOUNDS, device=pred_norm.device)
        pred_phys = pred_norm * (max_b - min_b) + min_b
        target_phys = target_norm * (max_b - min_b) + min_b
        
        v_pred, alt_pred, cpa_pred = pred_phys[:, 0], pred_phys[:, 1], pred_phys[:, 2]
        v_true, alt_true, cpa_true = target_phys[:, 0], target_phys[:, 1], target_phys[:, 2]
        
        # Quasi-static envelope characteristic width ~ CPA / v
        # Physically, the temporal duration of pass-by scales as CPA/v
        t_width_pred = cpa_pred / (v_pred + 1e-6)
        t_width_true = cpa_true / (v_true + 1e-6)
        
        envelope_physics_loss = self.mse(t_width_pred / 10.0, t_width_true / 10.0)
        
        total_loss = data_loss + self.lambda_phys * envelope_physics_loss
        return total_loss, data_loss, envelope_physics_loss

# -------------------------------------------------------------
# 4. TRAINING LOOP
# -------------------------------------------------------------
def train_model():
    print("Loading Layer 3 dataset...")
    X_tensor, Y_tensor, train_idx, val_idx = load_layer3_data()

    train_loader = DataLoader(
        TensorDataset(X_tensor[train_idx], Y_tensor[train_idx]),
        batch_size=64,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_tensor[val_idx], Y_tensor[val_idx]),
        batch_size=64,
        shuffle=False
    )

    print(f"Data ready: {len(train_idx)} training samples, {len(val_idx)} validation samples.")

    device = torch.device("cpu")
    print(f"Training on: {device}")

    model = KinematicsCNNLSTM().to(device)
    criterion = PhysicsInformedLoss(lambda_phys=0.15)
    optimizer = optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20)

    epochs = 20
    print("\nStarting Training Layer 3 (CNN-LSTM + PINN)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss, d_loss, p_loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss, _, _ = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_y)
                
        val_loss = val_loss / len(val_loader.dataset)
        
        if epoch % 2 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

    # Save checkpoint
    torch.save(model.state_dict(), "layer3_kinematics.pth")
    print("\nSuccess: Model weights saved to 'layer3_kinematics.pth'!")

if __name__ == "__main__":
    train_model()

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

# -------------------------------------------------------------
# 1. LOAD & NORMALIZE GRAPH DATASET
# -------------------------------------------------------------
def load_layer4_data(data_path="layer4_graph.npz"):
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"{data_path} not found! Run 'python3 wake_data_generator.py' first.")

    data = np.load(data_path)
    nodes = data["nodes"]  # Shape: (3000, 8, 4) -> 8 sensors, features: [sx, sy, |B|, snr]
    Y = data["Y"]          # Shape: (3000, 3) -> [X_target, Y_target, Z_target] in meters
    sensor_pos = data["pos"]  # (8, 2)

    # Normalization for fast, stable training
    # Node features: [sx/1e3, sy/1e3, B/100, snr/20]
    nodes_norm = nodes.copy()
    nodes_norm[:, :, 2] /= 100.0   # Scale magnetic field
    nodes_norm[:, :, 3] /= 20.0    # Scale SNR

    # Target 3D coordinates in km: (X, Y, Z) / 1000
    Y_km = Y / 1000.0

    X_tensor = torch.tensor(nodes_norm, dtype=torch.float32)
    Y_tensor = torch.tensor(Y_km, dtype=torch.float32)

    # 80/20 Train/Validation Split
    np.random.seed(42)
    torch.manual_seed(42)

    n_samples = len(nodes)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)

    train_idx, val_idx = indices[:n_train], indices[n_train:]
    return X_tensor, Y_tensor, train_idx, val_idx, sensor_pos

# -------------------------------------------------------------
# 2. GRAPHSAGE / MESSAGE PASSING GNN ARCHITECTURE
# -------------------------------------------------------------
class GraphSAGELayer(nn.Module):
    """Inductive GraphSAGE layer aggregating spatial messages from neighboring sensors."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.self_linear = nn.Linear(in_features, out_features)
        self.neighbor_linear = nn.Linear(in_features, out_features)
        self.act = nn.LeakyReLU(0.1)
        self.norm = nn.LayerNorm(out_features)

    def forward(self, h):
        # h shape: (Batch_Size, Num_Nodes=8, In_Features)
        # Mean aggregation over all sensor nodes
        mean_neighbors = torch.mean(h, dim=1, keepdim=True).expand_as(h)
        out = self.self_linear(h) + self.neighbor_linear(mean_neighbors)
        return self.norm(self.act(out))

class ArrayFusionGraphSAGE(nn.Module):
    def __init__(self, node_in_dim=4, hidden_dim=64):
        super().__init__()
        # Graph Message Passing Layers
        self.sage1 = GraphSAGELayer(node_in_dim, hidden_dim)
        self.sage2 = GraphSAGELayer(hidden_dim, hidden_dim)
        self.sage3 = GraphSAGELayer(hidden_dim, hidden_dim)
        
        # Global Attention Pooling: Learns which sensors have strongest SNR
        self.attention_weights = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.Tanh(),
            nn.Linear(32, 1)
        )
        
        # 3D Position Regression Head: outputs [X, Y, Z] in km
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1),
            nn.Linear(32, 3)
        )

    def forward(self, x):
        # x shape: (B, 8, 4)
        h1 = self.sage1(x)
        h2 = self.sage2(h1) + h1  # Residual skip connection
        h3 = self.sage3(h2) + h2
        
        # Attention Pooling
        attn = torch.softmax(self.attention_weights(h3), dim=1)  # (B, 8, 1)
        graph_embedding = torch.sum(attn * h3, dim=1)            # (B, hidden_dim)
        
        # 3D target coordinates in km
        target_3d = self.decoder(graph_embedding)                # (B, 3)
        return target_3d

# -------------------------------------------------------------
# 3. TRAINING LOOP
# -------------------------------------------------------------
def train_model():
    print("Loading Layer 4 dataset...")
    X_tensor, Y_tensor, train_idx, val_idx, sensor_pos = load_layer4_data()

    train_loader = DataLoader(
        TensorDataset(X_tensor[train_idx], Y_tensor[train_idx]),
        batch_size=32,
        shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_tensor[val_idx], Y_tensor[val_idx]),
        batch_size=32,
        shuffle=False
    )

    print(f"Data ready: {len(train_idx)} training graphs, {len(val_idx)} validation graphs.")

    device = torch.device("cpu")
    print(f"Training on: {device}")

    model = ArrayFusionGraphSAGE(node_in_dim=4, hidden_dim=64).to(device)
    criterion = nn.SmoothL1Loss()  # Huber loss for robust localization
    optimizer = optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=25)

    epochs = 25
    print("\nStarting Training Layer 4 (GraphSAGE 8-Node Fusion)...")

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            preds = model(batch_x)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * len(batch_y)
            
        scheduler.step()
        train_loss = total_loss / len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_err_m = []
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                preds = model(batch_x)
                loss = criterion(preds, batch_y)
                val_loss += loss.item() * len(batch_y)
                
                # 3D distance error in meters (convert km back to m)
                err = torch.norm((preds - batch_y) * 1000.0, dim=1)
                val_err_m.extend(err.cpu().numpy())
                
        val_loss = val_loss / len(val_loader.dataset)
        mean_3d_err = np.mean(val_err_m)
        
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f} | Mean 3D Error: {mean_3d_err:.1f} meters")

    # Save model checkpoint
    torch.save(model.state_dict(), "layer4_graph_fusion.pth")
    print("\nSuccess: Model saved to 'layer4_graph_fusion.pth'!")

if __name__ == "__main__":
    train_model()

# =====================================================================
# MAC STABILITY FIX
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import numpy as np
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score, mean_squared_error
from train_layer4 import ArrayFusionGraphSAGE, load_layer4_data

# -------------------------------------------------------------
# 1. LOAD VALIDATION DATASET
# -------------------------------------------------------------
print("Loading Layer 4 validation dataset...")
X_tensor, Y_tensor, train_idx, val_idx, sensor_pos = load_layer4_data()

raw_data = np.load("layer4_graph.npz")
val_y_true = raw_data["Y"][val_idx]  # Shape: (600, 3) in meters

# -------------------------------------------------------------
# 2. LOAD TRAINED MODEL
# -------------------------------------------------------------
print("Loading trained Layer 4 model weights ('layer4_graph_fusion.pth')...")
device = torch.device("cpu")
model = ArrayFusionGraphSAGE(node_in_dim=4, hidden_dim=64).to(device)

if not os.path.exists("layer4_graph_fusion.pth"):
    raise FileNotFoundError("Run 'python3 train_layer4.py' first to train and save 'layer4_graph_fusion.pth'!")

model.load_state_dict(torch.load("layer4_graph_fusion.pth", map_location=device))
model.eval()

val_x = X_tensor[val_idx].to(device)
with torch.no_grad():
    preds_km = model(val_x).cpu().numpy()

# Convert predictions from km back to physical meters
val_y_pred = preds_km * 1000.0

# -------------------------------------------------------------
# 3. COMPUTE 3D LOCALIZATION METRICS
# -------------------------------------------------------------
error_3d = np.linalg.norm(val_y_true - val_y_pred, axis=1)  # Euclidean 3D error [m]
error_xy = np.linalg.norm(val_y_true[:, :2] - val_y_pred[:, :2], axis=1)  # Horizontal error [m]
error_z = np.abs(val_y_true[:, 2] - val_y_pred[:, 2])  # Altitude error [m]

coords = ["X Position (m)", "Y Position (m)", "Altitude Z (m)"]

print("\n" + "=" * 65)
print("       LAYER 4: 8-SENSOR ARRAY FUSION LOCALIZATION REPORT")
print("=" * 65)
for i in range(3):
    rmse = np.sqrt(mean_squared_error(val_y_true[:, i], val_y_pred[:, i]))
    r2 = r2_score(val_y_true[:, i], val_y_pred[:, i])
    print(f"  {coords[i]:<18} | RMSE: {rmse:>8.2f} m | R² Score: {r2:>6.4f}")

print("-" * 65)
print(f"  Mean 3D Euclidean Error : {np.mean(error_3d):>8.2f} m")
print(f"  Median 3D Error         : {np.median(error_3d):>8.2f} m")
print(f"  90th Percentile 3D Error: {np.percentile(error_3d, 90):>8.2f} m")
print(f"  Mean Horizontal (XY) Err: {np.mean(error_xy):>8.2f} m")
print(f"  Mean Altitude (Z) Error : {np.mean(error_z):>8.2f} m")
print("=" * 65)

# -------------------------------------------------------------
# 4. PLOT 4-PANEL LOCALIZATION FIGURE
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(13, 11))

# Panel 1: Top-Down Array & Tracking Map (First 40 test samples for clarity)
ax1 = axes[0, 0]
n_plot = min(40, len(val_idx))
ax1.scatter(sensor_pos[:, 0] / 1e3, sensor_pos[:, 1] / 1e3, color="crimson", marker="^", s=100, label="Quantum Magnetometers (8-Node Array)", zorder=5)
circle = plt.Circle((0, 0), 2.0, color="crimson", fill=False, linestyle="--", alpha=0.5, label="Array Perimeter (2 km radius)")
ax1.add_patch(circle)

ax1.scatter(val_y_true[:n_plot, 0] / 1e3, val_y_true[:n_plot, 1] / 1e3, color="navy", s=35, label="True Aircraft Position", alpha=0.8)
ax1.scatter(val_y_pred[:n_plot, 0] / 1e3, val_y_pred[:n_plot, 1] / 1e3, color="forestgreen", marker="x", s=45, label="GNN Estimated Position")

for j in range(n_plot):
    ax1.plot([val_y_true[j, 0] / 1e3, val_y_pred[j, 0] / 1e3],
             [val_y_true[j, 1] / 1e3, val_y_pred[j, 1] / 1e3],
             color="gray", linestyle=":", alpha=0.6)

ax1.set_xlabel("X Distance (km)", fontsize=11)
ax1.set_ylabel("Y Distance (km)", fontsize=11)
ax1.set_title("2D Array Plane Localization Tracking", fontsize=12)
ax1.legend(loc="upper right", fontsize=8)
ax1.grid(True, alpha=0.3)
ax1.set_xlim([-5, 5])
ax1.set_ylim([-5, 5])

# Panel 2: 3D Euclidean Error Distribution
ax2 = axes[0, 1]
ax2.hist(error_3d, bins=30, color="#1f77b4", edgecolor="black", alpha=0.7)
ax2.axvline(np.mean(error_3d), color="red", linestyle="--", lw=2, label=f"Mean 3D Error: {np.mean(error_3d):.1f} m")
ax2.axvline(np.median(error_3d), color="orange", linestyle="-.", lw=2, label=f"Median: {np.median(error_3d):.1f} m")
ax2.set_xlabel("3D Euclidean Error (meters)", fontsize=11)
ax2.set_ylabel("Validation Sample Count", fontsize=11)
ax2.set_title("3D Position Error Distribution", fontsize=12)
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True, alpha=0.3)

# Panel 3: Altitude Ground Truth vs Prediction
ax3 = axes[1, 0]
r2_z = r2_score(val_y_true[:, 2], val_y_pred[:, 2])
ax3.scatter(val_y_true[:, 2] / 1e3, val_y_pred[:, 2] / 1e3, alpha=0.4, color="teal", s=25, edgecolors="none")
ax3.plot([0.1, 5.0], [0.1, 5.0], "k--", lw=1.8, label="Ideal ($y=x$)")
ax3.set_xlabel("True Altitude (km)", fontsize=11)
ax3.set_ylabel("Estimated Altitude (km)", fontsize=11)
ax3.set_title(f"Altitude Estimation (R² = {r2_z:.3f})", fontsize=12)
ax3.legend(loc="upper left")
ax3.grid(True, alpha=0.3)

# Panel 4: Coordinate-wise RMSE Comparison Bar Chart
ax4 = axes[1, 1]
rmses = [np.sqrt(mean_squared_error(val_y_true[:, i], val_y_pred[:, i])) for i in range(3)] + [np.mean(error_3d)]
labels = ["X Error", "Y Error", "Altitude Error", "Total 3D Error"]
bars = ax4.bar(labels, rmses, color=["#4e79a7", "#f28e2b", "#e15759", "#76b7b2"], edgecolor="black", width=0.55)
for bar in bars:
    yval = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width() / 2.0, yval + 10, f"{yval:.1f} m", ha="center", va="bottom", fontweight="bold", fontsize=10)

ax4.set_ylabel("Root Mean Squared Error (meters)", fontsize=11)
ax4.set_title("Array Fusion Accuracy Summary", fontsize=12)
ax4.grid(axis="y", alpha=0.3)

plt.suptitle("Layer 4: Multi-Node Quantum Magnetometer Array Fusion (GraphSAGE)", fontsize=14, fontweight="bold")
plt.tight_layout()
output_img = "Layer4_Array_Localization.png"
plt.savefig(output_img, dpi=300)
print(f"\nPlots successfully saved to '{output_img}'!")
plt.show()

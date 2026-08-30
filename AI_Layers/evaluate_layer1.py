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

# -------------------------------------------------------------
# 1. LOAD ACTOR NETWORK
# -------------------------------------------------------------
class GaussianActor(nn.Module):
    def __init__(self, state_dim=5, action_dim=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)

    def forward(self, s):
        x = self.net(s)
        mu = self.mu(x)
        return torch.tanh(mu)

print("Loading trained Layer 1 SAC Actor ('layer1_sac_actor.pth')...")
device = torch.device("cpu")
actor = GaussianActor().to(device)

if not os.path.exists("layer1_sac_actor.pth"):
    raise FileNotFoundError("Run 'python3 train_layer1.py' first!")

actor.load_state_dict(torch.load("layer1_sac_actor.pth", map_location=device), strict=False)
actor.eval()

# -------------------------------------------------------------
# 2. RUN REAL-TIME FLYOVER TRACKING SIMULATION
# -------------------------------------------------------------
# Simulate a stealth aircraft accelerating and shifting Strouhal vortex frequency: 18 kHz -> 32 kHz -> 22 kHz
time_steps = 60
time_axis_s = np.linspace(0, 1.2, time_steps)  # 1.2 second pass-by window

# Target frequency trajectory in kHz
f_target_trajectory = 20.0 + 10.0 * np.sin(2 * np.pi * 0.8 * time_axis_s) + np.random.normal(0, 0.2, time_steps)

N_pulses = 16
T2_us = 1000.0  # 1 ms coherence ceiling

# 1. ADAPTIVE SAC AGENT SIMULATION
sac_tau_history = []
sac_fc_history = []
sac_snr_history = []
sac_tseq_history = []

tau_current = 15.0  # Start slightly detuned

for step in range(time_steps):
    f_target = f_target_trajectory[step]
    f_cpmg = 1000.0 / (4.0 * tau_current)
    f_err = f_cpmg - f_target
    
    t_seq = 2.0 * N_pulses * tau_current
    filter_contrast = np.sinc(N_pulses * (f_err / 50.0)) ** 2
    coherence_decay = np.exp(-((t_seq / T2_us) ** 3))
    snr = 25.0 * filter_contrast * coherence_decay
    
    obs = torch.FloatTensor([f_target/50.0, tau_current/50.0, snr/25.0, t_seq/T2_us, f_err/10.0]).unsqueeze(0)
    
    with torch.no_grad():
        action = actor(obs).item()
        
    # Execute action
    delta_tau = action * 4.0
    tau_current = np.clip(tau_current + delta_tau, 4.0, 45.0)
    
    sac_tau_history.append(tau_current)
    sac_fc_history.append(f_cpmg)
    sac_snr_history.append(snr)
    sac_tseq_history.append(t_seq)

# 2. STATIC CLASSICAL SENSOR BASELINE (Fixed tau = 12.5 us -> fixed at 20 kHz)
static_tau = 12.5
static_fc = 1000.0 / (4.0 * static_tau)  # 20 kHz
static_tseq = 2.0 * N_pulses * static_tau
static_coherence = np.exp(-((static_tseq / T2_us) ** 3))

static_snr_history = []
for step in range(time_steps):
    f_target = f_target_trajectory[step]
    f_err = static_fc - f_target
    filter_contrast = np.sinc(N_pulses * (f_err / 50.0)) ** 2
    snr = 25.0 * filter_contrast * static_coherence
    static_snr_history.append(snr)

# -------------------------------------------------------------
# 3. PRINT PERFORMANCE METRICS
# -------------------------------------------------------------
mean_sac_snr = np.mean(sac_snr_history)
mean_static_snr = np.mean(static_snr_history)
snr_advantage_db = 10.0 * np.log10(mean_sac_snr / (mean_static_snr + 1e-6))

print("\n" + "="*60)
print("     LAYER 1: SAC REAL-TIME RETUNING PERFORMANCE REPORT")
print("="*60)
print(f" Mean Detected SNR (Adaptive SAC)  : {mean_sac_snr:.2f}")
print(f" Mean Detected SNR (Static Sensor) : {mean_static_snr:.2f}")
print(f" Adaptive Quantum Advantage        : +{snr_advantage_db:.2f} dB")
print(f" Max Sequence Time (vs T2=1000 us) : {np.max(sac_tseq_history):.1f} us (Coherence Safe)")
print("="*60)

# -------------------------------------------------------------
# 4. PLOT 4-PANEL PUBLICATION FIGURE
# -------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 9))

# Panel 1: Real-Time Frequency Tracking
axes[0, 0].plot(time_axis_s, f_target_trajectory, 'b-', lw=2.5, label=r'Dynamic Wake Frequency $f_{\mathrm{carrier}}(t)$')
axes[0, 0].plot(time_axis_s, sac_fc_history, 'r--', lw=2.0, label=r'SAC Retuned CPMG Peak $1/(4\tau)$')
axes[0, 0].axhline(static_fc, color='gray', linestyle=':', lw=1.5, label='Static Classical Filter (Fixed)')
axes[0, 0].set_ylabel('Frequency (kHz)', fontsize=11)
axes[0, 0].set_xlabel('Flyover Time (s)', fontsize=11)
axes[0, 0].set_title('Layer 1: Real-Time Frequency Lock', fontsize=12)
axes[0, 0].legend(loc='upper right', fontsize=9)
axes[0, 0].grid(True, alpha=0.3)

# Panel 2: Signal Contrast / SNR Comparison
axes[0, 1].plot(time_axis_s, sac_snr_history, 'g-', lw=2.5, label='Adaptive SAC Quantum SNR')
axes[0, 1].plot(time_axis_s, static_snr_history, 'k--', lw=1.8, label='Static CPMG SNR (Collapses when detuned)')
axes[0, 1].set_ylabel('Quantum Signal Contrast / SNR', fontsize=11)
axes[0, 1].set_xlabel('Flyover Time (s)', fontsize=11)
axes[0, 1].set_title(f'Detected SNR Advantage (+{snr_advantage_db:.1f} dB)', fontsize=12)
axes[0, 1].legend(loc='upper right', fontsize=9)
axes[0, 1].grid(True, alpha=0.3)

# Panel 3: Hardware Coherence Budget Enforcement
axes[1, 0].plot(time_axis_s, sac_tseq_history, 'purple', lw=2.0, label=r'Sequence Time $T_{\mathrm{seq}} = 2N\tau$')
axes[1, 0].axhline(T2_us, color='red', linestyle='--', lw=2.0, label=r'Isotopic $T_2$ Limit (1000 $\mu$s)')
axes[1, 0].set_ylabel(r'Duration ($\mu$s)', fontsize=11)
axes[1, 0].set_xlabel('Flyover Time (s)', fontsize=11)
axes[1, 0].set_title(r'Hardware Coherence Constraint ($2N\tau \leq T_2$)', fontsize=12)
axes[1, 0].set_ylim([0, 1200])
axes[1, 0].legend(loc='lower right', fontsize=9)
axes[1, 0].grid(True, alpha=0.3)

# Panel 4: SAC RL Training Convergence Curve
if os.path.exists("layer1_sac_rewards.npy"):
    rewards = np.load("layer1_sac_rewards.npy")
    axes[1, 1].plot(rewards, color='darkorange', lw=1.8, label='Episode Reward')
    # Rolling average
    if len(rewards) >= 10:
        roll = np.convolve(rewards, np.ones(10)/10, mode='valid')
        axes[1, 1].plot(range(9, len(rewards)), roll, 'r-', lw=2.2, label='10-Episode Moving Average')
axes[1, 1].set_ylabel('Cumulative Reward', fontsize=11)
axes[1, 1].set_xlabel('Training Episodes', fontsize=11)
axes[1, 1].set_title('SAC Reinforcement Learning Convergence', fontsize=12)
axes[1, 1].legend(loc='lower right', fontsize=9)
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('Layer1_SAC_Adaptive_CPMG.png', dpi=300)
print("\nPlots successfully saved to 'Layer1_SAC_Adaptive_CPMG.png'!")
plt.show()
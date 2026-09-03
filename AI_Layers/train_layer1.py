# =====================================================================
# MAC STABILITY FIX (Must be at the very top)
# =====================================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"

import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------------------------------------------
# 1. CUSTOM CPMG RETUNING REINFORCEMENT LEARNING ENVIRONMENT
# -------------------------------------------------------------
class CPMGRetuningEnv:
    """
    Quantum Dynamical Decoupling Environment.
    State:  [f_target / 50kHz, tau / 50us, SNR / 20, (2*N*tau)/T2, f_error / 10kHz]
    Action: continuous Delta_tau in [-5 us, +5 us]
    """
    def __init__(self, N_pulses=16, T2_us=1000.0):
        self.N_pulses = N_pulses
        self.T2_us = T2_us  # Isotopic purity hardware ceiling (1000 us)
        self.max_steps = 50
        self.reset()

    def reset(self):
        # Target wake carrier frequency between 10 kHz and 40 kHz
        self.f_target_khz = np.random.uniform(15.0, 35.0)
        # Target drift rate (aircraft accelerating or changing turbulence scale)
        self.f_drift = np.random.uniform(-0.4, 0.4)
        
        # Initial sensor pulse interval (tau in microseconds)
        self.tau = np.random.uniform(8.0, 25.0)
        self.step_count = 0
        return self._get_obs()

    def _get_obs(self):
        f_cpmg = 1000.0 / (4.0 * self.tau)  # CPMG center frequency in kHz
        f_err = (f_cpmg - self.f_target_khz)
        
        # Quantum filter response (Sinc-squared bandpass filter)
        delta_f = np.abs(f_err)
        filter_contrast = np.sinc(self.N_pulses * (delta_f / 50.0)) ** 2
        
        # Phase 1B Isotopic T2 decoherence envelope
        t_seq = 2.0 * self.N_pulses * self.tau
        coherence_decay = np.exp(-((t_seq / self.T2_us) ** 3))
        
        snr = 25.0 * filter_contrast * coherence_decay
        coherence_ratio = t_seq / self.T2_us
        
        obs = np.array([
            self.f_target_khz / 50.0,
            self.tau / 50.0,
            snr / 25.0,
            coherence_ratio,
            f_err / 10.0
        ], dtype=np.float32)
        return obs

    def step(self, action):
        # Action is in [-1, 1], scale to delta_tau in [-4 us, +4 us]
        delta_tau = float(action[0]) * 4.0
        self.tau = np.clip(self.tau + delta_tau, 4.0, 45.0)
        
        # Evolve target aircraft frequency over time
        self.f_target_khz = np.clip(self.f_target_khz + self.f_drift + np.random.normal(0, 0.1), 10.0, 45.0)
        self.step_count += 1
        
        # Physics calculations
        f_cpmg = 1000.0 / (4.0 * self.tau)
        f_err = np.abs(f_cpmg - self.f_target_khz)
        
        filter_contrast = np.sinc(self.N_pulses * (f_err / 50.0)) ** 2
        t_seq = 2.0 * self.N_pulses * self.tau
        coherence_decay = np.exp(-((t_seq / self.T2_us) ** 3))
        
        snr = 25.0 * filter_contrast * coherence_decay
        
        # Reward Function:
        # + High SNR / Contrast
        # - Frequency mismatch penalty
        # - Hard penalty for exceeding T2 hardware coherence limit
        reward = snr - 0.5 * (f_err ** 2) - 0.05 * (delta_tau ** 2)
        if t_seq > self.T2_us:
            reward -= 50.0  # Hardware boundary violation penalty
            
        done = self.step_count >= self.max_steps
        next_obs = self._get_obs()
        return next_obs, reward, done

# -------------------------------------------------------------
# 2. SOFT ACTOR-CRITIC (SAC) NETWORKS
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
        log_std = torch.clamp(self.log_std(x), -20, 2)
        std = torch.exp(log_std)
        return mu, std

    def sample(self, s):
        mu, std = self.forward(s)
        normal = torch.distributions.Normal(mu, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)
        # Enforcing Action Bounds log-prob adjustment
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob

class TwinCritic(nn.Module):
    def __init__(self, state_dim=5, action_dim=1):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, s, a):
        sa = torch.cat([s, a], dim=-1)
        return self.q1(sa), self.q2(sa)

# -------------------------------------------------------------
# 3. REPLAY BUFFER & TRAINING LOOP
# -------------------------------------------------------------
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = []
        self.capacity = capacity
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size=64):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = map(np.stack, zip(*batch))
        return (torch.FloatTensor(s), torch.FloatTensor(a),
                torch.FloatTensor(r).unsqueeze(1),
                torch.FloatTensor(ns), torch.FloatTensor(d).unsqueeze(1))

    def __len__(self):
        return len(self.buffer)

# -------------------------------------------------------------
# 4. TRAIN SAC AGENT
# -------------------------------------------------------------
print("Initializing Soft Actor-Critic (SAC) Agent...")
env = CPMGRetuningEnv()
actor = GaussianActor()
critic = TwinCritic()
critic_target = TwinCritic()
critic_target.load_state_dict(critic.state_dict())

actor_opt = optim.Adam(actor.parameters(), lr=1e-3)
critic_opt = optim.Adam(critic.parameters(), lr=1e-3)
buffer = ReplayBuffer()

# Automatic entropy tuning
target_entropy = -1.0
log_alpha = torch.zeros(1, requires_grad=True)
alpha_opt = optim.Adam([log_alpha], lr=1e-3)

gamma = 0.99
tau_polyak = 0.005
episodes = 200
reward_history = []

print("Training SAC Agent on dynamic frequency tracking...")

for ep in range(1, episodes + 1):
    state = env.reset()
    ep_reward = 0.0
    
    for step in range(env.max_steps):
        with torch.no_grad():
            s_t = torch.FloatTensor(state).unsqueeze(0)
            action, _ = actor.sample(s_t)
            action = action.squeeze(0).numpy()
            
        next_state, reward, done = env.step(action)
        buffer.push(state, action, reward, next_state, done)
        state = next_state
        ep_reward += reward
        
        # SAC Updates once buffer has sufficient warm-up transitions
        if len(buffer) > 256:
            s_b, a_b, r_b, ns_b, d_b = buffer.sample(64)
            alpha = log_alpha.exp()
            
            with torch.no_grad():
                next_a, next_log_pi = actor.sample(ns_b)
                q1_target, q2_target = critic_target(ns_b, next_a)
                min_q_target = torch.min(q1_target, q2_target) - alpha * next_log_pi
                y = r_b + (1 - d_b) * gamma * min_q_target
                
            q1, q2 = critic(s_b, a_b)
            critic_loss = nn.MSELoss()(q1, y) + nn.MSELoss()(q2, y)
            
            critic_opt.zero_grad()
            critic_loss.backward()
            critic_opt.step()
            
            curr_a, curr_log_pi = actor.sample(s_b)
            q1_pi, q2_pi = critic(s_b, curr_a)
            min_q_pi = torch.min(q1_pi, q2_pi)
            actor_loss = (alpha * curr_log_pi - min_q_pi).mean()
            
            actor_opt.zero_grad()
            actor_loss.backward()
            actor_opt.step()
            
            alpha_loss = -(log_alpha * (curr_log_pi + target_entropy).detach()).mean()
            alpha_opt.zero_grad()
            alpha_loss.backward()
            alpha_opt.step()
            
            # Target Critic soft update
            for param, target_param in zip(critic.parameters(), critic_target.parameters()):
                target_param.data.copy_(tau_polyak * param.data + (1 - tau_polyak) * target_param.data)
                
    reward_history.append(ep_reward)
    if ep % 25 == 0 or ep == 1:
        print(f"Episode [{ep:03d}/{episodes}] | Total Episode Reward: {ep_reward:.1f} | Buffer Size: {len(buffer)}")

# Save trained models
torch.save(actor.state_dict(), "layer1_sac_actor.pth")
np.save("layer1_sac_rewards.npy", np.array(reward_history))
print("\nSuccess: Trained SAC Actor saved to 'layer1_sac_actor.pth'!")
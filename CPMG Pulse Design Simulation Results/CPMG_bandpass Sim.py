"""
PHASE 2: CPMG Pulse Sequence Design and Filter Validation
---------------------------------------------------------
Objective: Simulate the NV center spin dynamics under a CPMG 
sequence to validate the bandpass filter frequency response.

Dependencies: 
- Phase 1B Output: Isotopic T2 limit = 1000 us (1 ms)

Methodology:
To achieve high-resolution frequency mapping without the prohibitive 
computational overhead of solving 160,000 Lindblad ODEs, this simulation 
uses exact analytical unitary propagators for the AC magnetic field 
phase accumulation. The Markovian T2 decoherence envelope is decoupled 
and applied analytically to the final state projection.

PHYSICS CORRECTION:
The target AC frequency is derived strictly from the Strouhal vortex-shedding 
relation (f = St * v / L), representing the high-frequency turbulent wake. 
Quasi-static dipole pass-by (f = 0.85 * v / CPA) is explicitly rejected for 
CPMG tuning, as dynamic decoupling actively destroys quasi-static DC signals.
"""

import numpy as np
import matplotlib.pyplot as plt
from qutip import basis, sigmax, sigmay, sigmaz

# ==========================================
# 1. HARDWARE BASELINE (STRICT PHASE 1B INPUT)
# ==========================================
T2_limit = 1000.0  # Isotopic purity boundary in microseconds

# ==========================================
# 2. QUTIP OPERATORS & SUBSPACE SETUP
# ==========================================
# Spin-1/2 subspace (ms=0, ms=+1) in the rotating frame
sz = sigmaz()
sx = sigmax()
sy = sigmay()

# Ideal, instantaneous microwave control pulses
U_pi2_y = (-1j * (np.pi / 4) * sy).expm()  # Pi/2 pulse on Y-axis
U_pi2_x = (-1j * (np.pi / 4) * sx).expm()  # Pi/2 pulse on X-axis (Projection)
U_pi_x  = (-1j * (np.pi / 2) * sx).expm()  # Pi pulse on X-axis (Refocusing)

# Initial state |0>
state_0 = basis(2, 0)

# ==========================================
# 3. UNITARY DYNAMICS AND CPMG SUPEROPERATOR
# ==========================================
def free_evolution_unitary(t_start, t_end, f_ac, B_amp):
    """
    Calculates the exact quantum phase accumulation from the target AC wake.
    Assumes B(t) = B_amp * cos(2 * pi * f_ac * t) to align with CPMG parity.
    The integral of cos(wt) is (1/w)*sin(wt).
    """
    omega = 2 * np.pi * f_ac
    
    # CORRECTED: Integration of a Cosine wave avoids the "even/odd" cancellation trap
    phase = (B_amp / omega) * (np.sin(omega * t_end) - np.sin(omega * t_start))
    
    return (-1j * (phase / 2) * sz).expm()

def run_cpmg_sequence(N_pulses, tau, f_ac, B_amp):
    """
    Executes the full CPMG sequence: Pi/2 -> (Wait tau -> Pi -> Wait tau)_N -> Pi/2
    """
    rho = U_pi2_y * state_0 * state_0.dag() * U_pi2_y.dag()
    t_current = 0.0
    
    for _ in range(N_pulses):
        # First free evolution (tau)
        U_tau1 = free_evolution_unitary(t_current, t_current + tau, f_ac, B_amp)
        rho = U_tau1 * rho * U_tau1.dag()
        t_current += tau
        
        # Microwave Pi pulse
        rho = U_pi_x * rho * U_pi_x.dag()
        
        # Second free evolution (tau)
        U_tau2 = free_evolution_unitary(t_current, t_current + tau, f_ac, B_amp)
        rho = U_tau2 * rho * U_tau2.dag()
        t_current += tau
        
    # Final projection for population readout
    rho = U_pi2_x * rho * U_pi2_x.dag()
    
    # Apply the Phase 1B Isotopic T2 decay envelope
    total_time = 2 * N_pulses * tau
    decoherence_multiplier = np.exp(-(total_time / T2_limit)**3) 
    
    return (rho * sz).tr().real * decoherence_multiplier

# ==========================================
# 4. HIGH-RESOLUTION SIMULATION SWEEP
# ==========================================
print("Executing Phase 2: Quantum Bandpass Filter Map...")
print(f"Hardware Constraint: T2 = {T2_limit} us (From Phase 1B)")

N_pulses = 16          
B_amp = 0.002  # Restricted to small-signal regime to prevent phase-wrapping artifacts

tau_array = np.linspace(5, 50, 150)        # Inter-pulse spacing (us)
freq_array = np.linspace(0.005, 0.1, 150)  # Target AC frequency (MHz)

signal_map = np.zeros((len(tau_array), len(freq_array)))

for i, tau in enumerate(tau_array):
    for j, f_ac in enumerate(freq_array):
        signal_map[i, j] = run_cpmg_sequence(N_pulses, tau, f_ac, B_amp)

print("Computation complete. Rendering thesis figures...")

# ==========================================
# 5. FIGURE GENERATION & VALIDATION
# ==========================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# ---- Panel 1: 1D Filter response (Targeting Strouhal Wake Frequency) ----
# KINEMATICS UPDATE: Deriving the 25 kHz target frequency from physical stealth parameters
St = 0.20           # Strouhal number for turbulent wake shedding
v_target = 250.0    # Target velocity in m/s (approx Mach 0.73)
L_char = 0.002      # Characteristic length of shear-layer micro-vortices (meters)

f_strouhal_hz = St * (v_target / L_char) # Yields 25,000 Hz (25 kHz)
target_freq = f_strouhal_hz / 1e6        # Convert to MHz for QuTiP time (us)

theoretical_tau = 1.0 / (4 * target_freq)
response_1d = [run_cpmg_sequence(N_pulses, t, target_freq, B_amp) for t in tau_array]

ax1.plot(tau_array, response_1d, 'b-', linewidth=2.5, label='Simulated NV Response')
ax1.axvline(theoretical_tau, color='r', linestyle='--', linewidth=2, label=rf'Theoretical Peak: $\tau = 1/(4f_c)$ = {theoretical_tau:.1f} $\mu$s')
ax1.set_title(f"CPMG Filter Response (Strouhal Wake: {target_freq*1000:.0f} kHz)", fontsize=14)
ax1.set_xlabel(r"Pulse Spacing $\tau$ ($\mu$s)", fontsize=12)
ax1.set_ylabel("Measured Contrast", fontsize=12)
ax1.grid(True, alpha=0.4)
ax1.legend()

# ---- Panel 2: 2D Heatmap and Analytical Validation ----
T_mesh, F_mesh = np.meshgrid(tau_array, freq_array, indexing='ij')
c = ax2.pcolormesh(F_mesh * 1000, T_mesh, signal_map, cmap='magma', shading='auto')

# Overlay the theoretical validation line
theory_f = np.linspace(10, 100, 100) # kHz
theory_t = 1000 / (4 * theory_f)     # us
ax2.plot(theory_f, theory_t, 'w--', linewidth=2.5, label=r'Validation: $f_c = 1/(4\tau)$')

ax2.set_title(r"$\tau$ vs Strouhal Frequency Map", fontsize=14)
ax2.set_xlabel("Target AC Magnetic Frequency (kHz)", fontsize=12)
ax2.set_ylabel(r"Pulse Spacing $\tau$ ($\mu$s)", fontsize=12)
ax2.set_ylim(5, 50)
ax2.legend(loc='upper right')
fig.colorbar(c, ax=ax2, label="Sensor Contrast")

plt.tight_layout()
plt.savefig('Phase2_Master_CPMG_Map_Kinematic.png', dpi=300)
plt.show()
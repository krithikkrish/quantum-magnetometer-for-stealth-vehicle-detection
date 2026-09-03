# ============================================================================
# AI LAYERS - SYNTHETIC WAKE DATA GENERATOR
# ----------------------------------------------------------------------------
# All four AI layers train on data produced here. There is no measured data,
# so we generate it from the Phase 3A wake physics.
#
# PHYSICS USED (consistent with Phase 3A / the Strouhal correction):
#   * Aircraft modelled as an effective magnetic dipole, m_eff [A.m^2].
#   * Field at the sensor falls as 1/r^3 (dipole near-field).
#   * Flyover geometry: r(t) = sqrt(CPA^2 + (v t)^2).
#   * TWO BANDS:
#       - slow PASS-BY ENVELOPE  (sub-Hz, quasi-static)  ~ v/CPA
#       - turbulent CARRIER  f = St * v / L  (St ~ 0.2)  <- the CPMG band
#     The measured signal is the envelope multiplying the carrier, plus noise.
#   * Noise: white sensor noise at the few-fT/rtHz floor + 1/f geomagnetic drift.
#
# OUTPUTS (three datasets):
#   layer2_auth.npz  -> X: windows, y: 1 = genuine wake, 0 = spoof/decoy/noise
#   layer3_kin.npz   -> X: windows, y: [v, altitude, CPA, heading]
#   layer4_graph.npz -> node features for an 8-sensor array, y: aircraft 3D pos
#
# RUN:  python wake_data_generator.py
# ============================================================================
import numpy as np

rng = np.random.default_rng(0)

# ---------------- constants / config ----------------
MU0   = 4*np.pi*1e-7
fT    = 1e-15
FS    = 200e3          # sample rate [Hz] (must resolve the ~25 kHz carrier)
WIN_S = 0.02           # window length [s] -> 4000 samples
NWIN  = int(FS*WIN_S)
B_FLOOR = 5*fT         # sensor noise floor (Phase 6 budget)
ST    = 0.2            # Strouhal number (vortex shedding

def dipole_B(m_eff, r):
    """On-axis magnetic dipole field magnitude [T] at range r [m]."""
    return MU0 * 2.0 * m_eff / (4.0*np.pi * r**3)

def flyover(v, alt, cpa, m_eff, t):
    """Slow pass-by envelope: range and field vs time about closest approach."""
    r = np.sqrt((v*t)**2 + cpa**2 + alt**2)
    return dipole_B(m_eff, r), r

def wake_window(v, alt, cpa, m_eff, L=0.0024, t0=0.0, snr_scale=1.0):
    """
    One measurement window: turbulent carrier modulated by the pass-by envelope,
    plus sensor white noise and 1/f drift.

    L = EDDY SCALE (metres), NOT wingspan.  IMPORTANT PHYSICS NOTE:
      f = St*v/L.  With L = wingspan (~15 m) this gives only ~4 Hz - far too slow
      for CPMG (tau would exceed T2).  The kHz CPMG band corresponds to FINE-SCALE
      turbulence in the wake's energy cascade: L ~ mm.  e.g. v=300 m/s, L=2.4 mm
      -> f = 25 kHz.  So the CPMG matched filter targets the small-eddy end of the
      cascade, and this must be stated explicitly in the paper.
    """
    t = t0 + np.arange(NWIN)/FS
    env, _ = flyover(v, alt, cpa, m_eff, t)

    f_carrier = ST * v / L                      # Strouhal carrier [Hz]
    phase = 2*np.pi*f_carrier*t + rng.uniform(0, 2*np.pi)

    # turbulence is broadband, not a pure tone: jitter the phase
    jitter = np.cumsum(rng.normal(0, 0.02, NWIN))
    carrier = np.cos(phase + jitter)

    sig = env * carrier * snr_scale
    white = rng.normal(0, B_FLOOR, NWIN)
    drift = np.cumsum(rng.normal(0, B_FLOOR*0.05, NWIN))   # 1/f-ish
    return (sig + white + drift).astype(np.float32), f_carrier

def spoof_window(kind=None):
    """Negative class: things that are NOT a genuine wake."""
    t = np.arange(NWIN)/FS
    k = kind if kind is not None else rng.integers(0, 4)
    if k == 0:                                   # pure tone, wrong statistics
        f = rng.uniform(1e3, 60e3)
        x = 50*B_FLOOR*np.cos(2*np.pi*f*t + rng.uniform(0, 2*np.pi))
    elif k == 1:                                 # noise only
        x = rng.normal(0, B_FLOOR, NWIN)
    elif k == 2:                                 # square-wave decoy
        f = rng.uniform(5e3, 40e3)
        x = 40*B_FLOOR*np.sign(np.cos(2*np.pi*f*t))
    else:                                        # amplitude-scaled replay (no jitter)
        f = rng.uniform(15e3, 35e3)
        x = 60*B_FLOOR*np.cos(2*np.pi*f*t) * np.exp(-t/ (WIN_S/2))
    return (x + rng.normal(0, B_FLOOR, NWIN)).astype(np.float32)

def sample_params():
    """Random but physically sensible flyover parameters (Phase 3A sweep ranges)."""
    return dict(
        v     = rng.uniform(200, 600),      # m/s
        alt   = rng.uniform(100, 5000),     # m
        cpa   = rng.uniform(500, 5000),     # m
        m_eff = 10**rng.uniform(2.5, 3.5),  # A.m^2 (assumption, swept)
        # EDDY SCALE (mm), not wingspan -> puts the carrier in the kHz CPMG band)
        L     = rng.uniform(0.0015, 0.005), # m  (1.5-5 mm eddies)
    )

# ============================================================================
# LAYER 2 - authentication (genuine wake vs spoof)
# ============================================================================
def make_layer2(n=4000):
    X, y = [], []
    for _ in range(n//2):
        p = sample_params()
        x, _ = wake_window(p['v'], p['alt'], p['cpa'], p['m_eff'], p['L'])
        X.append(x); y.append(1)
        X.append(spoof_window()); y.append(0)
    X = np.array(X); y = np.array(y, dtype=np.int64)
    # normalise each window (network sees shape, not absolute scale)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-20)
    return X.astype(np.float32), y

# ============================================================================
# LAYER 3 - kinematics regression (CNN-LSTM)
# ============================================================================
def make_layer3(n=4000):
    X, Y = [], []
    for _ in range(n):
        p = sample_params()
        heading = rng.uniform(0, 2*np.pi)
        x, _ = wake_window(p['v'], p['alt'], p['cpa'], p['m_eff'], p['L'])
        X.append(x)
        Y.append([p['v'], p['alt'], p['cpa'], heading])
    X = np.array(X); Y = np.array(Y, dtype=np.float32)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-20)
    return X.astype(np.float32), Y

# ============================================================================
# LAYER 4 - 8-node array fusion (GNN)
# ============================================================================
def make_layer4(n=3000, n_nodes=8, spacing=2000.0):
    """
    8 sensors on a ring of radius `spacing`. Each node sees the dipole field
    from its own distance. Labels = aircraft 3D position.
    node features: [x, y, |B| (fT), local SNR]
    """
    ang = np.linspace(0, 2*np.pi, n_nodes, endpoint=False)
    pos = np.stack([spacing*np.cos(ang), spacing*np.sin(ang)], 1)   # (8,2)

    NF, Y = [], []
    for _ in range(n):
        p = sample_params()
        ax = rng.uniform(-4000, 4000); ay = rng.uniform(-4000, 4000)
        az = p['alt']
        feats = []
        for (sx, sy) in pos:
            r = np.sqrt((ax-sx)**2 + (ay-sy)**2 + az**2)
            B = dipole_B(p['m_eff'], r)
            snr = B / B_FLOOR
            feats.append([sx/1e3, sy/1e3, B/fT, snr])
        NF.append(feats); Y.append([ax, ay, az])
    return (np.array(NF, dtype=np.float32),
            np.array(Y, dtype=np.float32),
            pos.astype(np.float32))

if __name__ == "__main__":
    print("generating Layer 2 (authentication)...")
    X2, y2 = make_layer2()
    np.savez_compressed("layer2_auth.npz", X=X2, y=y2)
    print("   X", X2.shape, " genuine:", int(y2.sum()), " spoof:", int((1-y2).sum()))

    print("generating Layer 3 (kinematics)...")
    X3, Y3 = make_layer3()
    np.savez_compressed("layer3_kin.npz", X=X3, Y=Y3)
    print("   X", X3.shape, " Y", Y3.shape, " [v, alt, cpa, heading]")

    print("generating Layer 4 (array fusion)...")
    NF, Y4, pos = make_layer4()
    np.savez_compressed("layer4_graph.npz", nodes=NF, Y=Y4, pos=pos)
    print("   nodes", NF.shape, " Y", Y4.shape, " sensor positions", pos.shape)

    # quick sanity readout
    p = sample_params()
    x, fc = wake_window(p['v'], p['alt'], p['cpa'], p['m_eff'], p['L'])
    print(f"\nsanity: v={p['v']:.0f} m/s, L={p['L']:.1f} m -> Strouhal carrier "
          f"= {fc/1e3:.2f} kHz  (CPMG band)")
    print("done. three .npz files written.")

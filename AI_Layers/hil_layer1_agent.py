import os
import sys
import time
import struct
import numpy as np

# Apply the same Mac crash fixes we used for the other AI layers
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
import torch
import torch.nn as nn

try:
    import serial
except ImportError:
    print("Error: pyserial not found. Please install it using: pip install pyserial")
    sys.exit(1)

# =====================================================================
# SAC ACTOR NETWORK DEFINITION
# =====================================================================
class GaussianActor(nn.Module):
    def __init__(self, state_dim=5, action_dim=1, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        self.mu = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mu = self.mu(x)
        # For HIL inference, we use the deterministic mean
        return torch.tanh(mu)

# =====================================================================
# HARDWARE-IN-THE-LOOP (HIL) CONTROLLER
# =====================================================================
def run_hil_loop(port_name, model_path):
    print("==================================================")
    print("   PHASE 4: FPGA HIL CPMG AI CONTROLLER")
    print("==================================================")
    
    # 1. Load the trained SAC AI Model
    device = torch.device("cpu")
    actor = GaussianActor().to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}. Make sure you are in the 'AI Layers' folder!")
        return
    
    actor.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    actor.eval()
    print(f"[AI] Loaded SAC Actor from {model_path}")

    # 2. Open the UART connection to the FPGA
    print(f"[UART] Attempting to connect to FPGA on {port_name} at 115200 baud...")
    try:
        ser = serial.Serial(port_name, 115200, timeout=1.0)
        ser.dtr = False
        ser.rts = False
        time.sleep(2) # Give the serial connection time to stabilize
        print("[UART] Connected successfully!\n")
    except Exception as e:
        print(f"Error opening serial port: {e}")
        print("Hint: On Mac, run 'ls /dev/cu.*' to find your ESP8266 port.")
        return

    # 3. Initial Hardware State
    f_target_kHz = 25.0
    tau_cycles = 500       # FPGA boot default: 500 cycles (10 us at 50 MHz)
    tau_us = tau_cycles / 50.0 
    N_pulses = 16
    T2_us = 1000.0

    print("--- STARTING REAL-TIME RETUNING LOOP ---")
    
    # KICKSTART: Send the initial tau to the FPGA in case it is waiting for us first
    print("[UART] Sending initial kickstart tau to FPGA...")
    high_byte = (tau_cycles >> 8) & 0xFF
    low_byte  = tau_cycles & 0xFF
    ser.write(bytes([high_byte, low_byte]))
    
    try:
        step = 0
        while True:
            # --- A. WAIT FOR FPGA TO SEND SNR ---
            # Flush the backlog of old bytes so we read the most recent SNR
            ser.reset_input_buffer()
            
            raw_byte = ser.read(1)
            if not raw_byte:
                print("[UART] Timeout waiting for FPGA. Is the FPGA running and wired correctly?")
                time.sleep(1)
                continue
                
            snr_val = int.from_bytes(raw_byte, byteorder='little')
            
            # Convert raw 0-255 back to an approximate SNR float for the AI
            snr_float = snr_val / 10.0 
            
            # --- B. PREPARE AI STATE ---
            f_current = 1000.0 / (4.0 * tau_us) # Approximation of center freq
            f_err = f_target_kHz - f_current
            t_seq = 2 * N_pulses * tau_us
            
            # State vector: [f_target/50, tau/50, snr/25, t_seq/T2, f_err/10]
            state = np.array([
                f_target_kHz / 50.0,
                tau_us / 50.0,
                snr_float / 25.0,
                t_seq / T2_us,
                f_err / 10.0
            ], dtype=np.float32)
            
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # --- C. GET AI ACTION ---
            with torch.no_grad():
                action = actor(state_tensor).item() # Output is [-1, +1]
            
            # Map action to delta_tau in microseconds [-4 us, +4 us]
            delta_tau_us = action * 4.0
            
            # Calculate new tau
            tau_us_new = tau_us + delta_tau_us
            tau_us_new = np.clip(tau_us_new, 2.0, 30.0) # Safety clamps
            
            # --- D. SEND NEW TAU TO FPGA ---
            # Convert tau_us back to 50 MHz clock cycles (1 us = 50 cycles)
            tau_cycles = int(tau_us_new * 50)
            
            # Send High Byte then Low Byte (exactly matching uart_rx.v logic)
            high_byte = (tau_cycles >> 8) & 0xFF
            low_byte  = tau_cycles & 0xFF
            ser.write(bytes([high_byte, low_byte]))
            
            # Log it
            print(f"Step {step:03d} | SNR: {snr_val:03d} | AI Action: {delta_tau_us:+.2f} us -> Sent FPGA Tau: {tau_cycles} cycles ({tau_us_new:.2f} us)")
            
            # Update local state for next loop
            tau_us = tau_us_new
            step += 1
            
            # Limit loop speed for testing
            time.sleep(0.1)

    except KeyboardInterrupt:
        print("\n[HIL] Interrupted by user. Closing connection.")
    finally:
        ser.close()

if __name__ == "__main__":
    import argparse
    # Default is a common Mac USB serial port, but you will override it with --port
    parser = argparse.ArgumentParser(description='Run HIL AI loop with FPGA.')
    parser.add_argument('--port', type=str, default='/dev/cu.usbserial-0001', 
                        help='Mac serial port of the ESP8266 (e.g. /dev/cu.SLAB_USBtoUART)')
    args = parser.parse_args()
    
    # Must run this script in the same folder as the PyTorch model
    model_file = "layer1_sac_actor.pth"
    run_hil_loop(args.port, model_file)
"""
VALIDATE EEGNET FEATURES (RAW TIME-SERIES)
==========================================
1. Loads the _eegnet_features.pkl files.
2. Checks Data Shape & Amplitude Scaling (Microvolts).
3. Plots Raw Waveforms (C3/C4).
4. Plots Average PSD to verify the 4-40Hz filter.
"""

import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(parent_dir, 'src'))

import pickle
import numpy as np
import matplotlib
try: matplotlib.use('TkAgg') 
except: pass
import matplotlib.pyplot as plt
from scipy import signal

from config import PROCESSED_DATA_DIR

# Standard Motor Channels (Approximate indices if names missing)
# C3 is roughly index 8-12, C4 roughly 22-26 in standard 64-chan setups
C3_IDX = 8 
C4_IDX = 22

def load_data(subject_id):
    sub_str = f"S{subject_id:03d}"
    # Note: We look in "processed_eegnet" folder, distinct from "processed"
    file_path = PROCESSED_DATA_DIR.parent / "processed_eegnet" / sub_str / f"{sub_str}_eegnet_features.pkl"
    
    if not file_path.exists():
        print(f"File not found: {file_path}")
        return None
    
    print(f"Loading {file_path.name}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)
    
    return data

def visualize_subject(subject_id):
    data = load_data(subject_id)
    if not data: return

    X = data['X'] # Shape: (Trials, Channels, TimePoints)
    y = data['y'] # Labels
    
    print(f"Data Shape: {X.shape}") 
    print(f"Labels: {np.unique(y, return_counts=True)}")
    print(f"Amplitude Stats: Min={X.min():.2f}, Max={X.max():.2f}, Mean={X.mean():.2f}")

    # 1. Amplitude Histogram
    # Check if we successfully converted to Microvolts (Values should be > 1.0)
    plt.figure(figsize=(10, 4))
    plt.hist(X.flatten(), bins=100, color='teal', alpha=0.7)
    plt.title(f"Amplitude Distribution (Microvolts) - S{subject_id:03d}")
    plt.xlabel("Amplitude (uV)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Raw Waveform Plot (Visual Sanity Check)
    # Plot C3 (Left) vs C4 (Right) for a single trial
    trial_idx = 0
    time_axis = np.linspace(0, 2.0, X.shape[2]) # 2 seconds
    
    plt.figure(figsize=(12, 6))
    plt.plot(time_axis, X[trial_idx, C3_IDX, :], label="C3 (Left Motor)", color='b', alpha=0.8)
    plt.plot(time_axis, X[trial_idx, C4_IDX, :], label="C4 (Right Motor)", color='r', alpha=0.6)
    plt.title(f"Raw EEG Trial #{trial_idx} (Label: {y[trial_idx]})")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (uV)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. Check Filter (PSD)
    # The graph should show energy ONLY between 4Hz and 40Hz.
    # If it's flat lines outside this range, your filter worked!
    freqs, psd = signal.welch(X, fs=160, axis=-1)
    mean_psd = np.mean(psd, axis=(0, 1)) # Average across all trials/channels

    plt.figure(figsize=(8, 5))
    plt.plot(freqs, 10 * np.log10(mean_psd), color='black')
    plt.title("Average Frequency Content (Filter Check)")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (dB)")
    plt.axvline(4, color='r', linestyle='--', label='4Hz Cutoff')
    plt.axvline(40, color='r', linestyle='--', label='40Hz Cutoff')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    while True:
        try:
            val = input("Enter Subject ID to visualize (1-109, or 'q'): ")
            if val.lower() == 'q': break
            visualize_subject(int(val))
        except Exception as e:
            print(f"Error: {e}")
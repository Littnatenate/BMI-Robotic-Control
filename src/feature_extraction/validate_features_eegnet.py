"""
EEGNET FEATURE VISUALIZATION (EDA)
==================================
Description:
    1. Loads the Raw Time-Series (.pkl) prepared for EEGNet.
    2. Check 1: Amplitude Histogram. Verifies data is in Microvolts (-100 to +100 uV).
    3. Check 2: Event-Related Potentials (ERP). Plots the Average Brainwave to ensure it's not noise.
"""

import os
import pickle
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg') 
except:
    pass
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# --- CONFIGURATION ---
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed_eegnet"

# Motor channels to focus on (C3=Left Motor, C4=Right Motor, Cz=Center)
# Indices depend on your channel list, but usually around 8-12 for standard 64-ch
INTERESTING_CHANNELS = [7, 9, 11] # Approximate indices for Fc5, Fc1, Fc2 (Adjust as needed)

# ==========================
# 1. Load Features
# ==========================
def load_features():
    while True:
        try:
            sub_input = input("\nEnter subject ID to analyze (1-109): ").strip()
            sub_id = int(sub_input)
            if 1 <= sub_id <= 109: break
            print("Please enter a number between 1 and 109.")
        except ValueError:
            print("Invalid input.")

    # Construct path
    folder_name = f"S{sub_id:03d}"
    file_name = f"{folder_name}_eegnet_features.pkl" 
    file_path = os.path.join(BASE_OUTPUT_PATH, folder_name, file_name)

    if not os.path.exists(file_path):
        print(f"\n[Error] File not found: {file_path}")
        print("Did you run 'extract_features_eegnet.py'?")
        return None, None

    print(f"\nLoading: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        X = data.get("X") # Shape: (Trials, Channels, Time)
        y = data.get("y")

        print("\n[Success] Data Loaded:")
        print(f"  X Shape: {X.shape} (Trials, Channels, Timepoints)")
        print(f"  y Shape: {y.shape}")
        return X, y

    except Exception as e:
        print(f"[Error] Corrupted file: {e}")
        return None, None

# ==========================
# 2. Check Amplitude (Microvolts)
# ==========================
def check_amplitude_range(X):
    """
    Ensures data is scaled correctly. 
    Raw EEG in Volts is tiny (0.00005). 
    EEGNet wants Microvolts (50.0).
    """
    print("\n[Info] Checking Signal Amplitude...")
    
    # Flatten all data to see global distribution
    flat_data = X.flatten()
    
    mean_val = np.mean(flat_data)
    std_val = np.std(flat_data)
    min_val = np.min(flat_data)
    max_val = np.max(flat_data)
    
    print(f"  Mean: {mean_val:.2f}")
    print(f"  Std Dev: {std_val:.2f}")
    print(f"  Range: {min_val:.2f} to {max_val:.2f}")
    
    # Heuristic check
    if std_val < 0.1:
        print("⚠️ WARNING: Values look extremely small. Did you multiply by 1e6?")
    elif std_val > 1000:
        print("⚠️ WARNING: Values look extremely large. Artifacts might be present.")
    else:
        print("✅ PASS: Amplitude looks like valid Microvolts (uV).")

    # Plot Histogram
    plt.figure(figsize=(10, 5))
    plt.hist(flat_data, bins=100, color='purple', alpha=0.7)
    plt.title(f"Global Amplitude Distribution (uV)\nRange: [{min_val:.1f}, {max_val:.1f}]")
    plt.xlabel("Microvolts (uV)")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.show()

# ==========================
# 3. Visualise Waveforms (ERP)
# ==========================
def plot_average_waveform(X, y):
    """
    Plots the 'Average' Left vs Right signal.
    If there is a brain signal, the lines should separate.
    """
    print("\n[Info] Plotting Average Waveforms (ERP)...")
    
    # Classes
    classes = np.unique(y)
    
    # Time axis (Assuming 160Hz, 2 seconds)
    time_points = X.shape[2]
    t = np.linspace(0, time_points/160.0, time_points)
    
    # Pick a channel to visualize (e.g., Channel 10 - likely Central)
    # We pick the channel with the highest variance (most activity)
    ch_var = np.var(X, axis=(0, 2))
    best_ch = np.argmax(ch_var)
    
    print(f"  Visualizing Channel Index: {best_ch} (Highest Variance)")

    plt.figure(figsize=(12, 6))
    
    colors = ['blue', 'red', 'green', 'orange']
    
    for i, cls in enumerate(classes):
        # Get all trials for this class
        trials = X[y == cls, best_ch, :]
        
        # Average them
        avg_wave = np.mean(trials, axis=0)
        
        # Plot
        plt.plot(t, avg_wave, label=f"Class {cls}", color=colors[i % len(colors)], linewidth=2)
        
        # Optional: Plot faint lines for individual trials
        # for tr in trials[:5]:
        #     plt.plot(t, tr, color=colors[i], alpha=0.1)

    plt.title(f"Average Event-Related Potential (Channel {best_ch})")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (uV)")
    plt.legend()
    plt.grid(True)
    plt.show()

# ==========================
# 4. Main
# ==========================
if __name__ == "__main__":
    X, y = load_features()
    
    if X is not None:
        while True:
            print("\n--- EEGNET VISUALIZATION MENU ---")
            print("1. Check Amplitudes (Is it uV?)")
            print("2. Check Waveforms (ERPs)")
            print("Q. Quit")
            
            choice = input("Selection: ").lower().strip()
            
            if choice == '1':
                check_amplitude_range(X)
            elif choice == '2':
                plot_average_waveform(X, y)
            elif choice == 'q':
                break
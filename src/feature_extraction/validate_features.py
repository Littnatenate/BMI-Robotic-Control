"""
VALIDATE SPECTROGRAMS (VISUAL INSPECTION)
=========================================
1. Loads the generated .pkl files.
2. Checks Data Shape & Class Balance.
3. Plots "Heatmaps" (Spectrograms) for Left vs Right.
4. Checks Normalization (Histogram).
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
try: matplotlib.use('TkAgg') # Pop-up window
except: pass
import matplotlib.pyplot as plt
from config import PROCESSED_DATA_DIR, SUBJECTS

def load_data(subject_id):
    sub_str = f"S{subject_id:03d}"
    file_path = PROCESSED_DATA_DIR / sub_str / f"{sub_str}_spectrograms.pkl"
    
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

    X = data['X'] # (Trials, Channels, Freqs, Time)
    y = data['y'] # Labels
    channels = data['channels']
    
    print(f"Data Shape: {X.shape}")
    print(f"Labels: {np.unique(y, return_counts=True)}")

    # 1. Check Normalization (Should look like a Bell Curve / Gaussian)
    plt.figure(figsize=(10, 4))
    plt.hist(X.flatten(), bins=100, color='purple', alpha=0.7)
    plt.title(f"Pixel Value Distribution (S{subject_id:03d})")
    plt.xlabel("Z-Score Value")
    plt.ylabel("Count")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    # 2. Plot Heatmaps (Left vs Right)
    # Find index of C3 and C4 (Motor Cortex)
    try:
        c3_idx = channels.index('C3')
        c4_idx = channels.index('C4')
    except:
        c3_idx, c4_idx = 0, 1 # Fallback
        
    # Get one example of Left (Class 0/2) and Right (Class 1/3)
    # Note: We group Actual(0,1) and Imagined(2,3) 
    left_trials = np.where(np.isin(y, [0, 2]))[0]
    right_trials = np.where(np.isin(y, [1, 3]))[0]

    if len(left_trials) == 0 or len(right_trials) == 0:
        print("Missing classes!")
        return

    idx_l = left_trials[0]
    idx_r = right_trials[0]

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle(f"Spectrogram Comparison: Subject {subject_id}")

    # Plot C3 (Left Brain) - Controls Right Hand
    axs[0,0].imshow(X[idx_l, c3_idx], aspect='auto', origin='lower', cmap='viridis')
    axs[0,0].set_title("Left Command - Channel C3")
    
    axs[0,1].imshow(X[idx_r, c3_idx], aspect='auto', origin='lower', cmap='viridis')
    axs[0,1].set_title("Right Command - Channel C3")

    # Plot C4 (Right Brain) - Controls Left Hand
    axs[1,0].imshow(X[idx_l, c4_idx], aspect='auto', origin='lower', cmap='viridis')
    axs[1,0].set_title("Left Command - Channel C4")
    
    axs[1,1].imshow(X[idx_r, c4_idx], aspect='auto', origin='lower', cmap='viridis')
    axs[1,1].set_title("Right Command - Channel C4")

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    while True:
        try:
            val = input("Enter Subject ID to visualize (1-109, or 'q'): ")
            if val.lower() == 'q': break
            visualize_subject(int(val))
        except Exception as e:
            print(f"Error: {e}")
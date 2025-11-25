"""
FEATURE VISUALIZATION (EDA) - GENERAL CNN
=========================================
Description:
    1. Loads the generated Spectrograms (_spectrograms.pkl).
    2. Interactive Histogram: Check if data is normalized.
    3. Feature Importance: Which channels matter?
    4. **NEW: Spectrogram Viewer:** See the actual heatmaps the CNN learns from.
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
from sklearn.ensemble import RandomForestClassifier

# --- CONFIGURATION ---
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

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

    # Construct path (Matching V5 Script output)
    folder_name = f"S{sub_id:03d}"
    file_name = f"{folder_name}_spectrograms.pkl" 
    file_path = os.path.join(BASE_OUTPUT_PATH, folder_name, file_name)

    if not os.path.exists(file_path):
        print(f"\n[Error] File not found: {file_path}")
        print("Did you run 'feature_engineering_v5.py'?")
        return None, None, None, None

    print(f"\nLoading: {file_path}...")
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        X = data.get("X")
        y = data.get("y")
        ch_names = data.get("channels")
        class_map = data.get("class_map")

        print("\n[Success] Data Loaded:")
        print(f"  X Shape: {X.shape} (Trials, Channels, Freqs, Time)")
        print(f"  y Shape: {y.shape}")
        return X, y, ch_names, class_map

    except Exception as e:
        print(f"[Error] Corrupted file: {e}")
        return None, None, None, None

# ==========================
# 2. Interactive Distribution
# ==========================
def interactive_feature_distribution(X):
    print("\n[Info] Opening Interactive Distribution Plot...")
    n_samples = X.shape[0]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)
    
    def update(val):
        idx = int(slider.val)
        ax.clear()
        # Plot Channel C3 (10) and C4 (50)
        target_chs = [10, 50] 
        for ch_idx in target_chs:
            if ch_idx < X.shape[1]:
                data = X[idx, ch_idx].flatten()
                ax.hist(data, bins=50, alpha=0.5, label=f"Ch {ch_idx}")
        
        ax.set_title(f"Pixel Value Distribution (Trial {idx})")
        ax.legend()
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Trial', 0, n_samples - 1, valinit=0, valstep=1)
    
    slider.on_changed(update)
    update(0)
    plt.show()

# ==========================
# 3. Feature Importance
# ==========================
def plot_feature_importance(X, y, ch_names):
    print("\n[Info] Calculating Channel Importance...")
    X_flat = X.mean(axis=(2, 3)) # Average over Freq/Time -> (N, Channels)
    
    clf = RandomForestClassifier(n_estimators=50, random_state=42)
    clf.fit(X_flat, y)
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    top_n = 20
    top_indices = indices[:top_n]
    
    if ch_names:
        labels = [ch_names[i] for i in top_indices]
    else:
        labels = [f"Ch {i}" for i in top_indices]
        
    plt.figure(figsize=(12, 6))
    plt.title("Top 20 Important Channels (RF on Spectrograms)")
    plt.bar(range(top_n), importances[top_indices], align="center")
    plt.xticks(range(top_n), labels, rotation=45)
    plt.tight_layout()
    plt.show()

# ==========================
# 4. Spectrogram Heatmaps (NEW!)
# ==========================
def plot_spectrogram_samples(X, y):
    print("\n[Info] Plotting Spectrogram Heatmaps...")
    
    # Find indices for Imagined Left (2) and Imagined Right (3)
    # Note: Depending on label map, might be 0/1 or 2/3. We search for whatever exists.
    classes = np.unique(y)
    print(f"  Classes found in file: {classes}")
    
    # Try to find two different classes to compare
    class_a = classes[0]
    class_b = classes[1] if len(classes) > 1 else classes[0]
    
    idx_a = np.where(y == class_a)[0][0]
    idx_b = np.where(y == class_b)[0][0]
    
    # Channels C3 (Left Motor) and C4 (Right Motor)
    # Adjust indices if your channel map is different
    c3 = 10 
    c4 = 50 
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Class A Row
    im1 = axes[0, 0].imshow(X[idx_a, c3], aspect='auto', origin='lower', cmap='jet')
    axes[0, 0].set_title(f"Class {class_a} - Channel C3 (Left)")
    plt.colorbar(im1, ax=axes[0, 0])
    
    im2 = axes[0, 1].imshow(X[idx_a, c4], aspect='auto', origin='lower', cmap='jet')
    axes[0, 1].set_title(f"Class {class_a} - Channel C4 (Right)")
    plt.colorbar(im2, ax=axes[0, 1])

    # Class B Row
    im3 = axes[1, 0].imshow(X[idx_b, c3], aspect='auto', origin='lower', cmap='jet')
    axes[1, 0].set_title(f"Class {class_b} - Channel C3 (Left)")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(X[idx_b, c4], aspect='auto', origin='lower', cmap='jet')
    axes[1, 1].set_title(f"Class {class_b} - Channel C4 (Right)")
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.show()

# ==========================
# 5. Main Menu
# ==========================
if __name__ == "__main__":
    X, y, ch_names, class_map = load_features()
    
    if X is not None:
        while True:
            print("\n--- GENERAL CNN VISUALIZATION ---")
            print("1. Check Normalization (Histograms)")
            print("2. Check Feature Importance (Channels)")
            print("3. Visualize Spectrograms (Heatmaps)")
            print("Q. Quit")
            
            choice = input("Selection: ").lower().strip()
            
            if choice == '1':
                interactive_feature_distribution(X)
            elif choice == '2':
                plot_feature_importance(X, y, ch_names)
            elif choice == '3':
                plot_spectrogram_samples(X, y)
            elif choice == 'q':
                break
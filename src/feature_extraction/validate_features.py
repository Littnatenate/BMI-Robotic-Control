"""
EEG Spectrogram Feature Explorer (Enhanced)
--------------------------------
- Asks for a Subject ID and automatically loads/combines the
  'imagined' and 'actual' feature files for that subject.
- Loads corresponding `.fif` file to extract *real channel names*.
- Supports interactive feature distribution, correlation, and importance.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
import mne

BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

# ==========================
# 1. Load Features + Channel Names
# ==========================
def load_features():
    # Ask for the subject ID first
    while True:
        try:
            sub_id = int(input("\nEnter subject ID to analyze (1-109): "))
            if 1 <= sub_id <= 109:
                break
            else:
                print("[Error] Please enter a number between 1 and 109.")
        except ValueError:
            print("[Error] Please enter a valid integer.")
            
    # Construct paths
    subject_folder_name = f"S{sub_id:03d}"
    subject_folder_path = os.path.join(BASE_OUTPUT_PATH, subject_folder_name)
    
    file_imagined = os.path.join(subject_folder_path, f"{subject_folder_name}_imagined_movement_features.pkl")
    file_actual = os.path.join(subject_folder_path, f"{subject_folder_name}_actual_movement_features.pkl")

    fif_imagined = os.path.join(subject_folder_path, f"{subject_folder_name}_imagined_movement_cleaned.fif")
    fif_actual = os.path.join(subject_folder_path, f"{subject_folder_name}_actual_movement_cleaned.fif")

    # Check if files exist
    for f in [file_imagined, file_actual]:
        if not os.path.exists(f):
            print(f"[Error] Missing feature file: {f}")
            print("Please run feature extraction before using this validator.")
            return None, None, None

    # Load features
    print(f"\nLoading Imagined: {file_imagined}")
    with open(file_imagined, 'rb') as f:
        data_imagined = pickle.load(f)
        X_imagined = data_imagined['X']
        y_imagined = data_imagined['y']

    print(f"Loading Actual: {file_actual}")
    with open(file_actual, 'rb') as f:
        data_actual = pickle.load(f)
        X_actual = data_actual['X']
        y_actual = data_actual['y']

    # Combine datasets
    X_combined = np.concatenate((X_imagined, X_actual), axis=0)
    y_combined = np.concatenate((y_imagined, y_actual), axis=0)
    
    print("\n[Info] Data combined successfully.")
    print(f"Total X shape: {X_combined.shape}")
    print(f"Total y shape: {y_combined.shape}")
    print(f"Unique labels: {np.unique(y_combined)}")

    # Try to load real channel names from .fif
    ch_names = None
    fif_path = fif_imagined if os.path.exists(fif_imagined) else fif_actual
    if fif_path and os.path.exists(fif_path):
        try:
            print(f"[Info] Loading channel names from: {os.path.basename(fif_path)}")
            raw = mne.io.read_raw_fif(fif_path, preload=False, verbose=False)
            ch_names = raw.ch_names
            print(f"[Info] {len(ch_names)} channel names loaded.")
        except Exception as e:
            print(f"[Warning] Failed to load channel names from {fif_path}: {e}")

    return X_combined, y_combined, ch_names

# ==========================
# 2. Interactive Feature Distribution
# ==========================
def interactive_feature_distribution(X):
    if isinstance(X, np.ndarray) and X.ndim == 4:
        n_samples, n_channels, n_freq, n_time = X.shape
        print(f"[Info] Spectrogram data detected: {X.shape}")

        while True:
            ch_input = input(f"Enter channels to inspect (0-{n_channels-1}) comma-separated or 'all': ").strip().lower()
            if ch_input == 'all':
                selected_channels = list(range(n_channels))
                break
            try:
                selected_channels = [int(c.strip()) for c in ch_input.split(',')]
                if any(c < 0 or c >= n_channels for c in selected_channels):
                    raise ValueError("Channel index out of range.")
                break
            except Exception as e:
                print(f"[Error] {e}. Try again.")

        fig, ax = plt.subplots(figsize=(8, 5))
        plt.subplots_adjust(bottom=0.25)
        current_idx = 0

        def update_plot(idx):
            ax.clear()
            for ch in selected_channels:
                vals = X[idx, ch].flatten()
                ax.hist(vals, bins=30, alpha=0.6, label=f'Ch {ch}')
            ax.set_title(f"Feature Distribution - Sample {idx}")
            ax.set_xlabel("Feature value")
            ax.set_ylabel("Count")
            ax.legend()
            fig.canvas.draw_idle()

        update_plot(current_idx)

        if n_samples > 1:
            slider_ax = fig.add_axes([0.2, 0.1, 0.6, 0.03])
            sample_slider = Slider(slider_ax, 'Sample', 0, n_samples - 1, valinit=0, valstep=1)
            sample_slider.on_changed(lambda val: update_plot(int(sample_slider.val)))

        plt.show(block=True)

    else:
        features = X.columns if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        num_features = len(features)
        fig, ax = plt.subplots(figsize=(7, 4))
        plt.subplots_adjust(bottom=0.25)

        feature_idx = 0
        vals = X[features[feature_idx]] if isinstance(X, pd.DataFrame) else X[:, feature_idx]
        ax.hist(vals, bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Feature Distribution: {features[feature_idx]}")

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Feature', 0, num_features - 1, valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            ax.clear()
            vals = X[features[idx]] if isinstance(X, pd.DataFrame) else X[:, idx]
            ax.hist(vals, bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f"Feature Distribution: {features[idx]}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show(block=True)

# ==========================
# 3. Correlation Heatmap
# ==========================
def plot_correlation_heatmap(X):
    if isinstance(X, np.ndarray):
        X_flat = X.reshape(X.shape[0], -1) if X.ndim > 2 else X
        X_df = pd.DataFrame(X_flat, columns=[f"f{i}" for i in range(X_flat.shape[1])])
    else:
        X_df = X

    corr = X_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, cmap='coolwarm', center=0)
    plt.title("Feature Correlation Heatmap")
    plt.show()

# ==========================
# 4. Feature Importance
# ==========================
def plot_feature_importance(X, y, ch_names=None):
    if y is None:
        print("[Info] No labels found — skipping feature importance.")
        return

    if len(np.unique(y)) < 2:
        print("[Error] Only one class present — cannot compute feature importance.")
        return

    if isinstance(X, np.ndarray) and X.ndim == 4:
        X_feat = X.mean(axis=(-1, -2))  # avg over freq & time → samples × channels
        feature_names = ch_names if ch_names and len(ch_names) == X_feat.shape[1] else [f"Ch{i}" for i in range(X_feat.shape[1])]
        print(f"[Info] Using {X_feat.shape[1]} channels as features (averaged over freq/time).")
    else:
        X_feat = X.reshape(X.shape[0], -1) if isinstance(X, np.ndarray) else X
        feature_names = X.columns if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X_feat.shape[1])]

    print("[Info] Training lightweight RandomForestClassifier...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    clf.fit(X_feat, y)
    importances = clf.feature_importances_

    sorted_idx = np.argsort(importances)[::-1]
    top_n = min(20, len(sorted_idx))

    plt.figure(figsize=(10, 8))
    plt.barh(range(top_n), importances[sorted_idx[:top_n]][::-1], color='lightgreen')
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[:top_n]][::-1])
    plt.xlabel("Importance Score")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

# ==========================
# 5. Main
# ==========================
def main():
    X, y, ch_names = load_features()
    if X is None:
        print("Failed to load data. Exiting.")
        return

    print("\n[Menu Options]")
    print("1: Explore feature distributions (interactive)")
    print("2: Plot correlation heatmap")
    print("3: Show feature importance")
    print("4: Run all checks")

    while True:
        choice = input("\nSelect an option (1-4) or 'q' to quit: ").strip().lower()
        if choice == 'q':
            print("Exiting.")
            break

        if choice not in ['1', '2', '3', '4']:
            print("[Error] Invalid choice.")
            continue

        if choice in ['1', '4']:
            interactive_feature_distribution(X)
        if choice in ['2', '4']:
            plot_correlation_heatmap(X)
        if choice in ['3', '4']:
            plot_feature_importance(X, y, ch_names)

        if choice == '4':
            print("\n[Done] All checks complete.")
        else:
            print(f"\n[Done] Option {choice} complete. Returning to menu.")

if __name__ == "__main__":
    main()

"""
Spectrogram Validation (GAP-CNN)
- Expects: *_spectrograms.pkl containing dict with keys:
    "X" -> (Trials, Channels, Freqs, Time)
    "y" -> (Trials,)
    "channels" -> list of channel names (optional)
    "class_map" -> optional
- Minimal edits from your original, robust channel & label handling
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

BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

def load_features():
    while True:
        try:
            sub_input = input("\nEnter subject ID to analyze (1-109): ").strip()
            sub_id = int(sub_input)
            if 1 <= sub_id <= 109:
                break
            print("Please enter a number between 1 and 109.")
        except ValueError:
            print("Invalid input.")

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

        if X is None or y is None:
            print("[Error] X or y missing in file.")
            return None, None, None, None

        print("\n[Success] Data Loaded:")
        print(f"  X Shape: {X.shape} (Trials, Channels, Freqs, Time)")
        print(f"  y Shape: {y.shape}")
        print("Unique labels before filtering:", np.unique(y))
        
        # FILTER: Keep ONLY Imagined Movement (classes 2, 3)
        if y is not None:
            unique_labels = set(np.unique(y))
            
            # If we have 4 classes (0,1,2,3), keep only 2,3 (Imagined Movement)
            if unique_labels == {0, 1, 2, 3}:
                print("[Info] Found 4 classes. Keeping ONLY classes 2,3 (Imagined Movement)...")
                mask = np.isin(y, [2, 3])
                X = X[mask]
                y = y[mask]
                print(f"  Filtered to {X.shape[0]} trials")
                # Remap 2->0, 3->1 for cleaner labels
                print("[Info] Remapping class 2->0 (Left), class 3->1 (Right)...")
                y = y - 2
            
            # If we already have only 2,3, just remap
            elif unique_labels == {2, 3}:
                print("[Info] Remapping class 2->0 (Left), class 3->1 (Right)...")
                y = y - 2
            
            print("Unique labels after filtering:", np.unique(y))
        
        if ch_names:
            print(f"  Channel names provided: yes ({len(ch_names)} names)")
        else:
            print("  Channel names provided: NO (using numeric indices)")

        return X, y, ch_names, class_map

    except Exception as e:
        print(f"[Error] Corrupted file: {e}")
        return None, None, None, None


def get_motor_channel_indices(ch_names, n_channels):
    # Attempt to find C3/C4 (PhysioNet / standard names). fallback to near-motor indices if missing.
    if ch_names:
        try:
            c3 = ch_names.index("C3")
            c4 = ch_names.index("C4")
            return c3, c4
        except ValueError:
            pass
    # fallback heuristics: assume 64 channels in common ordering - try expected motor region indices
    if n_channels >= 64:
        # Common approximate locations for many exports: C3 ~ index 8-18, C4 ~ 18-26 — pick safe defaults
        # but keep fallback the same as earlier (10, 50) as a last resort
        # we pick middle-of-head guesses:
        return 10, 50
    else:
        return 0, min(n_channels-1, 1)


def safe_label_names(y):
    classes = np.unique(y)
    if set(classes) == {0, 1}:
        return {0: "Left", 1: "Right"}
    if set(classes) == {2, 3}:
        return {2: "Left", 3: "Right"}
    # fallback map
    return {c: f"Class {c}" for c in classes}


def interactive_feature_distribution(X, ch_names):
    n_samples = X.shape[0]
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(bottom=0.25)

    c3_idx, c4_idx = get_motor_channel_indices(ch_names, X.shape[1])
    target_chs = [c3_idx, c4_idx]

    def update(val):
        idx = int(slider.val)
        ax.clear()
        for ch_idx in target_chs:
            if 0 <= ch_idx < X.shape[1]:
                data = X[idx, ch_idx].flatten()
                ax.hist(data, bins=60, alpha=0.6, label=f"Ch {ch_idx}" if not ch_names else ch_names[ch_idx])
        ax.set_title(f"Pixel Value Distribution (Trial {idx})")
        ax.legend()
        ax.set_xlabel("Pixel value")
        fig.canvas.draw_idle()

    ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
    slider = Slider(ax_slider, 'Trial', 0, max(0, n_samples - 1), valinit=0, valstep=1)
    slider.on_changed(update)
    update(0)
    plt.show()


def plot_feature_importance(X, y, ch_names):
    print("\n[Info] Calculating Channel Importance (RF on mean spectrogram power)...")
    # flatten frequency/time by averaging -> (Trials, Channels)
    X_flat = X.mean(axis=(2, 3))
    clf = RandomForestClassifier(n_estimators=80, random_state=42, n_jobs=-1)
    clf.fit(X_flat, y)
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(20, len(importances))
    top_indices = indices[:top_n]
    labels = [ch_names[i] if ch_names else f"Ch {i}" for i in top_indices]
    plt.figure(figsize=(12, 6))
    plt.title("Top Channels (RandomForest on mean spectrogram power)")
    plt.bar(range(top_n), importances[top_indices], align="center")
    plt.xticks(range(top_n), labels, rotation=45)
    plt.tight_layout()
    plt.show()


def plot_spectrogram_samples(X, y, ch_names):
    print("\n[Info] Plotting Spectrogram Heatmaps (sample trials from each class)...")
    classes = np.unique(y)
    print(f"  Classes found: {classes}")

    # Use safe label names
    label_names = safe_label_names(y)

    # Check we have at least 2 classes
    if len(classes) < 2:
        print("[Warning] Only one class found. Cannot compare classes.")
        class_a = classes[0]
        class_b = classes[0]
    else:
        class_a = classes[0]
        class_b = classes[1]

    # Check if classes have trials
    trials_a = np.where(y == class_a)[0]
    trials_b = np.where(y == class_b)[0]
    
    if len(trials_a) == 0 or len(trials_b) == 0:
        print("[Error] One or more classes have no trials.")
        return

    idx_a = trials_a[0]
    idx_b = trials_b[0]

    c3, c4 = get_motor_channel_indices(ch_names, X.shape[1])
    if not (0 <= c3 < X.shape[1] and 0 <= c4 < X.shape[1]):
        print("[Warning] motor channel indices out of range; using first two channels.")
        c3, c4 = 0, min(1, X.shape[1]-1)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    im1 = axes[0, 0].imshow(X[idx_a, c3], aspect='auto', origin='lower', cmap='jet')
    axes[0, 0].set_title(f"{label_names[class_a]} - {ch_names[c3] if ch_names else 'Ch'+str(c3)}")
    plt.colorbar(im1, ax=axes[0, 0])

    im2 = axes[0, 1].imshow(X[idx_a, c4], aspect='auto', origin='lower', cmap='jet')
    axes[0, 1].set_title(f"{label_names[class_a]} - {ch_names[c4] if ch_names else 'Ch'+str(c4)}")
    plt.colorbar(im2, ax=axes[0, 1])

    im3 = axes[1, 0].imshow(X[idx_b, c3], aspect='auto', origin='lower', cmap='jet')
    axes[1, 0].set_title(f"{label_names[class_b]} - {ch_names[c3] if ch_names else 'Ch'+str(c3)}")
    plt.colorbar(im3, ax=axes[1, 0])

    im4 = axes[1, 1].imshow(X[idx_b, c4], aspect='auto', origin='lower', cmap='jet')
    axes[1, 1].set_title(f"{label_names[class_b]} - {ch_names[c4] if ch_names else 'Ch'+str(c4)}")
    plt.colorbar(im4, ax=axes[1, 1])

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y, ch_names, class_map = load_features()
    if X is None:
        raise SystemExit(0)

    # safety shape check
    if X.ndim != 4:
        print("[Error] Expected X with 4 dims (Trials, Channels, Freqs, Time). Found:", X.shape)
        raise SystemExit(0)

    while True:
        print("\n--- SPECTROGRAM (GAP-CNN) VISUALIZATION ---")
        print("1. Check Normalization (Histograms)")
        print("2. Check Feature Importance (Channels)")
        print("3. Visualize Spectrograms (Heatmaps)")
        print("Q. Quit")
        choice = input("Selection: ").lower().strip()
        if choice == '1':
            interactive_feature_distribution(X, ch_names)
        elif choice == '2':
            plot_feature_importance(X, y, ch_names)
        elif choice == '3':
            plot_spectrogram_samples(X, y, ch_names)
        elif choice == 'q':
            break
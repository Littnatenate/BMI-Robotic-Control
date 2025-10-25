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

BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

# ==========================
# 1. Load Features + Channel Names
# ==========================
def load_features():
    while True:
        try:
            sub_id = int(input("\nEnter subject ID to analyze (1-109): "))
            if 1 <= sub_id <= 109:
                break
            else:
                print("[Error] Please enter a number between 1 and 109.")
        except ValueError:
            print("[Error] Please enter a valid integer.")

    # Construct the path to the SINGLE multi-class feature file
    subject_folder_name = f"S{sub_id:03d}"
    subject_folder_path = os.path.join(BASE_OUTPUT_PATH, subject_folder_name)
    # This is the filename saved by the updated feature_engineering script
    feature_filename = f"{subject_folder_name}_multiclass_L_R_features.pkl"
    feature_filepath = os.path.join(subject_folder_path, feature_filename)

    # Check if the file exists
    if not os.path.exists(feature_filepath):
        print(f"[Error] Feature file not found: {feature_filepath}")
        print("Please run the updated feature_engineering.py script first.")
        # Return None for all expected values
        return None, None, None, None

    # Load the single file
    print(f"\nLoading features from: {feature_filepath}")
    try:
        with open(feature_filepath, 'rb') as f:
            data = pickle.load(f)

        # Extract data directly from the loaded dictionary
        X = data.get("X")
        y = data.get("y")
        ch_names = data.get("channels")
        label_map = data.get("label_map")

        # Basic validation
        if X is None or y is None:
            print("[Error] '.pkl' file is missing 'X' or 'y' data.")
            return None, None, None, None
        if label_map is None:
             print("[Warning] 'label_map' not found in .pkl file. Label descriptions might be missing.")
        if ch_names is None:
             print("[Warning] 'channels' (channel names) not found in .pkl file.")
             print("Feature importance plot labels will be generic (Ch0, Ch1...).")
        else:
             print(f"[Info] Successfully loaded {len(ch_names)} channel names from .pkl file.")


        print("\n[Info] Data loaded successfully.")
        print(f"X shape: {X.shape}")
        print(f"y shape: {y.shape}")
        unique_labels, counts = np.unique(y, return_counts=True)
        print(f"Unique labels found: {unique_labels}")

        # Print label distribution using the map if available
        print("Label distribution:")
        if label_map:
            print(f" (Using label map: {label_map})")
            label_map_rev = {v: k for k, v in label_map.items()} # Reverse map
            for label, count in zip(unique_labels, counts):
                label_desc = label_map_rev.get(label, f"Unknown Label {label}")
                print(f"  Label {label} {label_desc}: {count} samples")
        else:
             for label, count in zip(unique_labels, counts):
                print(f"  Label {label}: {count} samples")

        # Return the loaded data
        return X, y, ch_names, label_map

    except Exception as e:
        print(f"[Error] Failed to load or parse {feature_filepath}: {e}")
        return None, None, None, None

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
# 4. Feature Importance
# ==========================
def plot_feature_importance(X, y, ch_names=None):
    if y is None:
        print("[Info] No labels found")
        return
    if len(np.unique(y)) < 2:
        print("[Error] Only one class present")
        return

    n_classes = len(np.unique(y))
    if n_classes > 2:
        print(f"[Info] Detected {n_classes} classes for feature importance")

    if isinstance(X, np.ndarray) and X.ndim == 4:
        X_feat = X.mean(axis=(-1, -2)) # avg over freq & time → samples × channels
        # Use loaded channel names
        feature_names = ch_names if ch_names and len(ch_names) == X_feat.shape[1] else [f"Ch{i}" for i in range(X_feat.shape[1])]
        print(f"[Info] Using {X_feat.shape[1]} channels as features (averaged over freq/time).")
    else: # Handling for non-4D data (same as your original)
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
    # Load data including the label map
    X, y, ch_names, label_map = load_features()
    if X is None:
        print("Failed to load data. Exiting.")
        return

    # --- Start Menu Update ---
    print("\n[Menu Options]")
    print("1: Explore feature distributions")
    print("2: Show feature importance")
    print("3: Run all checks")

    while True:
        # Adjusted prompt for new options
        choice = input("\nSelect an option (1-3) or 'q' to quit: ").strip().lower()
        if choice == 'q':
            print("Exiting.")
            break

        # Adjusted valid choices and logic
        if choice not in ['1', '2', '3']:
            print("[Error] Invalid choice.")
            continue

        run_distribution = False
        run_importance = False

        if choice == '1':
            run_distribution = True
        elif choice == '2':
            run_importance = True
        elif choice == '3':
            run_distribution = True
            run_importance = True

        # Execute selected functions
        if run_distribution:
            print("\nStarting: 1. Explore feature distributions...")
            interactive_feature_distribution(X)

        if run_importance:
            print("\nStarting: 2. Show feature importance...")
            plot_feature_importance(X, y, ch_names)

        # Print completion message
        if choice == '3':
            print("\n[Done] All checks complete.")
        else:
            # Get the description based on the choice number
            if choice == '1':
                 desc = "Explore distributions"
            elif choice == '2':
                 desc = "Show feature importance"
            print(f"\n[Done] Option {choice} ({desc}) complete. Returning to menu.")
    # --- End Menu Update ---

if __name__ == "__main__":
    main()
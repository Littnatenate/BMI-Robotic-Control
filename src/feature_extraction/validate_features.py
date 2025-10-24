"""
EEG Spectrogram Feature Explorer
--------------------------------
- Asks for a Subject ID and automatically loads/combines the
  'imagined' and 'actual' feature files for that subject.
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

BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

# ==========================
# 1. Load Features (NEW & IMPROVED)
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
            
    # Automatically construct the file paths
    subject_folder_name = f"S{sub_id:03d}"
    subject_folder_path = os.path.join(BASE_OUTPUT_PATH, subject_folder_name)
    
    file_imagined = os.path.join(subject_folder_path, f"{subject_folder_name}_imagined_movement_features.pkl")
    file_actual = os.path.join(subject_folder_path, f"{subject_folder_name}_actual_movement_features.pkl")

    # Check if both files exist before trying to load
    if not os.path.exists(file_imagined):
        print(f"[Error] File not found: {file_imagined}")
        print("Please run the feature engineering script for this subject and task.")
        return None, None
        
    if not os.path.exists(file_actual):
        print(f"[Error] File not found: {file_actual}")
        print("Please run the feature engineering script for this subject and task.")
        return None, None

    # Load both files
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

    # Combine them
    X_combined = np.concatenate((X_imagined, X_actual), axis=0)
    y_combined = np.concatenate((y_imagined, y_actual), axis=0)
    
    print("\n[Info] Data combined successfully.")
    print(f"Total X shape: {X_combined.shape}")
    print(f"Total y shape: {y_combined.shape}")
    
    unique_labels = np.unique(y_combined)
    print(f"Unique labels in data: {unique_labels}")

    # Final check to make sure we have two classes
    if len(unique_labels) < 2:
        print("[CRITICAL ERROR] Combined data has only one class.")
        print("This should not happen if files loaded correctly. Check labels in Script 2.")
    
    return X_combined, y_combined

# ==========================
# 2. Interactive Feature Distribution
# ==========================
def interactive_feature_distribution(X):
    # Handle 4D EEG spectrogram: samples × channels × freq × time
    if isinstance(X, np.ndarray) and X.ndim == 4:
        n_samples, n_channels, n_freq, n_time = X.shape
        print(f"[Info] Spectrogram data detected: {X.shape}")

        # Select channels to inspect
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
                hist_vals = X[idx, ch].flatten()
                ax.hist(hist_vals, bins=30, alpha=0.6, label=f'Ch {ch}')
                mean_val = np.mean(hist_vals)
                min_val = np.min(hist_vals)
                max_val = np.max(hist_vals)
                ax.text(0.95, 0.95 - (0.05 * selected_channels.index(ch)), f"Ch{ch}: mean={mean_val:.3f}, min={min_val:.3f}, max={max_val:.3f}",
                        transform=ax.transAxes, ha='right', va='top', fontsize=9)
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
        else:
            print("[Info] Only one sample. No slider needed.")

        plt.show(block=True)

    # Handle 2D / DataFrame features
    else:
        features = X.columns if isinstance(X, pd.DataFrame) else [f"f{i}" for i in range(X.shape[1])]
        num_features = len(features)

        fig, ax = plt.subplots(figsize=(7, 4))
        plt.subplots_adjust(bottom=0.25)

        feature_idx = 0
        hist_vals = X[features[feature_idx]] if isinstance(X, pd.DataFrame) else X[:, feature_idx]
        ax.hist(hist_vals, bins=30, color='skyblue', edgecolor='black')
        ax.set_title(f"Feature Distribution: {features[feature_idx]}")

        ax_slider = plt.axes([0.25, 0.1, 0.5, 0.03])
        slider = Slider(ax_slider, 'Feature', 0, num_features - 1, valinit=0, valstep=1)

        def update(val):
            idx = int(slider.val)
            ax.clear()
            hist_vals = X[features[idx]] if isinstance(X, pd.DataFrame) else X[:, idx]
            ax.hist(hist_vals, bins=30, color='skyblue', edgecolor='black')
            ax.set_title(f"Feature Distribution: {features[idx]}")
            fig.canvas.draw_idle()

        slider.on_changed(update)
        plt.show(block=True)

# ==========================
# 3. Correlation Heatmap
# ==========================
def plot_correlation_heatmap(X):
    if isinstance(X, np.ndarray):
        if X.ndim > 2:
            X_flat = X.reshape(X.shape[0], -1)
        else:
            X_flat = X
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
def plot_feature_importance(X, y):
    if y is None:
        print("[Info] Labels not found. Skipping feature importance.")
        return
    
    if len(np.unique(y)) < 2:
        print("[Error] Only one class found in labels (y).")
        print("Cannot calculate feature importance. Plot will be empty.")
    
    # -------------------------------
    # Handle 4D spectrogram features
    # -------------------------------
    if isinstance(X, np.ndarray) and X.ndim == 4:
        # Average power across time axis
        X_avg_time = X.mean(axis=-1)  # samples × channels × freq
        # Average power across frequency axis to reduce to channels only
        X_feat = X_avg_time.mean(axis=-1)  # samples × channels
        
        # Try to get channel names if available (assuming 64 channels from standard setup)
        # This is a hard-coded assumption but makes the plot more readable
        if X_feat.shape[1] == 64:
             # Standard 10-20 channel names used in PhysioNet
            ch_names = [
                'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6', 'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
                'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6', 'FP1', 'FPZ', 'FP2', 'AF7', 'AF3', 'AFZ',
                'AF4', 'AF8', 'F7', 'F5', 'F3', 'F1', 'FZ', 'F2', 'F4', 'F6', 'F8', 'FT7', 'FT8', 'T7',
                'T8', 'T9', 'T10', 'TP7', 'TP8', 'P7', 'P5', 'P3', 'P1', 'PZ', 'P2', 'P4', 'P6', 'P8',
                'PO7', 'PO3', 'POZ', 'PO4', 'PO8', 'O1', 'OZ', 'O2', 'IZ'
            ]
            # In case your data doesn't have exactly 64 channels, fall back to default
            if len(ch_names) == X_feat.shape[1]:
                 feature_names = ch_names
            else:
                 feature_names = [f"Ch{i}" for i in range(X_feat.shape[1])]
        else:
            feature_names = [f"Ch{i}" for i in range(X_feat.shape[1])]
            
        print(f"[Info] Using {X_feat.shape[1]} channels as features (averaged over freq & time).")
    else:
        # Flatten if 2D or DataFrame
        if isinstance(X, np.ndarray):
            X_feat = X.reshape(X.shape[0], -1)
            feature_names = [f"f{i}" for i in range(X_feat.shape[1])]
        else:
            X_feat = X
            feature_names = X.columns

    # -------------------------------
    # Train RandomForest
    # -------------------------------
    print("[Info] Training lightweight RandomForestClassifier for feature importance...")
    clf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    
    clf.fit(X_feat, y) 
    importances = clf.feature_importances_

    # -------------------------------
    # Plot top features
    # -------------------------------
    sorted_idx = importances.argsort()[::-1]
    top_n = min(20, len(sorted_idx)) # Show top 20 features

    plt.figure(figsize=(10, 8)) 
    plt.barh(range(top_n), importances[sorted_idx[:top_n]][::-1], color='lightgreen', align='center')
    plt.yticks(range(top_n), [feature_names[i] for i in sorted_idx[:top_n]][::-1])
    plt.xlabel("Importance Score")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    plt.show()

# 5. Main

def main():
    X, y = load_features()
    if X is None:
        print("Failed to load or combine data. Exiting.")
        return

    print("\n[Menu Options]")
    print("1: Explore feature distributions (interactive)")
    print("2: Plot correlation heatmap")
    print("3: Show feature importance (requires combined data)")
    print("4: Run all checks")

    while True:
        choice = input("\nSelect an option (1-4) or 'q' to quit: ").strip().lower()
        
        if choice == 'q':
            print("Exiting.")
            break
            
        if choice not in ['1', '2', '3', '4']:
            print("[Error] Invalid choice, try again.")
            continue

        if choice in ['1', '4']:
            print("\nStarting: 1. Explore feature distributions...")
            interactive_feature_distribution(X)
        if choice in ['2', '4']:
            print("\nStarting: 2. Plot correlation heatmap...")
            plot_correlation_heatmap(X)
        if choice in ['3', '4']:
            print("\nStarting: 3. Show feature importance...")
            plot_feature_importance(X, y)
        
        if choice != '4':
            print(f"\nCompleted option {choice}. Returning to menu.")
        else:
            print("\nAll checks complete. Returning to menu.")
            
    print("Program finished.")


if __name__ == "__main__":
    main()
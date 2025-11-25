"""
Feature Engineering for EEG Motor Imagery Data (Enhanced & Stable Version)
------------------------------------------------
Steps:
1. Load preprocessed EEG (.fif) files for specified tasks.
2. Detect and extract event markers (T1 = Left, T2 = Right) for each task.
3. Create 2-second epochs using a clean window [0.5s, 2.5s] to remove cue noise.
4. Compute time–frequency spectrograms (in dB scale) for each epoch and channel.
5. Apply per-channel Z-score standardization (preserves inter-channel differences).
6. Assign task-specific and event-specific labels.
7. Combine all tasks into a unified dataset with consistent shapes.
8. (Optional) Visualize representative epochs and spectrograms for QC.
9. Save the final feature set (X, y, channel_names) for CNN-based model training.

Improvements over previous version:
- Avoids MinMax squashing by using decibel normalization.
- Ignores cue artifacts by shifting epoch window.
- Guarantees consistent spectrogram dimensions across epochs.
- Includes better overlap handling and signal stability for model training.
"""

import os
import numpy as np
import mne
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import pickle
import matplotlib
matplotlib.use('Qt5Agg') # Use an interactive backend
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

# Interactive plotting for .py scripts
plt.ion()

# ------------------------------
# Config / Paths
# ------------------------------
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"
EPOCH_DURATION = 2.0 # seconds
FREQ_RANGE = (1, 40) # Hz
NFFT = 128          # FFT length
VISUALIZE = True    # <<< Set to True to enable interactive plots per task

# --- Define the classes and their final labels ---
LABEL_MAP = {
    ('actual_movement', 'left'): 0,
    ('actual_movement', 'right'): 1,
    ('imagined_movement', 'left'): 2,
    ('imagined_movement', 'right'): 3,
}
# List of tasks to process for this specific experiment
TASKS_TO_PROCESS = ['actual_movement', 'imagined_movement']


# ------------------------------
# Brain region mapping (Optional, used only for visualization selection)
# ------------------------------
BRAIN_REGIONS = {
    "Frontal":  ['FP1','FPZ','FP2','AF7','AF3','AFZ','AF4','AF8','F7','F5','F3','F1','FZ','F2','F4','F6','F8'],
    "Central": ['FT7','FC5','FC3','FC1','FCZ','FC2','FC4','FC6','FT8','T7','C5','C3','C1','CZ','C2','C4','C6','T8'],
    "Parietal": ['TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8','P7','P5','P3','P1','PZ','P2','P4','P6','P8'],
    "Occipital": ['PO7','PO3','POZ','PO4','PO8','O1','OZ','O2','IZ'],
}

def get_channel_indices_by_region(epochs_info, selected_regions):
    ch_indices = []
    ch_names_upper = [ch.upper() for ch in epochs_info['ch_names']]
    for region in selected_regions:
        for ch_name_target in BRAIN_REGIONS.get(region, []):
            try:
                idx = ch_names_upper.index(ch_name_target.upper())
                ch_indices.append(idx)
            except ValueError:
                pass 
    return sorted(list(set(ch_indices))) 

# ------------------------------
# EEG Loading & Epoching
# ------------------------------
def load_raw_eeg(subject_id, task_name):
    subject_folder = f"S{subject_id:03d}"
    processed_file = os.path.join(BASE_OUTPUT_PATH, subject_folder,
                                  f"{subject_folder}_{task_name}_cleaned.fif")
    if not os.path.exists(processed_file):
        print(f"\n[Error] File not found: {processed_file}")
        print("Please ensure the preprocessor script has been run for this subject and task.")
        return None
    try:
        raw = mne.io.read_raw_fif(processed_file, preload=True, verbose='WARNING')
        return raw
    except Exception as e:
        print(f"\n[Error] Failed to load {processed_file}: {e}")
        return None

def create_epochs(raw, tmin=0.0, tmax=2.0):
    """ Creates epochs based on 'T1' (Left) and 'T2' (Right) events. """
    try:
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)

        required_events = ['T1', 'T2']
        if not all(event in event_id_map for event in required_events):
            print(f"[Warning] Missing required event types (T1 or T2) in annotations for {raw.filenames}.")
            print(f"Found events: {list(event_id_map.keys())}")
            return None, None

        event_id = {'left': event_id_map['T1'], 'right': event_id_map['T2']}
        print(f"Found events: {event_id}")

        # Ensure tmax aligns exactly with expected duration based on sfreq
        tmax_exact = tmax - (1 / raw.info['sfreq']) if tmax > 0 else 0
        
        epochs = mne.Epochs(raw, events, event_id=event_id, tmin=tmin, tmax=tmax_exact,
                            proj=False, baseline=None, preload=True, verbose='WARNING',
                            on_missing='warning')

        if len(epochs) == 0:
            print("[Warning] No epochs were created. Check event timings and tmin/tmax.")
            return None, None

        return epochs, event_id 

    except Exception as e:
        print(f"[Error] Failed during epoch creation: {e}")
        return None, None

# ------------------------------
# Feature Extraction
# ------------------------------
def compute_spectrogram(segment, sfreq, nfft=128, freq_range=(1, 40)):
    """Computes spectrogram (in dB) for a single epoch (channels x samples)."""
    n_channels = segment.shape[0]
    spec_list = []
    all_Sxx_shape = None 
    for ch in range(n_channels):
        f, t, Sxx = spectrogram(segment[ch, :], fs=sfreq, nperseg=nfft, noverlap=int(nfft * 0.875))
        
        # Convert power to decibels
        Sxx_no_zeros = Sxx + 1e-10 
        Sxx_db = 10 * np.log10(Sxx_no_zeros)
        
        mask = (f >= freq_range[0]) & (f <= freq_range[1])
        Sxx_db_masked = Sxx_db[mask, :]
        
        if all_Sxx_shape is None:
            all_Sxx_shape = Sxx_db_masked.shape # (n_freqs, n_times)
        elif Sxx_db_masked.shape != all_Sxx_shape:
            # Handle potential edge cases
            target_shape = all_Sxx_shape
            pad_width = [(0, max(0, target_shape[i] - Sxx_db_masked.shape[i])) for i in range(Sxx_db_masked.ndim)]
            Sxx_db_masked = np.pad(Sxx_db_masked, pad_width, mode='constant', constant_values=0)
            Sxx_db_masked = Sxx_db_masked[:target_shape[0], :target_shape[1]] 

        spec_list.append(Sxx_db_masked)
    return np.array(spec_list) # (n_channels, n_freqs, n_time_bins)

def standardize_spectrograms(specs):
    """Standardizes (Z-scores) features across all epochs and channels."""
    if specs.size == 0: return specs
    n_epochs, n_channels, n_freqs, n_time_bins = specs.shape

    all_features_flat = specs.reshape(-1, 1)

    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(all_features_flat)

    scaled_specs = scaled_features.reshape(n_epochs, n_channels, n_freqs, n_time_bins)
    return scaled_specs


# ------------------------------
# Visualization [epochs]
# ------------------------------
def select_epochs_and_regions(epochs, n_epochs): # Pass epochs object and total count
    """ Interactive selection of epochs and brain regions for visualization. """
    while True:
        epoch_input = input(f"Enter epoch indices to view (0-{n_epochs-1}, comma-separated or 'all'): ").strip().lower()
        try:
            if epoch_input == 'all':
                epoch_indices = list(range(n_epochs))
            else:
                epoch_indices = [int(s.strip()) for s in epoch_input.split(',')]
                if any(idx < 0 or idx >= n_epochs for idx in epoch_indices):
                    raise ValueError("Epoch index out of range.")
            if not epoch_indices:
                raise ValueError("No valid epoch indices selected.")
            break
        except Exception as e:
            print(f"[Error] {e}. Please enter valid indices or 'all'.")

    print("Available regions:", list(BRAIN_REGIONS.keys()))
    while True:
        region_input = input("Enter brain regions to view (comma-separated or 'all'): ").strip().title()
        selected_regions = []
        if region_input.lower() == 'all':
            selected_regions = list(BRAIN_REGIONS.keys())
        else:
            selected_regions = [r.strip().title() for r in region_input.split(',')]
            invalid = [r for r in selected_regions if r not in BRAIN_REGIONS]
            if invalid:
                print(f"[Error] Invalid regions: {invalid}. Available: {list(BRAIN_REGIONS.keys())}")
                continue

        ch_indices = get_channel_indices_by_region(epochs.info, selected_regions)
        if not ch_indices:
            print("[Error] No channels found for the selected regions in this dataset. Try 'all' or check region definitions.")
            if region_input.lower() == 'all':
                print("[Detail] Channel names in data:", epochs.info['ch_names'])
                print("[Detail] Channel names expected in BRAIN_REGIONS values.")
            continue
        break

    while True:
        avg_input = input("Show average across selected channels? [y/n]: ").strip().lower()
        if avg_input in ['y', 'n']:
            average = (avg_input == 'y')
            break
        else:
            print("[Error] Please enter 'y' or 'n'.")

    return epoch_indices, ch_indices, average

def visualize_epochs_interactive(epochs, specs, epoch_indices, ch_indices, average=False):
    """ Interactive plot for selected epochs (raw trace + spectrogram). """
    n_selected_epochs = len(epoch_indices)
    sfreq = epochs.info['sfreq']
    all_epoch_data = epochs.get_data(picks=ch_indices) 
    selected_epoch_data = all_epoch_data[epoch_indices] 
    
    selected_specs = specs[epoch_indices][:, ch_indices, :, :] 

    selected_ch_names = [epochs.info['ch_names'][i] for i in ch_indices]
    n_display_rows = 1 if average else min(len(ch_indices), 3) 

    fig, axes = plt.subplots(n_display_rows, 2, figsize=(14, n_display_rows * 4), squeeze=False)
    plt.subplots_adjust(left=0.1, bottom=0.15, right=0.95, top=0.9, hspace=0.4, wspace=0.3)
    raw_lines, spec_images = [], []
    time_axis = epochs.times 

    epoch_idx_in_selection = 0
    current_global_epoch_idx = epoch_indices[epoch_idx_in_selection]

    for i in range(n_display_rows):
        ax_raw, ax_spec = axes[i]
        plot_ch_name = "Average" if average else selected_ch_names[i]

        if average:
            current_raw_data = selected_epoch_data[epoch_idx_in_selection].mean(axis=0)
            current_spec_data = selected_specs[epoch_idx_in_selection].mean(axis=0)
        else:
            current_raw_data = selected_epoch_data[epoch_idx_in_selection][i]
            current_spec_data = selected_specs[epoch_idx_in_selection][i]

        line, = ax_raw.plot(time_axis, current_raw_data)
        raw_lines.append(line)
        ax_raw.set_title(f"{plot_ch_name} - Epoch {current_global_epoch_idx}")
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Amplitude (µV)")
        ax_raw.grid(True)
        ax_raw.autoscale(enable=True, axis='y')

        spec_n_times = current_spec_data.shape[1]
        spec_time_axis = np.linspace(time_axis[0], time_axis[-1], spec_n_times) 
        extent = [spec_time_axis[0], spec_time_axis[-1], FREQ_RANGE[0], FREQ_RANGE[1]]
        
        # Data is already in dB, no need to log10 again
        db_spec_data = current_spec_data 

        im = ax_spec.imshow(db_spec_data, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        spec_images.append(im)
        ax_spec.set_title(f"Spectrogram - {plot_ch_name}")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        cbar = fig.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
        cbar.set_label('Power/Frequency (dB/Hz)') 


    # Slider update function
    def update_plot(val):
        idx_in_selection = int(val) 
        global_epoch_idx = epoch_indices[idx_in_selection]

        for i in range(n_display_rows):
            ax_raw, ax_spec = axes[i]
            plot_ch_name = "Average" if average else selected_ch_names[i]

            if average:
                new_raw_data = selected_epoch_data[idx_in_selection].mean(axis=0)
                new_spec_data = selected_specs[idx_in_selection].mean(axis=0)
            else:
                new_raw_data = selected_epoch_data[idx_in_selection][i]
                new_spec_data = selected_specs[idx_in_selection][i]

            raw_lines[i].set_ydata(new_raw_data)
            ax_raw.set_title(f"{plot_ch_name} - Epoch {global_epoch_idx}")
            ax_raw.relim() 
            ax_raw.autoscale_view(scaley=True) 

            db_spec_data = new_spec_data
            spec_images[i].set_data(db_spec_data)
            
            vmin = np.percentile(db_spec_data, 5)
            vmax = np.percentile(db_spec_data, 95)
            spec_images[i].set_clim(vmin=vmin, vmax=vmax)
            ax_spec.set_title(f"Spectrogram - {plot_ch_name}")

        fig.canvas.draw_idle()

    # Add slider if more than one epoch selected
    if n_selected_epochs > 1:
        slider_ax = fig.add_axes([0.2, 0.03, 0.6, 0.03]) 
        epoch_slider = Slider(slider_ax, 'Epoch Index', 0, n_selected_epochs - 1, valinit=0, valstep=1)
        epoch_slider.on_changed(update_plot)
    else:
        print("[Info] Only one epoch selected. No slider needed.")

    plt.show(block=True) 


# ------------------------------
# Pipeline per Task for a Subject
# ------------------------------
def process_subject_task(subject_id, task_name, visualize=False):
    """ Processes a single task (e.g., 'imagined_movement') for a subject. """
    print(f"\n--- Processing Subject {subject_id}, Task: {task_name} ---")
    raw = load_raw_eeg(subject_id, task_name)
    if raw is None:
        return None, None, None, None 

    sfreq = raw.info['sfreq']
    channel_names = raw.ch_names

    # Create epochs: shift window to [0.5s, 2.5s]
    print("Creating epochs from 0.5s to 2.5s...")
    epochs, event_id = create_epochs(raw, tmin=0.5, tmax=EPOCH_DURATION + 0.5)
    
    if epochs is None:
        return None, None, None, None 

    print(f"Created {len(epochs)} epochs for task '{task_name}'.")
    epoched_data = epochs.get_data() 

    # Compute spectrogram for each epoch
    specs_list = [compute_spectrogram(epoch, sfreq=sfreq, nfft=NFFT, freq_range=FREQ_RANGE)
                    for epoch in epoched_data]
    specs = np.array(specs_list) 

    if specs.size == 0:
        print("[Error] Spectrogram computation resulted in an empty array.")
        return None, None, None, None

    print(f"Spectrogram shape for task '{task_name}': {specs.shape}")

    # --- Visualization Point ---
    if visualize:
        print(f"\n--- Visualization for Subject {subject_id}, Task: {task_name} ---")
        epoch_indices, ch_indices, average = select_epochs_and_regions(epochs, len(epochs))
        # Pass unscaled (but dB-converted) specs for visualization
        visualize_epochs_interactive(epochs, specs, epoch_indices, ch_indices, average)
        print(f"--- End Visualization for Task: {task_name} ---")


    # --- Normalization ---
    # We will standardize the features here, but will re-standardize
    # after combining all tasks for better consistency.
    specs_scaled = standardize_spectrograms(specs)


    # --- Generate Labels ---
    final_labels = []
    raw_event_codes = epochs.events[:, -1] 
    event_id_rev = {v: k for k, v in event_id.items()} 

    for code in raw_event_codes:
        event_desc = event_id_rev.get(code, 'unknown') 
        label_key = (task_name, event_desc) 
        final_label = LABEL_MAP.get(label_key, -1) 
        if final_label == -1:
            print(f"[Warning] Could not find label mapping for key: {label_key}")
        final_labels.append(final_label)

    final_labels = np.array(final_labels)
    
    # Return the scaled specs for this task
    return specs_scaled, final_labels, channel_names, epochs.info 


# ------------------------------
# Save Combined Dataset
# ------------------------------
def save_dataset(X, y, channel_names, subject_id, label_map):
    subject_folder = f"S{subject_id:03d}"
    output_folder = os.path.join(BASE_OUTPUT_PATH, subject_folder)
    os.makedirs(output_folder, exist_ok=True)

    output_filename = f"{subject_folder}_multiclass_L_R_features.pkl"
    output_file = os.path.join(output_folder, output_filename)

    data_to_save = {
        "X": X,
        "y": y,
        "channels": channel_names,
        "label_map": label_map 
    }

    with open(output_file, "wb") as f:
        pickle.dump(data_to_save, f)
    print(f"\n✅ Combined features (X, y, channels, label_map) saved to: {output_file}")


# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    
    # This block simply asks for the subject ID.
    while True:
        try:
            subject_id = int(input("Enter subject ID (1-109): "))
            if 1 <= subject_id <= 109:
                break
            else:
                print("Please enter a number between 1 and 109.")
        except ValueError:
            print("Invalid input. Please enter an integer.")

    print(f"\nStarting Multi-Class (Left/Right) Feature Extraction for Subject {subject_id}...")
    print(f"Tasks to process: {TASKS_TO_PROCESS}")
    print(f"Label mapping used: {LABEL_MAP}")

    all_X = []
    all_y = []
    final_channel_names = None
    success = True

    for task in TASKS_TO_PROCESS:
        X_task, y_task, ch_names_task, info_task = process_subject_task(subject_id, task, visualize=VISUALIZE)

        if X_task is not None and y_task is not None:
            all_X.append(X_task)
            all_y.append(y_task)
            if final_channel_names is None: 
                final_channel_names = ch_names_task
            elif final_channel_names != ch_names_task:
                print(f"[Warning] Channel names mismatch between tasks for Subject {subject_id}!")
        else:
            print(f"[Error] Failed to process task '{task}' for subject {subject_id}. Skipping.")
            success = False 

    if not all_X or not all_y:
        print(f"\n[Error] No data successfully processed for Subject {subject_id}. Exiting.")
    elif final_channel_names is None:
        print(f"\n[Error] Could not determine channel names for Subject {subject_id}. Exiting.")
    else:
        # Combine data from all tasks
        X_combined = np.concatenate(all_X, axis=0)
        y_combined = np.concatenate(all_y, axis=0)

        # Apply the final standardization across the *entire* subject's data
        # This is a better approach than standardizing per task
        print(f"\nApplying final standardization across all {len(X_combined)} combined epochs...")
        X_combined_scaled = standardize_spectrograms(X_combined)


        print(f"\n--- Combined Results for Subject {subject_id} ---")
        print(f"Combined X shape: {X_combined_scaled.shape}")
        print(f"Combined y shape: {y_combined.shape}")
        unique_labels, counts = np.unique(y_combined, return_counts=True)
        print("Label distribution:")
        
        label_map_rev = {v: k for k, v in LABEL_MAP.items()} 
        for label, count in zip(unique_labels, counts):
            label_desc = label_map_rev.get(label, f"Unknown Label {label}")
            print(f"  Label {label} {label_desc}: {count} samples")

        save_prompt = input("Save combined dataset? [Y/N]: ").strip().lower()
        if save_prompt == 'y':
            # Save the final scaled data
            save_dataset(X_combined_scaled, y_combined, final_channel_names, subject_id, LABEL_MAP)
        else:
            print("Combined dataset not saved.")
"""
Feature Engineering for EEG Motor Imagery Data
------------------------------------------------
Steps:
1. Load preprocessed EEG (.fif) files from preprocessor.py
2. Segment EEG into sliding windows
3. Convert each segment into 2D spectrograms
4. Normalize features
5. Interactive visualization (optional)
6. Save final dataset for CNN training
"""

import os
import numpy as np
import mne
from scipy.signal import spectrogram
from sklearn.preprocessing import MinMaxScaler
import pickle
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

plt.ion()

# ------------------------------
# Config / Paths
# ------------------------------
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"
WINDOW_LENGTH = 2.0  # seconds
STEP_SIZE = 1.0      # seconds
FREQ_RANGE = (1, 40) # Hz
NFFT = 256           # FFT length
VISUALIZE = True     # Toggle visualization

# ------------------------------
# Brain region mapping
# ------------------------------
BRAIN_REGIONS = {
    "Frontal":  ['Fp1','Fp2','F3','F4','F7','F8','Fz'],
    "Parietal": ['P3','P4','P7','P8','Pz'],
    "Temporal": ['T7','T8','TP9','TP10'],
    "Occipital": ['O1','O2','Oz'],
    "Central": ['C3','C4','Cz'],
}

def get_channel_indices_by_region(raw, selected_regions):
    ch_indices = []
    for region in selected_regions:
        for ch_name in BRAIN_REGIONS.get(region, []):
            if ch_name in raw.ch_names:
                ch_indices.append(raw.ch_names.index(ch_name))
    return ch_indices

# ------------------------------
# EEG Loading & Segmentation
# ------------------------------
def load_raw_eeg(subject_id, task_name):
    subject_folder = f"S{subject_id:03d}"
    processed_file = os.path.join(BASE_OUTPUT_PATH, subject_folder,
                                  f"{subject_folder}_{task_name}_cleaned.fif")
    raw = mne.io.read_raw_fif(processed_file, preload=True)
    return raw

def segment_signal(raw, window_length=2.0, step_size=1.0):
    sfreq = raw.info['sfreq']
    n_samples_window = int(window_length * sfreq)
    step_samples = int(step_size * sfreq)
    data = raw.get_data()
    segments = []
    for start in range(0, data.shape[1] - n_samples_window + 1, step_samples):
        end = start + n_samples_window
        segments.append(data[:, start:end])
    return np.array(segments)

def compute_spectrogram(segment, sfreq, nfft=256, freq_range=(1, 40)):
    n_channels = segment.shape[0]
    spec_list = []
    for ch in range(n_channels):
        f, t, Sxx = spectrogram(segment[ch, :], fs=sfreq, nperseg=nfft, noverlap=nfft//2)
        mask = (f >= freq_range[0]) & (f <= freq_range[1])
        Sxx = Sxx[mask, :]
        spec_list.append(Sxx)
    return np.array(spec_list)

def normalize_spectrograms(specs):
    n_samples = specs.shape[0]
    scaled_specs = np.zeros_like(specs)
    for i in range(n_samples):
        flat = specs[i].flatten().reshape(-1, 1)
        scaler = MinMaxScaler()
        scaled = scaler.fit_transform(flat).reshape(specs[i].shape)
        scaled_specs[i] = scaled
    return scaled_specs

# ------------------------------
# Interactive Visualization
# ------------------------------
def select_segments_and_regions(raw, segments):
    while True:
        seg_input = input(f"Enter segment indices to view (comma-separated or 'all'): ").strip().lower()
        try:
            if seg_input == 'all':
                seg_indices = list(range(segments.shape[0]))
            else:
                seg_indices = [int(s.strip()) for s in seg_input.split(',')]
                if any(idx < 0 or idx >= segments.shape[0] for idx in seg_indices):
                    raise ValueError("Segment index out of range.")
            break
        except Exception as e:
            print(f"[Error] {e}. Try again.")

    print("Available regions:", list(BRAIN_REGIONS.keys()))
    while True:
        region_input = input("Enter brain regions to view (comma-separated or 'all'): ").strip().title()
        if region_input.lower() == 'all':
            selected_regions = list(BRAIN_REGIONS.keys())
        else:
            selected_regions = [r.strip().title() for r in region_input.split(',')]
            invalid = [r for r in selected_regions if r not in BRAIN_REGIONS]
            if invalid:
                print(f"[Error] Invalid regions: {invalid}. Try again.")
                continue
        ch_indices = get_channel_indices_by_region(raw, selected_regions)
        if not ch_indices:
            print("[Error] No valid channels found. Try again.")
            continue
        break

    while True:
        avg_input = input("Show average across selected channels? [y/n]: ").strip().lower()
        if avg_input in ['y','n']:
            average = avg_input == 'y'
            break
        else:
            print("[Error] Please enter 'y' or 'n'")

    return seg_indices, ch_indices, average

def visualize_segments_interactive(segments, specs, raw, seg_indices, ch_indices, average=False):
    n_segments_total, n_channels_total, n_samples = segments.shape
    sfreq = raw.info['sfreq']
    selected_segments = segments[seg_indices][:, ch_indices, :]
    selected_specs = specs[seg_indices][:, ch_indices, :, :]
    selected_ch_names = [raw.ch_names[i] for i in ch_indices]
    n_selected_segments = len(seg_indices)
    n_display_rows = 1 if average else min(len(ch_indices), 2)

    fig, axes = plt.subplots(n_display_rows, 2, figsize=(12, n_display_rows * 3), squeeze=False)
    plt.subplots_adjust(bottom=0.25, top=0.92, hspace=0.5, wspace=0.3)
    raw_lines, spec_images = [], []
    time_axis = np.arange(n_samples) / sfreq

    for i in range(n_display_rows):
        ax_raw, ax_spec = axes[i]
        if average:
            line, = ax_raw.plot(time_axis, selected_segments[0].mean(axis=0))
            spec_data = selected_specs[0].mean(axis=0)
            plot_ch_name = "Average"
            ax_raw.set_title(f"Average - Segment {seg_indices[0]}")
        else:
            channel_idx_in_selection = i
            line, = ax_raw.plot(time_axis, selected_segments[0][channel_idx_in_selection])
            plot_ch_name = selected_ch_names[channel_idx_in_selection]
            ax_raw.set_title(f"{plot_ch_name} - Segment {seg_indices[0]}")
            spec_data = selected_specs[0][channel_idx_in_selection]

        raw_lines.append(line)
        ax_raw.set_xlabel("Time (s)")
        ax_raw.set_ylabel("Amplitude")
        ax_raw.grid(True)

        extent = [0, time_axis[-1], FREQ_RANGE[0], FREQ_RANGE[1]]
        log_spec_data_init = 10 * np.log10(spec_data + 1e-10)
        im = ax_spec.imshow(log_spec_data_init, aspect='auto', origin='lower', extent=extent, cmap='viridis')
        spec_images.append(im)
        ax_spec.set_title(f"Spectrogram - {plot_ch_name}")
        ax_spec.set_xlabel("Time (s)")
        ax_spec.set_ylabel("Frequency (Hz)")
        cbar = fig.colorbar(im, ax=ax_spec, fraction=0.046, pad=0.04)
        cbar.set_label('Power (dB)')

    def update_plot(val):
        idx = int(val)
        for i in range(n_display_rows):
            ax_raw, ax_spec = axes[i]
            if average:
                new_raw_data = selected_segments[idx].mean(axis=0)
                plot_ch_name = "Average"
                new_spec_data = selected_specs[idx].mean(axis=0)
                ax_raw.set_title(f"{plot_ch_name} - Segment {seg_indices[idx]}")
            else:
                channel_idx_in_selection = i
                new_raw_data = selected_segments[idx][channel_idx_in_selection]
                plot_ch_name = selected_ch_names[channel_idx_in_selection]
                new_spec_data = selected_specs[idx][channel_idx_in_selection]
                ax_raw.set_title(f"{plot_ch_name} - Segment {seg_indices[idx]}")
            raw_lines[i].set_ydata(new_raw_data)
            ax_raw.relim()
            ax_raw.autoscale_view()
            log_spec_data = 10 * np.log10(new_spec_data + 1e-10)
            spec_images[i].set_data(log_spec_data)
            spec_images[i].set_clim(vmin=np.min(log_spec_data), vmax=np.max(log_spec_data))
            ax_spec.set_title(f"Spectrogram - {plot_ch_name}")
        fig.canvas.draw_idle()

    if n_selected_segments > 1:
        slider_ax = fig.add_axes([0.2, 0.05, 0.6, 0.03])
        seg_slider = Slider(slider_ax, 'Segment', 0, n_selected_segments - 1, valinit=0, valstep=1)
        seg_slider.on_changed(update_plot)
    else:
        print("[Info] Only one segment selected. No slider needed.")

    plt.show(block=True)

# ------------------------------
# Pipeline per subject
# ------------------------------
def process_subject(subject_id, task_name, visualize=False):
    print(f"Processing Subject {subject_id}, Task {task_name}...")
    raw = load_raw_eeg(subject_id, task_name)
    segments = segment_signal(raw, WINDOW_LENGTH, STEP_SIZE)
    specs = np.array([compute_spectrogram(seg, sfreq=raw.info['sfreq'], nfft=NFFT, freq_range=FREQ_RANGE)
                      for seg in segments])
    specs_scaled = normalize_spectrograms(specs)

    if visualize:
        seg_indices, ch_indices, average = select_segments_and_regions(raw, segments)
        visualize_segments_interactive(segments, specs_scaled, raw, seg_indices, ch_indices, average)

    return specs_scaled

# ------------------------------
# Save dataset
# ------------------------------
def save_dataset(X, y, subject_id, task_name):
    subject_folder = f"S{subject_id:03d}"
    output_folder = os.path.join(BASE_OUTPUT_PATH, subject_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_file = os.path.join(output_folder, f"{subject_folder}_{task_name}_features.pkl")
    with open(output_file, "wb") as f:
        pickle.dump({"X": X, "y": y}, f)
    print(f"✅ Features saved to: {output_file}")

# ------------------------------
# Main Execution
# ------------------------------
if __name__ == "__main__":
    subject_id = int(input("Enter subject ID (1-109): "))
    
    while True:
        task_input = input("Select task: [0] Imagined Movement, [1] Actual Movement: ").strip()
        if task_input == '0':
            task_name = 'imagined_movement'
            label_value = 1
            break
        elif task_input == '1':
            task_name = 'actual_movement'
            label_value = 0
            break
        else:
            print("Invalid input. Enter 0 or 1.")

    X = process_subject(subject_id, task_name, visualize=VISUALIZE)
    y = np.full(X.shape[0], label_value, dtype=int)

    print(f"\nGenerated {X.shape[0]} feature vectors (spectrograms).")
    print(f"Assigned label {label_value} to all feature vectors for task '{task_name}'.")

    save_prompt = input("Save dataset? [Y/N]: ").strip().lower()
    if save_prompt == 'y':
        save_dataset(X, y, subject_id, task_name)
    else:
        print("Dataset not saved.")

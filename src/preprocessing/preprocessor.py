# =========================================================
# Imports & Config
# =========================================================
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations
from tqdm import tqdm
from scipy.signal import welch, spectrogram
from sklearn.preprocessing import MinMaxScaler
from matplotlib.widgets import Slider

import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch

# Interactive plotting for .py scripts
plt.ion()

# File paths
BASE_RAW_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\raw"
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

TASK_RUNS = {
    'imagined_movement': [4, 8, 12],
    'actual_movement': [3, 7, 11]
}

# Brainwave bands
BANDS = {
    'delta': (1, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 40)
}

# =========================================================
# Load & Combine Data per Subject & Task
# =========================================================
def process_and_save_subject_task(subject_id, task_name, runs, base_raw_path, base_output_path):
    print(f"--- Processing Subject {subject_id}, Task: {task_name} ---")
    subject_folder = f"S{subject_id:03d}"
    subject_folder_path = os.path.join(base_raw_path, subject_folder)
    
    raw_files = []
    for run_number in runs:
        file_name = f"{subject_folder}R{run_number:02d}.edf"
        file_path = os.path.join(subject_folder_path, file_name)
        raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto')
        raw_files.append(raw)
        
    raw_combined = mne.concatenate_raws(raw_files)
    
    output_folder = os.path.join(base_output_path, subject_folder)
    os.makedirs(output_folder, exist_ok=True)
    
    output_filename = f"{subject_folder}_{task_name}_raw.fif"
    output_path = os.path.join(output_folder, output_filename)
    
    raw_combined.save(output_path, overwrite=True)
    print(f"✅ Saved combined data to: {output_path}\n")
    return output_path

# Example: Process first subject
while True:
    try:
        sub_id = int(input("Pick a subject from (1 to 109): "))
        if 1 <= sub_id <= 109:
            subject_id = sub_id
            break
        else:
            print("Please enter a valid number from 1 to 109 ")
    except ValueError:
        print("Invalid input, please input a valid number: (1 to 109)")

raw_path_imagined = process_and_save_subject_task(
    subject_id, 'imagined_movement', TASK_RUNS['imagined_movement'], BASE_RAW_PATH, BASE_OUTPUT_PATH
)
raw_path_actual = process_and_save_subject_task(
    subject_id, 'actual_movement', TASK_RUNS['actual_movement'], BASE_RAW_PATH, BASE_OUTPUT_PATH
)


# =========================================================
# Load Processed Data for Analysis
# =========================================================

while True:
    try:
        task_picker = int(input("pick which task to analyse:\nImagined movement: 0 \nActual movement: 1:"))
        if task_picker == 0:
            task_picker = 'imagined_movement'
            break
        elif task_picker == 1:
            task_picker = 'actual_movement'
            break
        else:
            print("Please input a valid number: 0 or 1")
    except ValueError:
        print("Invali input, please input a vlid number: (0 or 1)")

task_name = task_picker
subject_folder = f"S{subject_id:03d}"
processed_file = os.path.join(BASE_OUTPUT_PATH, subject_folder, f"{subject_folder}_{task_name}_raw.fif")
raw = mne.io.read_raw_fif(processed_file, preload=True)

print(raw.info)
print(f"Data Duration: {raw.times[-1]/60:.2f} minutes")


# =========================================================
# Rename channels, set EEG type, and apply montage
# =========================================================
raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
raw.set_channel_types({ch: 'eeg' for ch in raw.ch_names})

montage = mne.channels.make_standard_montage('standard_1020')
raw.set_montage(montage, match_case=False, match_alias=True, on_missing='warn')

# List channels with missing positions
missing_positions= [ch['ch_name'] for ch in raw.info['chs'] if np.allclose(ch['loc'][:3], 0.0)]
print("Missing positions (will appear in middle of diagram):", missing_positions)


# =========================================================
# Visualize Raw Signals, PSD, and Montages (Interactive)
# =========================================================

# Interactive raw EEG (all channels)
raw.plot(n_channels=64, duration=10, scalings='auto', title='raw EEG signals', block=False)

# Interactive PSD
psd_before = raw.compute_psd(fmin=1, fmax=80, average='mean')
psd_before.plot()
plt.suptitle("PSD before filtering", fontsize=14)

#---------------------------------------------------------------------------------------------------------

# 2D montage (sensor layout)
raw.plot_sensors(show_names=True)

# 3D montage (optional, matplotlib)
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

pos = np.array([ch['loc'][:3] for ch in raw.info['chs']])
names = raw.ch_names

fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(pos[:, 0], pos[:, 1], pos[:, 2], s=60, c='r')

for i, name in enumerate(names):
    ax.text(pos[i, 0], pos[i, 1], pos[i, 2], name, fontsize=8)

ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.set_title("3D EEG Electrode Layout")

# =========================================================
# Bandpass + Notch Filtering
# =========================================================
raw_filtered = raw.copy().filter(l_freq=1., h_freq=40.)
raw_filtered.notch_filter(freqs=[50])

psd_after = raw_filtered.compute_psd(fmin=1, fmax=80, average='mean')
psd_after.plot()
plt.suptitle("PSD After Bandpass + Notch Filtering", fontsize=14)

# ICA decomposition
ica = ICA(n_components=32, random_state=97, max_iter=800)
ica.fit(raw_filtered)

# Plot ICA sources interactively to inspect components
ica.plot_sources(raw_filtered)
plt.show()

# Plot ICA component topographies (0-19)
ica.plot_components()
plt.show()

# ------------------------------
# Fit ICA
# ------------------------------
ica = ICA(n_components=32, random_state=97, max_iter=800)
ica.fit(raw_filtered)

sources = ica.get_sources(raw_filtered).get_data()
sfreq = raw_filtered.info['sfreq']
times = np.arange(sources.shape[1]) / sfreq
n_components = sources.shape[0]

# Compute PSD for each IC (extend frequency range if needed)
psds, freqs = psd_array_welch(sources, sfreq=sfreq, fmin=1, fmax=80, n_fft=2048)

# ------------------------------
# Interactive Plot Setup
# ------------------------------
fig, axes = plt.subplots(3, 1, figsize=(14, 12))
plt.subplots_adjust(bottom=0.25, hspace=0.4)

# Initial IC
current_ic = 0

# Topography
topo_ax = axes[0]
ica.plot_components(picks=current_ic, show=False, axes=topo_ax)
topo_ax.set_title(f'IC {current_ic} Topography')

# Time series
time_ax = axes[1]
t_min, t_max = 0, 40  # seconds
mask = (times >= t_min) & (times <= t_max)
time_line, = time_ax.plot(times[mask], sources[current_ic, mask])
time_ax.set_title(f'IC {current_ic} Time Series ({t_min}-{t_max}s)')
time_ax.set_xlabel('Time (s)')
time_ax.set_ylabel('Amplitude')

# PSD
psd_ax = axes[2]
psd_line, = psd_ax.semilogy(freqs, psds[current_ic])
psd_ax.set_title(f'IC {current_ic} PSD')
psd_ax.set_xlabel('Frequency (Hz)')
psd_ax.set_ylabel('PSD')
psd_ax.set_xlim(1, 50)

# ------------------------------
# IC Slider
# ------------------------------
ax_slider = plt.axes([0.2, 0.1, 0.6, 0.03])
ic_slider = Slider(ax_slider, 'IC', 0, n_components-1, valinit=0, valstep=1)

def update_ic(val):
    ic = int(ic_slider.val)
    
    # Update topography
    topo_ax.clear()
    ica.plot_components(picks=ic, show=False, axes=topo_ax)
    topo_ax.set_title(f'IC {ic} Topography')
    
    # Update time series
    time_line.set_ydata(sources[ic, mask])
    time_ax.set_title(f'IC {ic} Time Series ({t_min}-{t_max}s)')
    
    # Update PSD
    psd_line.set_ydata(psds[ic])
    psd_ax.set_title(f'IC {ic} PSD')
    
    fig.canvas.draw_idle()

ic_slider.on_changed(update_ic)

plt.show()


# Doing the exclusion section (removing bad components)

print("\n" + "="*30)
exclude_str = input("Enter the ICA component indices to exclude (comma-separated, e.g., 0,10,11,17): ")

ica_exclude_list = [] # Initialize empty list

try:
    # Split the input string by commas
    component_strings = exclude_str.split(',')

    # Loop through each piece of the split string
    for comp_str in component_strings:
        # Remove leading/trailing whitespace
        cleaned_str = comp_str.strip()
        
        # Check if the cleaned string contains only digits AND is not empty
        if cleaned_str.isdigit():
            # If yes, convert it to an integer
            component_number = int(cleaned_str)
            # Add it to our list
            ica_exclude_list.append(component_number)
        elif cleaned_str: # If the string is not empty but not digits, print a warning
            print(f"[Warning] Ignoring non-numeric input: '{cleaned_str}'")

    # Assign the final list to the ica object
    ica.exclude = ica_exclude_list
    print(f"Components selected for exclusion: {ica.exclude}")

except Exception as e: # Catch any unexpected error during processing
    print(f"An error occurred processing input: {e}")
    print("Invalid input format. No components excluded.")
    ica.exclude = [] # Default to excluding nothing if any error occurs

print("="*30 + "\n")

raw_cleaned = ica.apply(raw_filtered.copy())
raw_cleaned.plot()
plt.suptitle("After Excluding the componenets", fontsize=14)

# This works, looking at the graph of FP1 (eye blinks) By comparing old channel with new channel in the specific
# brain nodes or brain sections that correspond with the ICA components

plt.show(block=True)  # This is blocking only the 3D plot, others remain interactive


# --- Define subject folder and output directory ---
while True:
    save_prompt = input("Would you like to save? [Y/y] or [N/n]:").lower()
    if save_prompt == 'y':
        subject_folder = f"S{subject_id:03d}"
        output_folder = os.path.join(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed",
                                    subject_folder)
        os.makedirs(output_folder, exist_ok=True)

        # --- Define filename for ICA-cleaned data ---
        output_filename = f"{subject_folder}_{task_name}_cleaned.fif"
        output_path = os.path.join(output_folder, output_filename)

        # --- Save the cleaned EEG data ---
        raw_cleaned.save(output_path, overwrite=True)
        print(f"✅ Cleaned EEG data saved to: {output_path}")
        
        break

    elif save_prompt == 'n':
        print("Terminating the program")

        break
    
    else:
        print("Please input a valid option [Y/y] or [N/n]:")


            

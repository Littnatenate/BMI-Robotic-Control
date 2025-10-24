# =========================================================
# Automated EEG Preprocessing Pipeline
# =========================================================
import os
import mne
import numpy as np

# ------------------------------
# Config: Paths & Tasks
# ------------------------------
BASE_RAW_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\raw"
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

TASK_RUNS = {
    'imagined_movement': [4, 8, 12],
    'actual_movement': [3, 7, 11]
}

ICA_EXCLUDE_COMPONENTS = [0, 10, 11, 17, 18, 19]

# ------------------------------
# Function: Load & Combine Data
# ------------------------------
def process_subject_task(subject_id, task_name, runs):
    print(f"--- Processing Subject {subject_id}, Task: {task_name} ---")
    subject_folder = f"S{subject_id:03d}"
    subject_folder_path = os.path.join(BASE_RAW_PATH, subject_folder)
    
    # Load all runs
    raw_list = []
    for run_number in runs:
        file_name = f"{subject_folder}R{run_number:02d}.edf"
        file_path = os.path.join(subject_folder_path, file_name)
        raw = mne.io.read_raw_edf(file_path, preload=True, stim_channel='auto')
        raw_list.append(raw)
        
    # Concatenate runs
    raw_combined = mne.concatenate_raws(raw_list)
    
    # Channel Setup
    raw_combined.rename_channels(lambda name: name.replace('.', '').strip().upper())
    raw_combined.set_channel_types({ch: 'eeg' for ch in raw_combined.ch_names})
    montage = mne.channels.make_standard_montage('standard_1020')
    raw_combined.set_montage(montage, match_case=False, match_alias=True, on_missing='warn')
    
    # Filtering
    raw_filtered = raw_combined.copy().filter(l_freq=1., h_freq=40.)
    raw_filtered.notch_filter(freqs=[50])
    
    # ICA
    ica = mne.preprocessing.ICA(n_components=20, random_state=97, max_iter=800)
    ica.fit(raw_filtered)
    ica.exclude = ICA_EXCLUDE_COMPONENTS
    raw_cleaned = ica.apply(raw_filtered.copy())
    
    # Save cleaned data
    output_folder = os.path.join(BASE_OUTPUT_PATH, subject_folder)
    os.makedirs(output_folder, exist_ok=True)
    output_filename = f"{subject_folder}_{task_name}_cleaned.fif"
    output_path = os.path.join(output_folder, output_filename)
    raw_cleaned.save(output_path, overwrite=True)
    print(f"✅ Cleaned EEG data saved to: {output_path}\n")
    
    return output_path

# ------------------------------
# Run pipeline for all subjects & tasks
# ------------------------------
N_SUBJECTS = 109  # example: change to your total number of subjects

for subj in range(64, N_SUBJECTS + 1):
    for task, runs in TASK_RUNS.items():
        process_subject_task(subj, task, runs)



"""
Temporary use of this .py file to save all cleaned raw EEG data (filtered & ICA processed) into a separate folder.
2 Data run types (iimaginary & actual movements)
"""

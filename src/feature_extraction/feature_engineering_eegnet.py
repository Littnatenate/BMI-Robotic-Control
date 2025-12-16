"""
Processes the cleaned EEG data for Deep Learning models.
It performs the following steps:
1. Resampling to 160Hz.
2. Bandpass filtering (4-40Hz).
3. Epoch extraction (0.5s to 2.5s).
4. Data standardization (shape enforcement).
5. Feature serialization (.pkl).

Author: [Your Name/Lab]
"""

import sys
import os
import logging
import pickle
from pathlib import Path
import numpy as np
import mne
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(os.path.join(parent_dir, 'src'))

from config import PROCESSED_DATA_DIR, SUBJECTS, TASKS

# Configuration
OUTPUT_DIR = PROCESSED_DATA_DIR.parent / "processed_eegnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Signal Processing Parameters
TARGET_SFREQ = 160
FREQ_BAND = (4.0, 40.0)  # (Low, High)
EPOCH_WINDOW = (0.5, 2.5) # (Tmin, Tmax)

# Calculate expected samples: (2.5 - 0.5) * 160 = 320
EXPECTED_SAMPLES = int((EPOCH_WINDOW[1] - EPOCH_WINDOW[0]) * TARGET_SFREQ)

# Logging Setup
logging.basicConfig(evel=logging.INFO, format='%(message)s')
logger = logging.getLogger("EEGNet_Gen")
mne.set_log_level('ERROR')


def _preprocess_task_data(file_path, task_name):
    
    #Loads, filters, and epochs a single EEG file.
    #Returns X (features) and y (labels) or (None, None) if failed.

    try:
        raw = mne.io.read_raw_fif(file_path, preload=True)

        # Resample
        # Enforce target sampling rate to ensure consistent array sizes
        if raw.info['sfreq'] != TARGET_SFREQ:
            raw.resample(TARGET_SFREQ, npad="auto")

        # Filter
        # FIR bandpass filter (4-40Hz)
        raw.filter(FREQ_BAND[0], FREQ_BAND[1], fir_design='firwin', verbose=False)

        # Event Extraction
        events, event_id_map = mne.events_from_annotations(raw, verbose=False)
        
        # Identify specific event codes for T1 and T2
        #t1 = next((v for k, v in event_id_map.items() if 'T1' in k), None)
        #t2 = next((v for k, v in event_id_map.items() if 'T2' in k), None)
        
        t1= None

        for label, code in event_id_map.items():
            if 'T1' in label:
                t1 = code
                break # Once T1 is found, stop finding for it

        t2 = None

        for label, code in event_id_map.items():
            if 'T2' in label:
                t2 = code

        if not t1 or not t2:
            return None, None

        # 4. Epoching
        # Subtract one sample from tmax to handle MNE's inclusive slicing
        # ensuring the output length matches EXPECTED_SAMPLES exactly.
        epochs = mne.Epochs(
            raw, 
            events, 
            event_id={str(t1): t1, str(t2): t2}, 
            tmin=EPOCH_WINDOW[0], 
            tmax=EPOCH_WINDOW[1] - (1 / TARGET_SFREQ),
            baseline=None, 
            verbose=False
        )

        X = epochs.get_data(copy=True) * 1e6  # Scale V to uV
        y_raw = epochs.events[:, -1]

        # Shape Enforcement
        # Ensure exact time-dimension length (N, Channels, Time)
        current_samples = X.shape[2]
        
        if current_samples != EXPECTED_SAMPLES:
            if current_samples > EXPECTED_SAMPLES:
                # Trim excess samples
                X = X[:, :, :EXPECTED_SAMPLES]
            elif current_samples < EXPECTED_SAMPLES:
                # Edge pad if samples are missing (rare fallback)
                diff = EXPECTED_SAMPLES - current_samples # Calculates how much to add
                X = np.pad(X, ((0, 0), (0, 0), (0, diff)), mode='edge') # This is for N_Epochs, N_Channels, N_Timepoints

        # Label Mapping
        # Real Movement: 0, 1 | Imagined Movement: 2, 3
        y = np.zeros_like(y_raw)
        #base_offset = 2 if task_name == 'imagined_movement' else 0
        if task_name == 'imagined_movement':
            base_offset = 2
        else:
            base_offset = 0


        y[y_raw == t1] = base_offset + 0
        y[y_raw == t2] = base_offset + 1

        return X, y

    except Exception as e:
        logger.warning(f"Error processing {file_path.name}: {e}")
        return None, None


def process_subject(subject_id):
    
    #Aggregates all tasks for a subject and saves the feature set.
    
    sub_str = f"S{subject_id:03d}"
    clean_dir = PROCESSED_DATA_DIR / sub_str
    
    subject_X = []
    subject_y = []
    
    for task_name in TASKS.keys():
        file_path = clean_dir / f"{sub_str}_{task_name}_cleaned_eeg.fif"
        
        if not file_path.exists():
            continue
        
        X, y = _preprocess_task_data(file_path, task_name)
        
        if X is not None:
            subject_X.append(X)
            subject_y.append(y)

    if not subject_X:
        return

    # Concatenate all tasks
    X_final = np.concatenate(subject_X, axis=0)
    y_final = np.concatenate(subject_y, axis=0)
    
    # Save to disk
    out_sub_dir = OUTPUT_DIR / sub_str
    out_sub_dir.mkdir(parents=True, exist_ok=True)
    
    save_path = out_sub_dir / f"{sub_str}_eegnet_features.pkl"
    
    with open(save_path, 'wb') as f:
        pickle.dump({
            'X': X_final.astype(np.float32), 
            'y': y_final.astype(np.int64)
        }, f)


if __name__ == "__main__":
    logger.info(f"Starting EEGNet Feature Extraction")
    logger.info(f"Target Shape: {EXPECTED_SAMPLES} samples @ {TARGET_SFREQ}Hz")
    
    for sub in tqdm(SUBJECTS, desc="Processing Subjects"):
        process_subject(sub)
        
    logger.info("Processing complete.")
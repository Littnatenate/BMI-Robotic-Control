"""
FEATURE EXTRACTION: EEGNET (RAW TIME-SERIES)
============================================
Description:
    1. Loads cleaned EEG data.
    2. Bandpass filters 4-40Hz (Standard for Motor Imagery to capture Mu/Beta).
    3. Epochs the data (Raw Voltage) without converting to Spectrograms.
    4. Saves as .pkl files specifically for EEGNet training.

Output Shape: (Trials, Channels, TimePoints) -> e.g., (N, 64, 320)
"""

import os
import numpy as np
import mne
import pickle
import logging
from tqdm import tqdm
from pathlib import Path

# --- Configuration ---
BASE_RAW_PATH = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed")
BASE_OUTPUT_PATH = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed_eegnet")
BASE_OUTPUT_PATH.mkdir(parents=True, exist_ok=True)

SUBJECT_RANGE = range(1, 110)
EPOCH_DURATION = 2.0
TMIN = 0.5 
TMAX = TMIN + EPOCH_DURATION 

# EEGNet Settings
# We filter 4-40Hz to isolate Mu (8-12) and Beta (13-30) rhythms, 
# while removing low-frequency drift (<4Hz) and high-frequency noise (>40Hz).
L_FREQ = 4.0
H_FREQ = 40.0
TARGET_SFREQ = 160  # Standardize sampling rate to 160Hz

LABEL_MAP = {
    'imagined_movement': {'left': 0, 'right': 1},
    'actual_movement':   {'left': 0, 'right': 1} 
}
TASKS_TO_PROCESS = ['imagined_movement', 'actual_movement'] 

# --- Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

def process_subject_eegnet(subject_id):
    sub_str = f"S{subject_id:03d}"
    all_X = []
    all_y = []
    
    for task in TASKS_TO_PROCESS:
        # FIX: Use correct filename _cleaned_eeg.fif
        file_path = BASE_RAW_PATH / sub_str / f"{sub_str}_{task}_cleaned_eeg.fif"
        
        if not file_path.exists():
            continue
            
        try:
            raw = mne.io.read_raw_fif(file_path, preload=True, verbose='ERROR')
            
            # 1. Resample (Standardize input size for Neural Net)
            if raw.info['sfreq'] != TARGET_SFREQ:
                raw.resample(TARGET_SFREQ, npad="auto", verbose='ERROR')
            
            # 2. Filter (4-40Hz)
            raw.filter(L_FREQ, H_FREQ, fir_design='firwin', skip_by_annotation='edge', verbose='ERROR')
            
            # 3. Find Events (Robust Method for PhysioNet)
            events, event_id_map = mne.events_from_annotations(raw, verbose=False)
            
            # Robustly find T1/T2 even if named "Event/T1" or similar
            t1_key = next((k for k in event_id_map if 'T1' in k), None)
            t2_key = next((k for k in event_id_map if 'T2' in k), None)
            
            if not t1_key or not t2_key:
                continue
                
            event_dict = {'T1': event_id_map[t1_key], 'T2': event_id_map[t2_key]}
            
            # 4. Create Epochs
            epochs = mne.Epochs(raw, events, event_dict, 
                                tmin=TMIN, tmax=TMAX - (1/raw.info['sfreq']),
                                baseline=None, preload=True, verbose='ERROR')
            
            # 5. Get Data (Microvolts)
            # Scaling by 1e6 converts Volts to Microvolts.
            # Neural Networks converge faster with values like 10.0 instead of 0.00001
            X = epochs.get_data(copy=True) * 1e6 
            y_raw = epochs.events[:, -1]
            
            # 6. Remap Labels
            y = np.zeros_like(y_raw)
            # Map T1 -> Class ID
            y[y_raw == event_dict['T1']] = LABEL_MAP[task]['left']
            y[y_raw == event_dict['T2']] = LABEL_MAP[task]['right']
            
            all_X.append(X)
            all_y.append(y)
            
        except Exception as e:
            logger.error(f"Error S{subject_id}: {e}")
            continue

    if not all_X:
        return False
        
    X_final = np.concatenate(all_X, axis=0)
    y_final = np.concatenate(all_y, axis=0)
    
    # Save
    subj_dir = BASE_OUTPUT_PATH / sub_str
    subj_dir.mkdir(exist_ok=True)
    
    save_path = subj_dir / f"{sub_str}_eegnet_features.pkl"
    with open(save_path, 'wb') as f:
        pickle.dump({'X': X_final.astype(np.float32), 'y': y_final.astype(np.int64)}, f)
        
    return True

if __name__ == "__main__":
    logger.info("="*60)
    logger.info(f"STARTING EEGNET FEATURE EXTRACTION")
    logger.info(f"Output: {BASE_OUTPUT_PATH}")
    logger.info("="*60)
    
    count = 0
    for subj in tqdm(SUBJECT_RANGE, desc="Processing"):
        if process_subject_eegnet(subj):
            count += 1
            
    logger.info(f"Done. Processed {count}/{len(SUBJECT_RANGE)} subjects.")
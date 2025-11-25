"""
FEATURE ENGINEERING V5 (FINAL REPAIR): SHAPE ENFORCEMENT
========================================================
Description:
    1. Loads raw files.
    2. Converts to Spectrograms.
    3. ***CRITICAL FIX***: Forces every single image to be exactly (32, 4).
    4. Saves as .pkl.
"""

import os
import pickle
import numpy as np
import mne
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path
import logging

# --- CONFIGURATION ---
BASE_OUTPUT_PATH = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed")
SUBJECT_RANGE = range(1, 110) # Run everyone to be safe
TASKS_TO_PROCESS = ['actual_movement', 'imagined_movement']

# Spectrogram Settings
FREQ_RANGE = (1, 40)
NFFT = 128            
NOVERLAP = 64         

# *** THE FIX ***
TARGET_SHAPE = (32, 4) 

LABEL_MAP = {
    ('actual_movement', 'left'): 0,
    ('actual_movement', 'right'): 1,
    ('imagined_movement', 'left'): 2,
    ('imagined_movement', 'right'): 3,
}

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("FE_V5_Repair")

def load_cleaned_eeg(subject_id, task_name):
    sub_str = f"S{subject_id:03d}"
    file_path = BASE_OUTPUT_PATH / sub_str / f"{sub_str}_{task_name}_cleaned_eeg.fif"
    if not file_path.exists(): return None
    try: return mne.io.read_raw_fif(file_path, preload=True, verbose='ERROR')
    except: return None

def pad_or_trim(spec, target_shape):
    """Forces spectrogram to match target_shape (32, 4)."""
    f_tgt, t_tgt = target_shape
    f_curr, t_curr = spec.shape
    
    # Trim
    if f_curr > f_tgt: spec = spec[:f_tgt, :]
    if t_curr > t_tgt: spec = spec[:, :t_tgt]
    
    # Pad
    if spec.shape != target_shape:
        pad_val = np.min(spec)
        new_spec = np.full(target_shape, pad_val, dtype=spec.dtype)
        new_spec[:spec.shape[0], :spec.shape[1]] = spec
        return new_spec
    return spec

def compute_spectrogram(epoch_data, sfreq):
    n_channels = epoch_data.shape[0]
    specs = []
    for ch in range(n_channels):
        f, t, Sxx = spectrogram(epoch_data[ch], fs=sfreq, nperseg=NFFT, noverlap=NOVERLAP)
        mask = (f >= FREQ_RANGE[0]) & (f <= FREQ_RANGE[1])
        Sxx_db = 10 * np.log10(Sxx[mask, :] + 1e-12)
        # APPLY FIX
        specs.append(pad_or_trim(Sxx_db, TARGET_SHAPE))
    return np.array(specs) 

def process_subject(subject_id):
    features, labels, channel_names = [], [], None
    
    for task in TASKS_TO_PROCESS:
        raw = load_cleaned_eeg(subject_id, task)
        if not raw: continue
        if channel_names is None: channel_names = raw.ch_names
        
        try:
            events, event_id_map = mne.events_from_annotations(raw, verbose=False)
            t1 = next((v for k, v in event_id_map.items() if 'T1' in k), None)
            t2 = next((v for k, v in event_id_map.items() if 'T2' in k), None)
            if not t1 or not t2: continue
            
            epochs = mne.Epochs(raw, events, {'L': t1, 'R': t2}, tmin=0.5, tmax=2.5, baseline=None, verbose='ERROR')
            data = epochs.get_data(copy=False)
            ev = epochs.events[:, -1]
            inv_map = {t1: 'left', t2: 'right'}
            
            for i in range(len(data)):
                spec = compute_spectrogram(data[i], raw.info['sfreq'])
                lbl = LABEL_MAP.get((task, inv_map.get(ev[i])))
                if lbl is not None:
                    features.append(spec)
                    labels.append(lbl)
        except: continue

    if not features: return
    
    # Standardization
    X = np.array(features)
    y = np.array(labels)
    
    X_out = np.zeros_like(X)
    for ch in range(X.shape[1]):
        scaler = StandardScaler()
        flat = X[:, ch, :, :].reshape(-1, 1)
        X_out[:, ch, :, :] = scaler.fit_transform(flat).reshape(X.shape[0], X.shape[2], X.shape[3])

    # Save
    out_dir = BASE_OUTPUT_PATH / f"S{subject_id:03d}"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / f"S{subject_id:03d}_spectrograms.pkl", 'wb') as f:
        pickle.dump({'X': X_out, 'y': y, 'channels': channel_names, 'class_map': LABEL_MAP}, f)

if __name__ == '__main__':
    print(f"--- FIXING DATASET SHAPES TO {TARGET_SHAPE} ---")
    for sub in tqdm(SUBJECT_RANGE): process_subject(sub)
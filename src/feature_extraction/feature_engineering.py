"""
FEATURE ENGINEERING: SPECTROGRAMS (GAP-CNN)
===========================================
1. Loads Cleaned EEG (.fif).
2. Epochs the data (T1/T2 events).
3. Computes STFT (Short-Time Fourier Transform).
4. Converts to Log-Scale (dB) for visibility.
5. Standardizes (Z-Score) for Neural Net stability.
6. Saves as .pkl.
"""

import sys
import os

# --- PATH FIX ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir)) # Go up two levels to find src
sys.path.append(os.path.join(parent_dir, 'src'))

import numpy as np
import mne
import pickle
import logging
from scipy.signal import spectrogram
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from pathlib import Path

# Import Config
from config import (
    PROCESSED_DATA_DIR, 
    SUBJECTS, TASKS, LABEL_MAP
)

# --- SPECTROGRAM SETTINGS ---
# Refined for Motor Imagery (4-40Hz)
FREQ_RANGE = (4, 40)
NFFT = 128            
NOVERLAP = 64         
TARGET_SHAPE = (32, 40) # (FreqBins, TimeBins) - Standardized size for CNN

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger("Spectrogram_Gen")
mne.set_log_level('ERROR')

def process_subject(subject_id):
    sub_str = f"S{subject_id:03d}"
    sub_dir = PROCESSED_DATA_DIR / sub_str
    
    features = []
    labels = []
    
    # Process both Actual and Imagined tasks
    for task in TASKS.keys():
        file_path = sub_dir / f"{sub_str}_{task}_cleaned_eeg.fif"
        
        if not file_path.exists(): continue
        
        try:
            raw = mne.io.read_raw_fif(file_path, preload=True)
            
            # 1. Extract Events (T1=Left, T2=Right)
            events, event_id_map = mne.events_from_annotations(raw, verbose=False)
            
            # Robust mapping (Handle different naming variations)
            t1_id = next((v for k, v in event_id_map.items() if 'T1' in k), None)
            t2_id = next((v for k, v in event_id_map.items() if 'T2' in k), None)
            
            if not t1_id or not t2_id: continue
            
            # 2. Epoching (0.5s to 2.5s)
            # We cut 2 seconds of data where the user is "Imagining"
            epochs = mne.Epochs(raw, events, event_id={str(t1_id): t1_id, str(t2_id): t2_id}, 
                                tmin=0.5, tmax=2.5, baseline=None, verbose=False)
            
            data = epochs.get_data(copy=True)
            event_list = epochs.events[:, -1]
            
            # 3. Compute Spectrograms
            for i in range(len(data)):
                epoch_data = data[i] # Shape: (Channels, Time)
                trial_specs = []
                
                for ch_idx in range(epoch_data.shape[0]):
                    f, t, Sxx = spectrogram(epoch_data[ch_idx], fs=raw.info['sfreq'], 
                                          nperseg=NFFT, noverlap=NOVERLAP)
                    
                    # Bandpass Mask (4-40Hz)
                    mask = (f >= FREQ_RANGE[0]) & (f <= FREQ_RANGE[1])
                    Sxx_roi = Sxx[mask, :]
                    
                    # Log-Scale (dB) -> Makes low-power signals visible
                    Sxx_db = 10 * np.log10(Sxx_roi + 1e-12)
                    
                    # Resize to Fixed Shape (pad or trim)
                    Sxx_fixed = _resize_spec(Sxx_db, TARGET_SHAPE)
                    trial_specs.append(Sxx_fixed)
                
                features.append(np.array(trial_specs))
                
                # Label Mapping: T1->0 (Left), T2->1 (Right)
                # We encode task type into label: 0/1=Actual, 2/3=Imagined
                base_label = 0 if event_list[i] == t1_id else 1
                offset = 2 if task == 'imagined_movement' else 0
                labels.append(base_label + offset)

        except Exception as e:
            logger.warning(f"Error {sub_str}: {e}")
            continue

    if not features: return

    # 4. Standardization (Z-Score per channel)
    # Critical for CNN convergence
    X = np.array(features) # (N, Ch, Freq, Time)
    y = np.array(labels)
    
    X_scaled = np.zeros_like(X)
    for ch in range(X.shape[1]):
        scaler = StandardScaler()
        # Flatten (N, F, T) -> (N*T, F) to fit scaler
        ch_data = X[:, ch, :, :]
        shape = ch_data.shape
        flat = ch_data.reshape(-1, 1)
        scaled = scaler.fit_transform(flat).reshape(shape)
        X_scaled[:, ch, :, :] = scaled

    # 5. Save
    out_file = sub_dir / f"{sub_str}_spectrograms.pkl"
    with open(out_file, 'wb') as f:
        pickle.dump({'X': X_scaled, 'y': y, 'channels': raw.ch_names}, f)

def _resize_spec(spec, target_shape):
    """Ensures every image is exactly (32, 40)."""
    curr_f, curr_t = spec.shape
    tgt_f, tgt_t = target_shape
    
    # Trim Freqs
    if curr_f > tgt_f: spec = spec[:tgt_f, :]
    
    # Pad Freqs
    if curr_f < tgt_f:
        pad = np.min(spec)
        spec = np.pad(spec, ((0, tgt_f - curr_f), (0, 0)), constant_values=pad)
        
    # Trim Time
    if curr_t > tgt_t: spec = spec[:, :tgt_t]
        
    # Pad Time
    if curr_t < tgt_t:
        pad = np.min(spec)
        spec = np.pad(spec, ((0, 0), (0, tgt_t - curr_t)), constant_values=pad)
        
    return spec

if __name__ == "__main__":
    logger.info("STARTING SPECTROGRAM GENERATION")
    for sub in tqdm(SUBJECTS):
        process_subject(sub)
    logger.info("Done.")
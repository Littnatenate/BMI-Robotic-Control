"""
AUTOMATED CLEANING PIPELINE
===========================
1. Loads Raw EDFs (concatenating runs).
2. Standardizes Names (10-20 system).
3. Filters Data (1-40Hz + Notch).
4. Robust ICA: Removes Blinks, Muscle, and Line Noise.
5. Saves as .fif for Feature Extraction.
"""

import sys
import os

# Path setup to allow importing from src/config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import warnings
import logging
import numpy as np
import mne
from mne.preprocessing import ICA
from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm

from config import (
    RAW_DATA_DIR, PROCESSED_DATA_DIR, 
    SUBJECTS, TASKS, CONFIG_PREPROC
)

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("Cleaner")
mne.set_log_level('ERROR')
warnings.filterwarnings("ignore")

def process_single_subject(subject_id):
    """
    Worker function: Processes one subject completely.
    """
    sub_str = f"S{subject_id:03d}"
    results = {}

    out_dir = PROCESSED_DATA_DIR / sub_str
    out_dir.mkdir(parents=True, exist_ok=True)

    for task_name, run_indices in TASKS.items():
        try:
            out_file = out_dir / f"{sub_str}_{task_name}_cleaned_eeg.fif"
            
            if out_file.exists():
                results[task_name] = "Skipped"
                continue

            # 1. Load & Concatenate
            raw = _load_and_concat(subject_id, run_indices)
            if not raw:
                results[task_name] = "Failed (Load)"
                continue

            # 2. Filtering
            raw.filter(CONFIG_PREPROC['l_freq'], CONFIG_PREPROC['h_freq'], fir_design='firwin')
            raw.notch_filter(CONFIG_PREPROC['notch_freq'])

            # 3. ICA (Artifact Removal)
            raw_ica_fit = raw.copy() 
            
            ica = ICA(n_components=CONFIG_PREPROC['ica_n_components'], 
                      method=CONFIG_PREPROC['ica_method'], 
                      max_iter=CONFIG_PREPROC['ica_max_iter'],
                      random_state=97)
            ica.fit(raw_ica_fit)

            exclude = []

            # A. EOG (Blinks)
            eog_inds, _ = ica.find_bads_eog(raw, ch_name=['Fp1', 'Fpz', 'Fp2'], 
                                          threshold=CONFIG_PREPROC['eog_threshold'])
            if eog_inds: exclude.extend(eog_inds)

            # B. Muscle (High Freq)
            muscle_inds = _find_muscle_artifacts(ica, raw)
            if muscle_inds: exclude.extend(muscle_inds)

            # C. Line Noise
            line_inds = _find_line_noise_components(ica, raw)
            if line_inds: exclude.extend(line_inds)

            exclude = sorted(list(set(exclude)))
            if len(exclude) > CONFIG_PREPROC['max_exclude']:
                exclude = exclude[:CONFIG_PREPROC['max_exclude']]

            ica.apply(raw, exclude=exclude)

            # 4. Save
            # Resample to target freq before saving
            if raw.info['sfreq'] != CONFIG_PREPROC['target_sfreq']:
                raw.resample(CONFIG_PREPROC['target_sfreq'])
                
            raw.save(out_file, overwrite=True)
            results[task_name] = f"Success (Rem: {len(exclude)})"

        except Exception as e:
            results[task_name] = f"Error: {str(e)}"

    return subject_id, results

# --- HELPER FUNCTIONS ---

def _load_and_concat(subject_id, runs):
    sub_str = f"S{subject_id:03d}"
    raw_list = []
    
    for run_num in runs:
        fpath = RAW_DATA_DIR / sub_str / f"{sub_str}R{run_num:02d}.edf"
        if fpath.exists():
            try:
                raw = mne.io.read_raw_edf(fpath, preload=True, verbose=False)
                raw_list.append(raw)
            except: pass
    
    if not raw_list: return None
    
    combined = mne.concatenate_raws(raw_list)
    
    def smart_rename(name):
        return name.replace('.', '').strip().upper().replace('FP', 'Fp').replace('Z', 'z')
    
    combined.rename_channels(smart_rename)
    try:
        combined.set_montage('standard_1005', on_missing='warn')
    except: pass
    
    return combined

def _find_muscle_artifacts(ica, raw):
    muscle_idx = []
    try:
        sources = ica.get_sources(raw).get_data()
        sfreq = raw.info['sfreq']
        
        for i, comp in enumerate(sources):
            freqs, psd = signal.welch(comp, fs=sfreq, nperseg=256)
            high_p = np.mean(psd[(freqs >= 20) & (freqs <= 40)])
            low_p = np.mean(psd[(freqs >= 1) & (freqs <= 20)])
            
            if low_p > 0 and (high_p / low_p) > CONFIG_PREPROC['muscle_threshold']:
                muscle_idx.append(i)
    except: pass
    return muscle_idx

def _find_line_noise_components(ica, raw):
    line_indices = []
    try:
        sources = ica.get_sources(raw).get_data()
        sfreq = raw.info['sfreq']
        
        for idx, comp in enumerate(sources):
            freqs, psd = signal.welch(comp, fs=sfreq, nperseg=256)
            line_idx = np.argmin(np.abs(freqs - 50.0))
            line_power = psd[line_idx]
            surrounding_power = np.mean(psd[(freqs >= 45) & (freqs <= 55)])
            
            if surrounding_power > 0 and (line_power / surrounding_power) > 4.0:
                line_indices.append(idx)
    except: pass
    return line_indices

if __name__ == "__main__":
    logger.info(f"STARTING CLEANING PIPELINE | {len(SUBJECTS)} SUBJECTS")
    
    results = Parallel(n_jobs=-1)(
        delayed(process_single_subject)(sub) for sub in tqdm(SUBJECTS)
    )
    
    logger.info("Pipeline Complete.")
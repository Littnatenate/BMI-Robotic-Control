import mne
import matplotlib
matplotlib.use('Agg')  # No pop-ups
import matplotlib.pyplot as plt
import os
import numpy as np
from mne.preprocessing import ICA
from scipy import signal
from tqdm import tqdm

# --- PATHS ---
BASE_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"
OUTPUT_IMG_DIR = os.path.join(BASE_PATH, "comparison_full_diagnostic")
os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

# --- CONFIGURATION ---
SUBJECT_RANGE = range(1, 110)
TASK_NAME = 'imagined_movement'

# --- NEW SETTINGS TO VALIDATE ---
NEW_EOG_THRESHOLD = 2.5      # Aggressive on Blinks
NEW_MUSCLE_THRESHOLD = 1.5   # Aggressive on Muscle (Need to verify this isn't too strong!)
MAX_REMOVE = 8               # Max components to remove
ICA_ITER = 1500              # High iteration count for convergence

# --- HELPER: MUSCLE DETECTION ---
def find_muscle_artifacts(ica, raw, threshold=NEW_MUSCLE_THRESHOLD):
    """
    Detects muscle noise by checking if High Freq Power > Low Freq Power * Threshold.
    """
    muscle_indices = []
    sfreq = raw.info['sfreq']
    try:
        sources = ica.get_sources(raw).get_data()
        for idx in range(sources.shape[0]):
            freqs, psd = signal.welch(sources[idx], fs=sfreq, nperseg=min(2048, len(sources[idx])))
            
            # Muscle is broad high freq (20-40Hz+), Brain is low freq (1-20Hz)
            high_mask = (freqs >= 20) & (freqs <= 40)
            low_mask = (freqs >= 1) & (freqs <= 20)
            
            if np.any(high_mask) and np.any(low_mask):
                high_power = np.mean(psd[high_mask])
                low_power = np.mean(psd[low_mask])
                
                if low_power > 0:
                    ratio = high_power / low_power
                    if ratio > threshold:
                        muscle_indices.append(idx)
    except Exception as e:
        pass # Fail silently for preview
    return muscle_indices

def sanitize_info(raw):
    raw.rename_channels(lambda name: name.replace('.', '').strip().upper())
    new_bads = [b.replace('.', '').strip().upper() for b in raw.info['bads'] if b.replace('.', '').strip().upper() in raw.ch_names]
    raw.info['bads'] = new_bads
    return raw

def check_subject(subject_id):
    subject_folder = f"S{subject_id:03d}"
    raw_path = os.path.join(BASE_PATH, subject_folder, f"{subject_folder}_{TASK_NAME}_raw.fif")
    old_clean_path = os.path.join(BASE_PATH, subject_folder, f"{subject_folder}_{TASK_NAME}_auto_cleaned.fif")

    if not os.path.exists(raw_path) or not os.path.exists(old_clean_path): return 

    try:
        # Load
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose='ERROR')
        old_clean = mne.io.read_raw_fif(old_clean_path, preload=True, verbose='ERROR')
        
        # Sanitize
        raw = sanitize_info(raw)
        old_clean = sanitize_info(old_clean)

        # PREPARE NEW CLEANING
        raw_for_new = raw.copy()
        raw_for_new.interpolate_bads(reset_bads=True, verbose='ERROR')
        raw_for_new.set_eeg_reference('average', projection=True, verbose='ERROR')
        
        # Filter 1Hz for ICA fit
        raw_ica_fit = raw_for_new.copy().filter(l_freq=1.0, h_freq=40.0, verbose='ERROR')
        
        # Fit ICA
        ica = ICA(n_components=30, method='fastica', random_state=97, max_iter=ICA_ITER)
        ica.fit(raw_ica_fit, verbose='ERROR')
        
        # 1. BLINKS (Proxy)
        eog_proxy = None
        for ch in ['FP1', 'FP2', 'FPZ', 'Fpz', 'AF3', 'AF4', 'FZ']:
            if ch in raw_ica_fit.ch_names: eog_proxy = ch; break
        eog_inds = []
        if eog_proxy:
            eog_inds, _ = ica.find_bads_eog(raw_ica_fit, ch_name=eog_proxy, threshold=NEW_EOG_THRESHOLD, verbose='ERROR')

        # 2. MUSCLE (Spectral)
        muscle_inds = find_muscle_artifacts(ica, raw_ica_fit, threshold=NEW_MUSCLE_THRESHOLD)

        # Combine
        exclude = list(set(eog_inds + muscle_inds))
        if len(exclude) > MAX_REMOVE: exclude = exclude[:MAX_REMOVE]
        
        # Apply
        new_clean = raw_for_new.copy()
        ica.apply(new_clean, exclude=exclude, verbose='ERROR')

        # --- PLOTTING: CHECK MOTOR CORTEX (C3) ---
        # This is where the "Imagination" signal lives. 
        # If this is flat, we over-cleaned.
        picks = ['C3', 'C4', 'CZ', 'CP3', 'CP4'] 
        present = [ch for ch in picks if ch in raw.ch_names]
        chan = present[0] if present else raw.ch_names[0]

        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot a 10s window from the middle (usually less blinks, more brain)
        t_mid = raw.times[-1] / 2
        t_start, t_end = t_mid, t_mid + 10
        
        r_data = raw.copy().crop(t_start, t_end).get_data(picks=chan)[0] * 1e6
        o_data = old_clean.copy().crop(t_start, t_end).get_data(picks=chan)[0] * 1e6
        n_data = new_clean.copy().crop(t_start, t_end).get_data(picks=chan)[0] * 1e6
        times = raw.copy().crop(t_start, t_end).times

        # PLOT LAYERS
        ax.plot(times, r_data, color='black', label='Raw (Dirty)', linewidth=2.5, zorder=1)
        ax.plot(times, o_data, color='red', label='Old Clean', linewidth=1.5, linestyle='--', alpha=0.85, zorder=2)
        ax.plot(times, n_data, color='blue', label='New Clean', linewidth=1.5, alpha=0.9, zorder=3)
        
        # Title Metrics
        title_str = (f"S{subject_id:03d} [{chan}] - Removed Total: {len(exclude)}\n"
                     f"Breakdown: Blinks (EOG): {len(eog_inds)} | Muscle: {len(muscle_inds)}")
        
        ax.set_title(title_str)
        ax.set_ylabel("Amplitude (µV)")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle=':', alpha=0.6)
        
        save_path = os.path.join(OUTPUT_IMG_DIR, f"S{subject_id:03d}_full_check.png")
        plt.savefig(save_path, dpi=100)
        plt.close(fig)
        
    except Exception as e:
        print(f"Failed on S{subject_id}: {e}")

if __name__ == "__main__":
    print(f"Generating FULL DIAGNOSTIC images in: {OUTPUT_IMG_DIR}")
    for subj in tqdm(SUBJECT_RANGE):
        check_subject(subj)
    print("Done!")
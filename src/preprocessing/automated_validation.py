"""
AUTOMATED VALIDATION PIPELINE
=============================
Generates 'Evidence Reports' to verify cleaning quality.
1. Plots PSD (Red=Raw vs Blue=Clean).
2. Checks for Variance Reduction.
3. Checks Time Domain for Artifact Removal.
"""

import sys
import os

# Path setup to allow importing from src/config.py
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import logging
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import mne
from pathlib import Path

from config import (
    PROCESSED_DATA_DIR, RAW_DATA_DIR, 
    SUBJECTS, TASKS, CONFIG_PREPROC
)

logger = logging.getLogger("Validator")

def validate_subject(subject_id, task='imagined_movement'):
    sub_str = f"S{subject_id:03d}"
    
    clean_path = PROCESSED_DATA_DIR / sub_str / f"{sub_str}_{task}_cleaned_eeg.fif"
    
    # Reconstruct raw data for comparison
    raw_list = []
    for r in TASKS[task]:
        p = RAW_DATA_DIR / sub_str / f"{sub_str}R{r:02d}.edf"
        if p.exists(): raw_list.append(mne.io.read_raw_edf(p, preload=True, verbose=False))
    
    if not clean_path.exists() or not raw_list:
        return None

    raw = mne.concatenate_raws(raw_list, verbose=False)
    clean = mne.io.read_raw_fif(clean_path, preload=True, verbose=False)

    _sanitize_names(raw)

    # Plotting
    fig = plt.figure(figsize=(16, 12))
    
    # 1. PSD
    ax1 = plt.subplot(2, 2, 1)
    raw.compute_psd(fmax=60, verbose=False).plot(axes=ax1, color='r', show=False, average=True, spatial_colors=False)
    clean.compute_psd(fmax=60, verbose=False).plot(axes=ax1, color='b', show=False, average=True, spatial_colors=False)
    ax1.set_title(f"PSD Comparison (Red=Raw, Blue=Clean)")
    ax1.legend(['Raw', 'Clean'])

    # 2. Variance Reduction
    ax2 = plt.subplot(2, 2, 2)
    var_pre = np.var(raw.get_data(), axis=1)
    var_post = np.var(clean.get_data(), axis=1)
    ax2.scatter(var_pre, var_post, alpha=0.5, c='purple')
    m = max(var_pre.max(), var_post.max())
    ax2.plot([0, m], [0, m], 'k--')
    ax2.set_title("Variance Reduction")
    ax2.set_xlabel("Raw Variance"); ax2.set_ylabel("Clean Variance")
    ax2.set_xscale('log'); ax2.set_yscale('log')

    # 3. Time Domain
    ax3 = plt.subplot(2, 1, 2)
    picks = [ch for ch in ['Fp1', 'Fpz', 'Fp2', 'Fz'] if ch in raw.ch_names]
    target_ch = picks[0] if picks else raw.ch_names[0]
    
    start = raw.times[-1] / 2
    raw_d = raw.copy().crop(start, start+5).get_data(picks=target_ch)[0]
    cln_d = clean.copy().crop(start, start+5).get_data(picks=target_ch)[0]
    t = np.linspace(0, 5, len(raw_d))
    
    ax3.plot(t, raw_d, 'r', alpha=0.4, label='Raw')
    ax3.plot(t, cln_d, 'b', label='Clean')
    ax3.set_title(f"Time Domain: Artifact Removal ({target_ch})")
    ax3.legend()

    report_dir = PROCESSED_DATA_DIR / "validation_reports"
    report_dir.mkdir(exist_ok=True)
    plt.tight_layout()
    plt.savefig(report_dir / f"{sub_str}_{task}_validation.png")
    plt.close()
    
    return True

def _sanitize_names(inst):
    def smart_rename(name):
        return name.replace('.', '').strip().upper().replace('FP', 'Fp').replace('Z', 'z')
    inst.rename_channels(smart_rename)

if __name__ == "__main__":
    print(f"Generating Validation Reports...")
    for sub in SUBJECTS[:5]: 
        try:
            validate_subject(sub)
            print(f"Validated S{sub:03d}")
        except Exception as e:
            print(f"Error S{sub:03d}: {e}")
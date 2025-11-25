"""
Saves 3 pdf files pre ICA exclusion
IC Time series
IC PSD
Topography map
"""


# =========================================================
# Imports
# =========================================================
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import mne
from mne.preprocessing import ICA
from mne.time_frequency import psd_array_welch

# =========================================================
# Parameters
# =========================================================
BASE_RAW_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\raw"
BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed"

SUBJECT_RANGE = range(104, 110)  # range of subjects
TIME_WINDOW = 80  # seconds to plot in time series

TASK_RUNS = {
    'imagined_movement': [4, 8, 12],
    'actual_movement': [3, 7, 11]
}

# =========================================================
# Function to load and concatenate runs
# =========================================================
def load_concat_runs(subject_folder, runs):
    raws = []
    for run in runs:
        run_file = os.path.join(BASE_RAW_PATH, subject_folder, f"{subject_folder}R{run:02d}.edf")
        if os.path.exists(run_file):
            raw = mne.io.read_raw_edf(run_file, preload=True)
            raws.append(raw)
        else:
            print(f"[Skipped] {run_file} not found")
    if not raws:
        return None
    return mne.concatenate_raws(raws)

# =========================================================
# Batch Processing
# =========================================================
for subject_id in SUBJECT_RANGE:
    subject_folder = f"S{subject_id:03d}"
    print(f"\n=== Processing Subject {subject_id} ===")
    
    for task_name, runs in TASK_RUNS.items():
        print(f"\n--- Task: {task_name} ---")
        raw_combined = load_concat_runs(subject_folder, runs)
        if raw_combined is None:
            continue
        
        # Preprocessing
        raw_combined.rename_channels(lambda name: name.replace('.', '').strip().upper())
        raw_combined.set_channel_types({ch: 'eeg' for ch in raw_combined.ch_names})
        montage = mne.channels.make_standard_montage('standard_1020')
        raw_combined.set_montage(montage, match_case=False, match_alias=True, on_missing='warn')
        
        raw_filtered = raw_combined.copy().filter(l_freq=1., h_freq=40.)
        raw_filtered.notch_filter(freqs=[50])
        
        # ICA (inspection only)
        ica = ICA(n_components=min(32, len(raw_filtered.ch_names)), random_state=97, max_iter=800)
        ica.fit(raw_filtered)
        
        sources = ica.get_sources(raw_filtered).get_data()
        sfreq = raw_filtered.info['sfreq']
        times = np.arange(sources.shape[1]) / sfreq
        psds, freqs = psd_array_welch(sources, sfreq=sfreq, fmin=1, fmax=40, n_fft=2048)
        
        out_folder = os.path.join(BASE_OUTPUT_PATH, subject_folder)
        os.makedirs(out_folder, exist_ok=True)
        
        # --------------------
        # ICA Topographies
        # --------------------
        topo_pdf = os.path.join(out_folder, f"{task_name}_ICA_topos.pdf")
        with PdfPages(topo_pdf) as pdf:
            n_per_page = 2
            n_pages = int(np.ceil(ica.n_components_ / n_per_page))
            for page in range(n_pages):
                fig, axes = plt.subplots(1, n_per_page, figsize=(12, 5))
                axes = np.array(axes).flatten()
                start_ic = page * n_per_page
                end_ic = min(start_ic + n_per_page, ica.n_components_)
                for i, ic in enumerate(range(start_ic, end_ic)):
                    ica.plot_components(picks=ic, show=False, axes=axes[i])
                for j in range(end_ic - start_ic, len(axes)):
                    axes[j].axis('off')
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
        
        # --------------------
        # IC Time Series (limited seconds)
        # --------------------
        ts_pdf = os.path.join(out_folder, f"{task_name}_IC_timeseries.pdf")
        with PdfPages(ts_pdf) as pdf:
            n_per_page = 2
            n_pages = int(np.ceil(ica.n_components_ / n_per_page))
            for page in range(n_pages):
                fig, axes = plt.subplots(n_per_page, 1, figsize=(20, 6))
                axes = np.array(axes).flatten()
                start_ic = page * n_per_page
                end_ic = min(start_ic + n_per_page, ica.n_components_)
                mask = times <= TIME_WINDOW
                for i, ic in enumerate(range(start_ic, end_ic)):
                    axes[i].plot(times[mask], sources[ic, mask])
                    axes[i].set_title(f"IC {ic}")
                    axes[i].set_xlabel("Time (s)")
                    axes[i].set_ylabel("Amplitude")
                for j in range(end_ic - start_ic, len(axes)):
                    axes[j].axis('off')
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
        
        # --------------------
        # IC PSD
        # --------------------
        psd_pdf = os.path.join(out_folder, f"{task_name}_IC_PSD.pdf")
        with PdfPages(psd_pdf) as pdf:
            n_per_page = 2
            n_pages = int(np.ceil(ica.n_components_ / n_per_page))
            for page in range(n_pages):
                fig, axes = plt.subplots(n_per_page, 1, figsize=(20, 6))
                axes = np.array(axes).flatten()
                start_ic = page * n_per_page
                end_ic = min(start_ic + n_per_page, ica.n_components_)
                for i, ic in enumerate(range(start_ic, end_ic)):
                    axes[i].semilogy(freqs, psds[ic])
                    axes[i].set_title(f"IC {ic}")
                    axes[i].set_xlabel("Freq (Hz)")
                    axes[i].set_ylabel("PSD")
                for j in range(end_ic - start_ic, len(axes)):
                    axes[j].axis('off')
                pdf.savefig(fig, dpi=300)
                plt.close(fig)
        
        print(f"✅ PDFs saved for task {task_name}")

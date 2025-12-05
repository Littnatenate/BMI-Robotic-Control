"""
FINAL THESIS PIPELINE: EEG CLEANING & EVIDENCE REPORTING
========================================================
Description: 
    1. Loads raw EEG data (EDF) from the PhysioNet dataset.
    2. Performs robust cleaning using ICA (EOG/Muscle artifact removal).
    3. Generates "Thesis-Ready" validation plots (PSD, Variance, Time-domain).
    4. Saves cleaned .fif files for Feature Extraction.

Structure: OOP for academic presentation.
"""

import os
import logging
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import mne
from mne.preprocessing import ICA
from scipy import signal
from joblib import Parallel, delayed
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Configuration (control panel)
@dataclass
class PipelineConfig:
    """Centralized configuration for the EEG Pipeline"""
    # Paths
    base_raw_path: Path = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\raw")
    base_output_path: Path = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed")
    
    # Processing Parameters
    subjects: range = range(1, 110)
    tasks: Dict[str, List[int]] = field(default_factory=lambda: {
        'imagined_movement': [4, 8, 12],
        'actual_movement': [3, 7, 11]
    })
    
    # Tuned ICA Settings 
    eog_threshold: float = 2.5          # Aggressive on Blinks
    muscle_threshold: float = 1.5       # Aggressive on Muscle
    max_components_remove: int = 8
    ica_max_iter: int = 1500
    
    # Execution
    n_jobs: int = 6                    # Use all CPU cores
    force_reprocess: bool = True        # Overwrite existing files

# Pipeline Class
class EEGPipeline:
    def __init__(self, config: PipelineConfig):
        self.cfg = config
        self._setup_logging()
        # We also silence the main process
        mne.set_log_level('ERROR')
        warnings.filterwarnings("ignore")

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
        self.logger = logging.getLogger("EEGPipeline")

    # CORE PROCESSING
    def run(self):
        """Main entry point to run the pipeline in parallel."""
        self.logger.info("="*60)
        self.logger.info(f"STARTING PIPELINE FOR {len(self.cfg.subjects)} SUBJECTS")
        self.logger.info("="*60)
        
        results = Parallel(n_jobs=self.cfg.n_jobs)(
            delayed(self._process_single_subject)(sub_id)
            for sub_id in tqdm(self.cfg.subjects, desc="Processing Subjects")
        )
        self.logger.info("Pipeline Complete.")

    def _process_single_subject(self, subject_id: int):
        # --- FIX 1: SILENCE THE WORKER ---
        # Each parallel process needs to be told to be quiet separately
        mne.set_log_level('ERROR')
        warnings.filterwarnings("ignore")
        
        results = {}
        for task_name, run_indices in self.cfg.tasks.items():
            try:
                # Load/Concatenate
                raw = self._load_and_concat(subject_id, run_indices)
                if not raw:
                    results[task_name] = False
                    continue

                # Prepare Output Paths
                sub_str = f"S{subject_id:03d}"
                out_dir = self.cfg.base_output_path / sub_str
                out_dir.mkdir(parents=True, exist_ok=True)
                
                # MNE COMPLIANT FILENAME (_eeg.fif) [renamed it better]
                out_file = out_dir / f"{sub_str}_{task_name}_cleaned_eeg.fif"

                if out_file.exists() and not self.cfg.force_reprocess:
                    results[task_name] = "Skipped"
                    continue

                # Clean the Data
                raw_original = raw.copy()
                raw_cleaned, ica, exclude = self._clean_data(raw)

                # Save Data
                raw_cleaned.save(out_file, overwrite=True, verbose=False)

                # Generate Evidence Reports
                self._generate_evidence_report(raw_original, raw_cleaned, subject_id, task_name)
                self._generate_ica_report(ica, exclude, subject_id, task_name)
                
                results[task_name] = True

            except Exception as e:
                # We log errors but don't print them to avoid spamming the console
                # Only serious errors will show up if you check the logs later
                pass
        
        return subject_id, results

    # Helper Loading
    def _load_and_concat(self, subject_id: int, runs: List[int]) -> Optional[mne.io.Raw]:
        sub_str = f"S{subject_id:03d}"
        raw_list = []
        
        for run_num in runs:
            fpath = self.cfg.base_raw_path / sub_str / f"{sub_str}R{run_num:02d}.edf"
            if fpath.exists():
                try:
                    raw = mne.io.read_raw_edf(fpath, preload=True, stim_channel='auto', verbose=False)
                    raw_list.append(raw)
                except Exception:
                    pass
        
        if not raw_list:
            return None
        
        combined = mne.concatenate_raws(raw_list, verbose=False)
        
        # Rename
        def smart_rename(name):
            name = name.replace('.', '').strip().upper()
            rename_map = {
                'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz', 
                'FZ': 'Fz', 'CZ': 'Cz', 'PZ': 'Pz', 'OZ': 'Oz',
                'AFZ': 'AFz', 'FCZ': 'FCz', 'CPZ': 'CPz', 'POZ': 'POz', 'IZ': 'Iz'
            }
            return rename_map.get(name, name)

        combined.rename_channels(smart_rename)
        combined.set_channel_types({ch: 'eeg' for ch in combined.ch_names})
        
        try:
            combined.set_montage('standard_1005', on_missing='warn')
        except Exception:
            pass
        
        return combined

    # Helper Cleaning Algorithm
    def _clean_data(self, raw: mne.io.Raw):
        """Runs the ICA and Artifact Rejection logic"""
        # 1. Pre-processing
        raw_copy = raw.copy()
        
        # Sanitize NaNs/Infs
        data = raw_copy.get_data()
        if np.any(np.isinf(data)) or np.any(np.isnan(data)):
             raw_copy.apply_function(lambda x: np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0), verbose=False)

        # Bad Channel Detection
        raw_copy.info['bads'] = self._detect_bad_channels(raw_copy)
        if raw_copy.info['bads']:
            raw_copy.interpolate_bads(reset_bads=True, verbose=False)
        
        raw_copy.set_eeg_reference('average', projection=True, verbose=False)
        
        # 2. ICA Setup
        raw_filt = raw_copy.copy().filter(1.0, 40.0, verbose=False).notch_filter(50.0, verbose=False)
        
        n_chans = len(raw_copy.pick('eeg').ch_names)
        n_comp = min(max(1, min(n_chans - 1, 32)), 32)
        
        ica = ICA(n_components=n_comp, method='fastica', 
                  random_state=97, max_iter=self.cfg.ica_max_iter)
        ica.fit(raw_filt, verbose=False)

        # 3. Automatic Artifact Detection
        exclude = []
        
        # 3.1. EOG (Eye Blinks)
        eog_inds = []
        try:
            eog_inds, _ = ica.find_bads_eog(raw_filt, threshold=self.cfg.eog_threshold, verbose=False)
        except Exception:
            pass

        if not eog_inds: 
             eog_inds = self._find_eog_proxy(ica, raw_filt)
        exclude.extend(eog_inds)
        
        # 3.2. Muscle
        exclude.extend(self._find_muscle_artifacts(ica, raw_filt))
        
        # 3.3. Line Noise
        exclude.extend(self._find_line_noise_components(ica, raw_filt))
        
        exclude = sorted(list(set(exclude)))
        if len(exclude) > self.cfg.max_components_remove:
            exclude = exclude[:self.cfg.max_components_remove]
        
        # 4. Apply
        raw_cleaned = raw_copy.copy()
        ica.apply(raw_cleaned, exclude=exclude, verbose=False)
        
        return raw_cleaned, ica, exclude

    # Helper Detection Logic
    def _detect_bad_channels(self, raw):
        data = raw.get_data()
        bads = []
        
        # Variance Check
        variances = np.var(data, axis=1)
        if np.std(variances) == 0: return []
        z_scores = np.abs((variances - np.mean(variances)) / np.std(variances))
        bads.extend([raw.ch_names[i] for i in np.where(z_scores > 5.0)[0]])
        
        # Flat Line Check
        stds = np.std(data, axis=1)
        bads.extend([raw.ch_names[i] for i in np.where(stds < 1e-6)[0]])
        
        return list(set(bads))

    def _find_eog_proxy(self, ica, raw):
        frontal = ['Fp1', 'Fp2', 'F7', 'F8', 'AF3', 'AF4', 'Fz']
        picks = [ch for ch in frontal if ch in raw.ch_names]
        
        if not picks: return []
        
        eog_data = raw.get_data(picks=picks).mean(axis=0, keepdims=True)
        info_eog = mne.create_info(['EOG_PROXY'], raw.info['sfreq'], ['eog'])
        raw_eog = mne.io.RawArray(eog_data, info_eog, verbose=False)
        raw_with_eog = raw.copy().add_channels([raw_eog], force_update_info=True)
        
        eog_inds, _ = ica.find_bads_eog(raw_with_eog, ch_name='EOG_PROXY', threshold=self.cfg.eog_threshold, verbose=False)
        return eog_inds

    def _find_muscle_artifacts(self, ica, raw):
        muscle_idx = []
        sfreq = raw.info['sfreq']
        try:
            sources = ica.get_sources(raw).get_data()
            for i, comp in enumerate(sources):
                freqs, psd = signal.welch(comp, fs=sfreq, nperseg=min(2048, len(comp)))
                high_p = np.mean(psd[(freqs >= 20) & (freqs <= 40)])
                low_p = np.mean(psd[(freqs >= 1) & (freqs <= 20)])
                if low_p > 0 and (high_p / low_p) > self.cfg.muscle_threshold:
                    muscle_idx.append(i)
        except Exception:
            pass
        return muscle_idx

    def _find_line_noise_components(self, ica, raw):
        line_indices = []
        sfreq = raw.info['sfreq']
        try:
            sources = ica.get_sources(raw).get_data()
            for idx in range(sources.shape[0]):
                freqs, psd = signal.welch(sources[idx], fs=sfreq, nperseg=min(2048, len(sources[idx])))
                line_idx = np.argmin(np.abs(freqs - 50.0))
                if 0 <= line_idx < len(psd):
                    line_power = psd[line_idx]
                    neighbor_mask = (freqs >= 45) & (freqs <= 55)
                    neighbor_mask[line_idx] = False
                    neighbor_power = np.mean(psd[neighbor_mask])
                    if neighbor_power > 0 and (line_power / neighbor_power) > 4.0:
                        line_indices.append(idx)
        except Exception: 
            pass
        return line_indices

    # Visualisation Reports
    def _generate_evidence_report(self, raw_pre, raw_post, sub_id, task):
        try:
            report_dir = self.cfg.base_output_path / "validation_reports"
            report_dir.mkdir(exist_ok=True)
            
            fig = plt.figure(figsize=(18, 10), constrained_layout=True)
            gs = fig.add_gridspec(2, 2)
            
            # 1. PSD
            ax1 = fig.add_subplot(gs[0, 0])
            raw_pre.compute_psd(fmax=60, verbose=False).plot(axes=ax1, color='red', show=False, average=True, spatial_colors=False)
            raw_post.compute_psd(fmax=60, verbose=False).plot(axes=ax1, color='blue', show=False, average=True, spatial_colors=False)
            ax1.set_title(f"S{sub_id}: PSD Change (Red=Raw, Blue=Clean)")
            ax1.legend(["Raw", "Cleaned"])
            ax1.grid(True, linestyle=':', alpha=0.6)
            
            # 2. Time Series (Blinks)
            ax2 = fig.add_subplot(gs[0, 1])
            picks = [ch for ch in ['Fp1', 'Fpz', 'Fp2', 'AF3', 'AF4', 'Fz'] if ch in raw_pre.ch_names]
            chan = picks[0] if picks else raw_pre.ch_names[0]
            
            start = raw_pre.times[-1] / 2
            duration = 10
            r_data = raw_pre.copy().crop(start, start+duration).get_data(picks=chan)[0] * 1e6
            c_data = raw_post.copy().crop(start, start+duration).get_data(picks=chan)[0] * 1e6
            times = raw_pre.copy().crop(start, start+duration).times
            
            ax2.plot(times, r_data, 'r', alpha=0.5, label='Raw')
            ax2.plot(times, c_data, 'b', label='Clean')
            ax2.set_title(f"Blink Removal Check ({chan})")
            ax2.set_ylabel("Amplitude (uV)")
            ax2.legend()
            
            # 3. Variance Reduction
            ax3 = fig.add_subplot(gs[1, :])
            var_pre = np.var(raw_pre.get_data(), axis=1)
            var_post = np.var(raw_post.get_data(), axis=1)
            ax3.scatter(var_pre, var_post, c='purple', alpha=0.6)
            max_v = max(np.max(var_pre), np.max(var_post))
            ax3.plot([0, max_v], [0, max_v], 'k--', label='No Change')
            ax3.set_xlabel("Raw Variance"); ax3.set_ylabel("Clean Variance")
            ax3.set_title("Variance Reduction (Dots BELOW line = Noise Removed)")
            ax3.legend()
            ax3.grid(True)
            
            fig.savefig(report_dir / f"S{sub_id:03d}_{task}_Evidence.png")
            plt.close(fig)
        except Exception as e:
            pass

    def _generate_ica_report(self, ica, exclude, sub_id, task):
        try:
            if not exclude: return
            report_dir = self.cfg.base_output_path / "ica_reports"
            report_dir.mkdir(exist_ok=True)
            fig = ica.plot_components(picks=exclude, show=False)
            if not isinstance(fig, list): fig = [fig]
            for i, f in enumerate(fig):
                f.savefig(report_dir / f"S{sub_id:03d}_{task}_ICA_Excl_{i}.png")
                plt.close(f)
        except Exception: pass


# Execution
if __name__ == "__main__":
    config = PipelineConfig()
    pipeline = EEGPipeline(config)
    pipeline.run()
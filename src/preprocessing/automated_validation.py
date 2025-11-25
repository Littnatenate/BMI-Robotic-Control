"""
VALIDATION PIPELINE: POST-CLEANING QUALITY ASSURANCE (V4 - RENAME FIX)
======================================================================
Updates:
- Added '_sanitize_names' to fix Raw channel names (removes dots/fixes caps).
- Solves the "Channel not found" / "np.str_" plotting crash.
"""

import os
import logging
import warnings
import numpy as np
import matplotlib
matplotlib.use('Agg') # Non-interactive backend
import matplotlib.pyplot as plt
import mne
from scipy import signal, stats
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional

# Configuration
@dataclass
class ValidationConfig:
    """Centralized configuration for Validation."""
    base_output_path: Path = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed")
    subjects: range = range(20, 32)
    task: str = 'imagined_movement'
    
    motor_channels: List[str] = field(default_factory=lambda: [
        'FC5', 'FC3', 'FC1', 'FCz', 'FC2', 'FC4', 'FC6',
        'C5', 'C3', 'C1', 'Cz', 'C2', 'C4', 'C6',
        'CP5', 'CP3', 'CP1', 'CPz', 'CP2', 'CP4', 'CP6'
    ])

    freq_bands: Dict[str, tuple] = field(default_factory=lambda: {
        'delta': (1, 4), 'theta': (4, 8), 'mu': (8, 12), 'beta': (13, 30), 'gamma': (30, 40)
    })

# Validation pipeline
class ValidationPipeline:
    def __init__(self, config: ValidationConfig):
        self.cfg = config
        self._setup_logging()
        self.report_dir = self.cfg.base_output_path / 'validation_metrics'
        self.report_dir.mkdir(parents=True, exist_ok=True)
        mne.set_log_level('ERROR')
        warnings.filterwarnings('ignore')

    def _setup_logging(self):
        logging.basicConfig(level=logging.INFO, format='%(message)s')
        self.logger = logging.getLogger("Validator")

    def run(self):
        print("="*80)
        print(f"STARTING VALIDATION FOR TASK: {self.cfg.task}")
        print("="*80)

        results = []
        for sub_id in self.cfg.subjects:
            res = self._validate_subject(sub_id)
            if res: results.append(res)

        self._print_final_summary(results)

    def _validate_subject(self, subject_id: int):
        sub_str = f"S{subject_id:03d}"
        folder = self.cfg.base_output_path / sub_str
        raw_path = folder / f"{sub_str}_{self.cfg.task}_raw.fif"
        clean_path = folder / f"{sub_str}_{self.cfg.task}_cleaned_eeg.fif"

        if not raw_path.exists() or not clean_path.exists():
            return None

        # Corruption Check
        try:
            mne.io.read_info(clean_path, verbose=False)
        except Exception:
            print(f"\nSkipping {sub_str}: File corrupted (re-run cleaning).")
            return None

        print(f"\nProcessing {sub_str}...", end=" ")
        
        try:
            # 1. Load Data
            raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)
            clean = mne.io.read_raw_fif(clean_path, preload=True, verbose=False)

            # This aligns Raw names (e.g. "FC5.") with Clean names ("FC5")
            self._sanitize_names(raw)

            if len(raw.times) < 100:
                print("Skipped (Empty Data)")
                return None

            metrics = {'subject': subject_id}

            # 2. Metrics
            try:
                metrics.update(self._calc_snr_metrics(raw, clean))
                print(f"SNR: {metrics.get('snr_gain', 0):+.1f}dB", end=" | ")
            except Exception: print("SNR err", end=" | ")

            try: metrics.update(self._detect_residuals(raw, clean))
            except Exception: pass

            try: metrics.update(self._calc_band_power_changes(raw, clean))
            except Exception: pass

            try: metrics.update(self._check_motor_erd(clean))
            except Exception: pass

            try: metrics.update(self._run_statistical_check(raw, clean))
            except Exception: pass

            # 3. Plotting
            try:
                self._plot_comparison(raw, clean, subject_id)
                print("Plot Saved.")
            except Exception as e: 
                print(f"Plot Fail: {str(e)}")
            
            return metrics

        except Exception as e:
            print(f"Critical Fail: {e}")
            return None

    # --- HELPERS ---
    def _sanitize_names(self, inst):
        """Fixes capitalization and removes dots from channel names."""
        def smart_rename(name):
            name = str(name).replace('.', '').strip().upper()
            mapping = {
                'FP1': 'Fp1', 'FP2': 'Fp2', 'FPZ': 'Fpz', 'FZ': 'Fz', 
                'CZ': 'Cz', 'PZ': 'Pz', 'FCZ': 'FCz', 'CPZ': 'CPz',
                'AFZ': 'AFz', 'POZ': 'POz', 'IZ': 'Iz'
            }
            return mapping.get(name, name)
        
        inst.rename_channels(smart_rename)
        # Force re-mapping of channel types just in case
        inst.set_channel_types({ch: 'eeg' for ch in inst.ch_names}, on_unit_change='ignore')

    def _get_motor_picks(self, inst):
        # Return strict list of strings that ACTUALLY exist in the instance
        available = set(inst.ch_names)
        found = []
        for target in self.cfg.motor_channels:
            # We assume instance names are already sanitized
            if target in available:
                found.append(target)
        return found

    # Metric Calculations
    def _calc_snr_metrics(self, raw, clean):
        def get_snr(inst):
            # Safe pick: use all channels
            data = inst.get_data()
            sig_power = np.mean(np.var(data, axis=1))
            if inst.info['sfreq'] > 80:
                high_freq = inst.copy().filter(40.0, None, verbose=False).get_data()
                noise_power = np.mean(np.var(high_freq, axis=1))
                if noise_power == 0: return 0, sig_power
                return 10 * np.log10(sig_power / noise_power), sig_power
            return 0, sig_power

        snr_pre, p_pre = get_snr(raw)
        snr_post, p_post = get_snr(clean)
        return {'power_reduction': (1 - p_post/p_pre) * 100 if p_pre > 0 else 0, 'snr_gain': snr_post - snr_pre}

    def _detect_residuals(self, raw, clean):
        def get_artifacts(inst):
            data = inst.get_data()
            freqs, psd = signal.welch(data, fs=inst.info['sfreq'], nperseg=min(2048, data.shape[1]))
            line_p = np.mean(psd[:, np.argmin(np.abs(freqs - 50))])
            high = np.mean(psd[:, (freqs >= 30) & (freqs <= 40)])
            low = np.mean(psd[:, (freqs >= 8) & (freqs <= 12)])
            return line_p, (high / low if low > 0 else 0)

        l_pre, m_pre = get_artifacts(raw)
        l_post, m_post = get_artifacts(clean)
        return {'line_reduction_pct': (l_pre - l_post)/l_pre * 100 if l_pre > 0 else 0, 'muscle_reduction_pct': (m_pre - m_post)/m_pre * 100 if m_pre > 0 else 0}

    def _calc_band_power_changes(self, raw, clean):
        results = {}
        for band, (low, high) in self.cfg.freq_bands.items():
            p_pre = self._get_band_power(raw, low, high)
            p_post = self._get_band_power(clean, low, high)
            if p_pre > 0: results[f'{band}_change'] = ((p_post - p_pre) / p_pre) * 100
        return results

    def _check_motor_erd(self, clean):
        try:
            events, _ = mne.events_from_annotations(clean, verbose=False)
            if len(events) < 5: return {}
            event_ids = {'left': 2, 'right': 3} if self.cfg.task == 'imagined_movement' else {'left': 5, 'right': 6}
            valid_events = events[np.isin(events[:, 2], list(event_ids.values()))]
            if len(valid_events) < 5: return {}

            found_ids = {k: v for k, v in event_ids.items() if v in valid_events[:, 2]}
            # Explicitly picks motor channels
            picks = self._get_motor_picks(clean)
            if not picks: return {}
            
            epochs = mne.Epochs(clean, valid_events, event_id=found_ids, tmin=-1, tmax=4, picks=picks, baseline=None, verbose=False)
            
            base = epochs.copy().crop(-1, 0).get_data()
            task = epochs.copy().crop(0.5, 2.5).get_data()
            
            def get_mu_power(data, sfreq):
                freqs, psd = signal.welch(data, fs=sfreq, axis=-1, nperseg=min(data.shape[-1], 256))
                mask = (freqs >= 8) & (freqs <= 12)
                return np.mean(psd[..., mask])

            mu_base = get_mu_power(base, clean.info['sfreq'])
            mu_task = get_mu_power(task, clean.info['sfreq'])
            
            if mu_base == 0: return {}
            return {'mu_erd': (mu_base - mu_task) / mu_base}
        except: return {}

    def _run_statistical_check(self, raw, clean):
        try:
            _, p_val = stats.wilcoxon(np.var(raw.get_data(), axis=1), np.var(clean.get_data(), axis=1))
            return {'wilcoxon_p': p_val}
        except: return {'wilcoxon_p': 1.0}

    def _get_band_power(self, inst, low, high):
        picks = self._get_motor_picks(inst)
        if not picks: return 0.0
        data = inst.get_data(picks=picks)
        freqs, psd = signal.welch(data, fs=inst.info['sfreq'], nperseg=min(2048, data.shape[1]))
        mask = (freqs >= low) & (freqs <= high)
        return np.mean(psd[:, mask])

    # Graph plotting
    def _plot_comparison(self, raw, clean, sub_id):
        fig = plt.figure(figsize=(16, 12))
        
        # 1. PSD
        ax1 = plt.subplot(2, 2, 1)
        raw.compute_psd(fmax=60, verbose=False).plot(axes=ax1, color='r', show=False, average=True, spatial_colors=False)
        clean.compute_psd(fmax=60, verbose=False).plot(axes=ax1, color='b', show=False, average=True, spatial_colors=False)
        ax1.set_title(f"S{sub_id}: Global PSD")
        ax1.legend(['Raw', 'Clean'])

        # 2. Band Changes
        ax2 = plt.subplot(2, 2, 2)
        bands = list(self.cfg.freq_bands.keys())
        vals = []
        for b in bands:
            l, h = self.cfg.freq_bands[b]
            pre = self._get_band_power(raw, l, h)
            post = self._get_band_power(clean, l, h)
            vals.append(((post-pre)/pre * 100) if pre > 0 else 0)
        ax2.bar(bands, vals, color=['g' if v>0 else 'r' for v in vals])
        ax2.set_title("Band Power Change (%)")
        ax2.axhline(0, color='k')

        # 3. Variance
        ax3 = plt.subplot(2, 2, 3)
        v_pre = np.var(raw.get_data(), axis=1)
        v_post = np.var(clean.get_data(), axis=1)
        ax3.scatter(v_pre, v_post, alpha=0.5)
        m = max(v_pre.max(), v_post.max())
        ax3.plot([0, m], [0, m], 'k--')
        ax3.set_xlabel("Raw Var"); ax3.set_ylabel("Clean Var")
        ax3.set_xscale('log'); ax3.set_yscale('log')

        # 4. Motor Time Series
        ax4 = plt.subplot(2, 2, 4)
        picks = self._get_motor_picks(clean)[:3] 
        if picks:
            start = clean.times[-1]/2
            d_raw = raw.copy().crop(start, start+4).get_data(picks=picks)
            d_cln = clean.copy().crop(start, start+4).get_data(picks=picks)
            t = np.arange(d_raw.shape[1])/raw.info['sfreq']
            for i in range(len(picks)):
                ax4.plot(t, d_raw[i] + i*1e-4, 'r', alpha=0.3)
                ax4.plot(t, d_cln[i] + i*1e-4, 'b')
            ax4.set_title("Motor Channels (Time)")
        
        plt.tight_layout()
        plt.savefig(self.report_dir / f"S{sub_id:03d}_Validation_Full.png")
        plt.close(fig)

    def _print_final_summary(self, results):
        if not results:
            print("No results collected.")
            return
        print("\n" + "="*80)
        print("FINAL VERDICT")
        print("="*80)
        avg_red = np.mean([r['power_reduction'] for r in results])
        avg_mu_erd = np.mean([r.get('mu_erd', 0) for r in results])
        print(f"Validated Subjects: {len(results)}")
        print(f"Avg Signal Reduction: {avg_red:.1f}%")
        print(f"Avg Mu-ERD: {avg_mu_erd:.2%}")
        if avg_red > 90: print("High Power reduction (>90%)")
        if avg_mu_erd < 0.01: print("Low Mu-ERD")
        print("DONE." if avg_red <= 90 else "REVIEW REQUIRED.")

if __name__ == "__main__":
    ValidationPipeline(ValidationConfig()).run()
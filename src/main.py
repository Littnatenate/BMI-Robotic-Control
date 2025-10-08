import mne

# Example: load a single EEG file (EDF format)
raw = mne.io.read_raw_edf(r'../physionet.org/S001/S001R01.edf', preload=True)
raw.plot()

# =================================================================
# ✅ DEBUGGING SCRIPT: Full pipeline on a SINGLE file
# =================================================================

import mne
from mne.preprocessing import ICA
import os

# This will show plots inside the notebook, which is more stable

print("--- 1. Loading ONE file for debugging ---")
# We load only ONE run to guarantee it fits in memory
file_paths = mne.datasets.eegbci.load_data(subjects=1, runs=[4])
raw = mne.io.read_raw_edf(file_paths[0], preload=True, stim_channel='auto')
print("✅ Data loaded.")

# --- 2. Preprocessing ---
print("\n--- 2. Filtering and setting montage... ---")
raw_processed = raw.copy()
mapping = {ch_name: ch_name.strip('.') for ch_name in raw_processed.ch_names}
raw_processed.rename_channels(mapping)
raw_processed.set_montage('standard_1005', match_case=False)
raw_processed.filter(l_freq=1., h_freq=40.)
print("✅ Preprocessing complete.")

# --- 3. Automated ICA ---
print("\n--- 3. Running automated ICA... ---")
ica = ICA(n_components=20, random_state=97, max_iter=800)
ica.fit(raw_processed)
eog_indices, eog_scores = ica.find_bads_eog(raw_processed, ch_name=['Fp1', 'Fp2'])
ica.exclude = eog_indices
raw_cleaned = raw_processed.copy()
ica.apply(raw_cleaned)
print("✅ ICA cleaning complete.")

# --- 4. Epoching ---
print("\n--- 4. Creating epochs... ---")
events, event_id_from_annot = mne.events_from_annotations(raw_cleaned)
event_id = {'left_fist': event_id_from_annot['T1'], 'right_fist': event_id_from_annot['T2']}

epochs = mne.Epochs(
    raw_cleaned,
    events,
    event_id=event_id,
    tmin=-0.5,
    tmax=4.0,
    preload=True,
    baseline=(-0.5, 0),
    reject=dict(eeg=200e-6) # Using a relaxed threshold
)
print("✅ Epoching complete. Final trials:")
print(epochs)

# --- 5. Final Visualization ---
print("\n--- 5. Plotting drop log... ---")
epochs.plot_drop_log()
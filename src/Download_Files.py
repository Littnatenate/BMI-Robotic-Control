import mne

# --- Define the subject and runs you want ---
subject_id = 1
# These are the motor imagery runs for left vs. right fist
runs_to_load = [4, 8, 12] 

# --- Use MNE's downloader ---
# This will download the files to the correct MNE-managed directory
# and return a list of the file paths.
file_paths = mne.datasets.eegbci.load_data(
    subject=subject_id, 
    runs=runs_to_load
)

print(f"✅ MNE has located/downloaded the following files for Subject {subject_id}:")
print(file_paths)

# --- Load and combine the data from the paths MNE provided ---
raw_files = [mne.io.read_raw_edf(path, preload=True) for path in file_paths]
raw_combined = mne.concatenate_raws(raw_files)

print("\n--- Combined Data Info ---")
print(raw_combined.info)
import os
import mne

# Directory
RAW_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\raw"
os.makedirs(RAW_PATH, exist_ok=True)

# downloading all subjects and runs
subjects = range(49, 64)
runs = range(1, 15)

# downloading section
for subject_id in subjects:
    print(f"\n📥 Downloading data for Subject {subject_id:03d}...")

    try:
        # Download all runs for each subject
        file_paths = mne.datasets.eegbci.load_data(
            subjects=subject_id,
            runs=runs,
            path=RAW_PATH
        )

        print(f"✅ Successfully downloaded all runs for Subject {subject_id:03d}")
        for fp in file_paths:
            print("  -", fp)

    except Exception as e:
        print(f"❌ Failed to download Subject {subject_id:03d}: {e}")

print("\n🎯 All subjects processed.")
print(f"📂 Data saved in: {RAW_PATH}")

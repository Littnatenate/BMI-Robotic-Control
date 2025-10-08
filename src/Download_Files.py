import os
import requests

base_url = "https://physionet.org/files/eegmmidb/1.0.0/"
# Keep the original range, so it checks everything
subjects = [f"S{str(i).zfill(3)}" for i in range(1, 110)] 

root_folder = "physionet.org"
os.makedirs(root_folder, exist_ok=True)

for subj in subjects:
    runs = [f"{subj}R{str(r).zfill(2)}.edf" for r in range(1, 15)]
    subj_folder = os.path.join(root_folder, subj)
    os.makedirs(subj_folder, exist_ok=True)
    for run in runs:
        url = f"{base_url}{subj}/{run}"
        local_path = os.path.join(subj_folder, run)

        # Check if the file already exists before downloading
        if not os.path.exists(local_path):
            try:
                print(f"Downloading {run} ...")
                r = requests.get(url)
                r.raise_for_status()
                with open(local_path, "wb") as f:
                    f.write(r.content)
            except Exception as e:
                print(f"Failed {run}: {e}")
        else:
            # If it exists, just print a skip message
            print(f"Skipping {run}, already exists.")
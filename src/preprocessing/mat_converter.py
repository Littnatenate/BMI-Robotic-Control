import os
import numpy as np
import scipy.io as sio
import glob

print("--- Intelligent MAT to NPZ Conversion Script ---")

def convert_files_based_on_structure():
    """
    Automatically detects the structure of the .mat files and converts them
    using the appropriate method (generic or specialized).
    """
    # --- Step 1: Set up paths using the robust method ---
    try:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(script_dir))
    except NameError:
        project_root = os.getcwd()

    input_folder = os.path.join(project_root, 'Datasets', 'mat_raw')
    output_folder = os.path.join(project_root, 'Datasets', 'mat_processed')

    os.makedirs(output_folder, exist_ok=True)

    # --- Step 2: Find all .mat files ---
    mat_files = glob.glob(os.path.join(input_folder, '*.mat'))

    if not mat_files:
        print(f"No .mat files were found in '{input_folder}'")
        return

    print(f"Found {len(mat_files)} files. Inspecting structure...")

    # --- Step 3: Inspect the first file to decide the conversion mode ---
    sample_file_path = mat_files[0]
    mat_contents = sio.loadmat(sample_file_path)
    
    conversion_mode = 'generic' # Default mode
    if 'EEG' in mat_contents:
        conversion_mode = 'specialized' # Found the key for the specialized structure
    
    print(f"Auto-detected file structure as: '{conversion_mode.upper()}'")
    print("Starting batch conversion...")

    # --- Step 4: Loop through all files and convert using the detected mode ---
    for mat_path in mat_files:
        try:
            base_filename = os.path.basename(mat_path)
            npz_filename = os.path.splitext(base_filename)[0] + '.npz'
            npz_path = os.path.join(output_folder, npz_filename)

            # Load the current file
            current_mat_contents = sio.loadmat(mat_path)
            
            # Use the correct method to get the data
            if conversion_mode == 'specialized':
                EEG = current_mat_contents['EEG']
                data = EEG['data'][0, 0]
                labels_onehot = EEG['y'][0, 0]
                labels = np.argmax(labels_onehot, axis=1)
                data_to_save = {'data': data, 'labels': labels}
            else: # 'generic' mode
                data_to_save = {k: v for k, v in current_mat_contents.items() if not k.startswith('__')}

            # Save the extracted data
            np.savez_compressed(npz_path, **data_to_save)
            print(f"  > Converted '{base_filename}' to '{npz_filename}'")

        except Exception as e:
            print(f"  > FAILED to convert {os.path.basename(mat_path)}: {e}")

    print("\nBatch conversion complete.")


# --- Main execution block ---
if __name__ == "__main__":
    convert_files_based_on_structure()
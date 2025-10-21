import mne
import numpy as np
import sys
import matplotlib
import matplotlib.pyplot as plt # Import pyplot



print(f"--- Minimal Plotting Test ---")
print(f"Using Python executable: {sys.executable}")
print(f"Using MNE version: {mne.__version__}")
print(f"Using Matplotlib version: {matplotlib.__version__}")

# --- 2. Create a Fake 'raw' Object ---
print("Creating a fake raw object with standard_1020 montage...")
try:
    montage = mne.channels.make_standard_montage('standard_1020')
    ch_names = montage.ch_names
    sfreq = 250
    n_channels = len(ch_names)
    
    # 10 seconds of fake random data
    data = np.random.rand(n_channels, 10 * sfreq) 
    
    info = mne.create_info(ch_names=ch_names, sfreq=sfreq, ch_types='eeg')
    raw = mne.io.RawArray(data, info)
    
    # --- 3. Set the Montage ---
    raw.set_montage(montage)
    print("Montage set successfully.")

    # --- 4. Test 1: 2D Plot ---
    print("Attempting 2D plot...")
    fig_2d = raw.plot_sensors(show_names=True)
    plt.show() # Make sure the 2D plot shows up
    print("✅ 2D plot successful.")

    # --- 5. Test 2: 3D Plot ---
    print("Attempting 3D plot (this will open a new window)...")
    
    # This is the command that was crashing
    fig_3d = raw.plot_sensors(kind='3d', show_names=True)
    
    print("\n✅ 3D plot window opened!")
    print("If the window is interactive and not frozen, your environment is fixed!")
    print("You can close the 3D window and the 2D plot window to finish the test.")

except Exception as e:
    print(f"\n--- ❌ TEST FAILED ---")
    print(f"An error occurred: {e}")
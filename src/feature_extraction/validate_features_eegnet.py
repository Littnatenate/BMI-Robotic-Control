"""
EEGNet / Raw EEG Validation
- Expects: *_eegnet_features.pkl containing dict with:
    "X" -> (Trials, Channels, Samples)  e.g. (N, 64, 320)
    "y" -> (Trials,) labels (0/1 or 2/3)
- Produces: amplitude histogram, sample waveform (C3/C4), PSD left vs right, topomap difference (beta)
"""
import os
import pickle
import numpy as np
import matplotlib
try:
    matplotlib.use('TkAgg')
except:
    pass
import matplotlib.pyplot as plt
from scipy import signal
import mne

BASE_OUTPUT_PATH = r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed_eegnet"

# default standard 10-05 (64) names — adjust if your data uses different ordering
DEFAULT_CH_NAMES = [
    'Fp1','Fp2','F7','F3','Fz','F4','F8','FC5','FC3','FC1','FCz','FC2','FC4','FC6',
    'T7','C5','C3','C1','Cz','C2','C4','C6','T8','CP5','CP3','CP1','CPz','CP2','CP4','CP6',
    'P7','P3','Pz','P4','P8','PO7','PO3','O1','Oz','O2','PO4','PO8','AF7','AF3','AFz','AF4','AF8',
    'F5','F1','F2','F6','FT7','FT8','TP7','TP8','P5','P1','P2','P6','POz','Iz','Cz'  # if extra, trimmed later
]
# We'll try to use ch_names from file if provided.

def load_features():
    while True:
        try:
            sub_input = input("\nEnter subject ID to analyze (1-109): ").strip()
            sub_id = int(sub_input)
            break
        except:
            pass

    folder_name = f"S{sub_id:03d}"
    file_name = f"{folder_name}_eegnet_features.pkl"
    file_path = os.path.join(BASE_OUTPUT_PATH, folder_name, file_name)

    if not os.path.exists(file_path):
        print(f"[Error] File not found: {file_path}")
        return None, None, None

    print(f"Loading: {file_path}...")
    with open(file_path, 'rb') as f:
        data = pickle.load(f)

    X = data.get("X")
    y = data.get("y")
    ch_names = data.get("channels", None)
    if ch_names is None:
        # attempt to detect common keys
        ch_names = data.get("chan_names", None) or data.get("channel_names", None)

    print("Loaded shapes:", None if X is None else X.shape, None if y is None else y.shape)
    print("Unique labels before filtering:", np.unique(y))
    
    # FILTER: Keep ONLY Imagined Movement (classes 2, 3)
    if y is not None:
        unique_labels = set(np.unique(y))
        
        # If we have 4 classes (0,1,2,3), keep only 2,3 (Imagined Movement)
        if unique_labels == {0, 1, 2, 3}:
            print("[Info] Found 4 classes. Keeping ONLY classes 2,3 (Imagined Movement)...")
            mask = np.isin(y, [2, 3])
            X = X[mask]
            y = y[mask]
            print(f"  Filtered to {X.shape[0]} trials")
            # Remap 2->0, 3->1 for cleaner labels
            print("[Info] Remapping class 2->0 (Left), class 3->1 (Right)...")
            y = y - 2
        
        # If we already have only 2,3, just remap
        elif unique_labels == {2, 3}:
            print("[Info] Remapping class 2->0 (Left), class 3->1 (Right)...")
            y = y - 2
        
        print("Unique labels after filtering:", np.unique(y))
    
    return X, y, ch_names

def safe_label_names(y):
    classes = np.unique(y)
    if set(classes) == {0, 1}:
        return {0: "Left", 1: "Right"}
    if set(classes) == {2, 3}:
        return {2: "Left", 3: "Right"}
    # fallback map
    return {c: f"Class {c}" for c in classes}

def amplitude_histogram(X):
    # Flatten all data for amplitude histogram; report min/max
    vals = X.flatten()
    print(f"Amplitude stats: min={vals.min():.3f}, max={vals.max():.3f}, mean={vals.mean():.3f}, std={vals.std():.3f}")
    plt.figure(figsize=(8,5))
    plt.hist(vals, bins=200)
    plt.title("Amplitude Histogram (All channels & trials)")
    plt.xlabel("Amplitude")
    plt.ylabel("Count")
    plt.show()

def plot_sample_waveforms(X, y, ch_names):
    # choose representative trials per class
    label_map = safe_label_names(y)
    classes = np.unique(y)
    sample_idx = {c: np.where(y==c)[0][0] for c in classes}
    # find C3/C4 indices
    if ch_names:
        try:
            c3 = ch_names.index("C3")
            c4 = ch_names.index("C4")
        except ValueError:
            # fallback heuristics
            c3, c4 = 16, 20 if X.shape[1] > 20 else (0,1)
    else:
        c3, c4 = 16, 20 if X.shape[1] > 20 else (0,1)

    t = np.arange(X.shape[2]) / 160.0  # assuming fs=160
    plt.figure(figsize=(12,6))
    for i, c in enumerate(classes):
        idx = sample_idx[c]
        plt.subplot(len(classes),2, 2*i+1)
        plt.plot(t, X[idx, c3, :])
        plt.title(f"{label_map[c]} - {ch_names[c3] if ch_names else 'C3'}")
        plt.xlabel("Time (s)")
        plt.subplot(len(classes),2, 2*i+2)
        plt.plot(t, X[idx, c4, :])
        plt.title(f"{label_map[c]} - {ch_names[c4] if ch_names else 'C4'}")
        plt.xlabel("Time (s)")
    plt.tight_layout()
    plt.show()

def plot_psd(X, y):
    fs = 160
    classes = np.unique(y)
    label_map = safe_label_names(y)
    plt.figure(figsize=(8,5))
    for c in classes:
        sel = X[y==c]  # (trials, ch, time)
        # compute welch across trials and channels -> average PSD
        f, Pxx = signal.welch(sel, fs=fs, nperseg=160, axis=-1)  # returns (trials, ch, freq)
        # Pxx shape: (trials, channels, freqs)
        avg = np.mean(Pxx, axis=(0,1))  # average trials and channels
        plt.plot(f, 10*np.log10(avg), label=label_map[c])
    plt.xlim(0,40)
    plt.xlabel("Freq (Hz)")
    plt.ylabel("Power (dB)")
    plt.legend()
    plt.title("Global PSD (all channels averaged)")
    plt.show()

def plot_topomap_beta(X, y, ch_names):
    # 1. Setup Info and Montage
    n_ch = X.shape[1]
    # Handle channel names
    if ch_names and len(ch_names) >= n_ch:
        names = ch_names[:n_ch]
    else:
        names = DEFAULT_CH_NAMES[:n_ch]

    # Remove duplicates if any
    seen = {}
    unique_names = []
    keep_indices = []
    for i, name in enumerate(names):
        if name not in seen:
            seen[name] = True
            unique_names.append(name)
            keep_indices.append(i)
    
    # Subset data to unique channels
    X_unique = X[:, keep_indices, :]
    info = mne.create_info(ch_names=unique_names, sfreq=160, ch_types='eeg')
    
    try:
        montage = mne.channels.make_standard_montage('standard_1005')
        info.set_montage(montage, on_missing='ignore')
    except Exception as e:
        print(f"[Warning] Montage issue: {e}")

    # 2. Compute Beta Power (13-30Hz)
    print("Computing Welch PSD...")
    f, Pxx = signal.welch(X_unique, fs=160, nperseg=160, axis=-1)
    beta_idx = np.where((f >= 13) & (f <= 30))[0]
    
    # Average power in beta band per trial/channel
    beta_power = np.mean(Pxx[:, :, beta_idx], axis=2)  # Shape: (trials, channels)

    classes = np.unique(y)
    label_map = safe_label_names(y)

    # 3. Calculate Mean Topomaps per Class
    p_list = []
    labels = []
    for c in classes:
        sel = beta_power[y==c]
        if sel.size == 0:
            p_list.append(np.zeros(len(unique_names)))
        else:
            p_list.append(np.mean(sel, axis=0))
        labels.append(label_map[c])

    # --- SCALING ---
    # Find global min/max across both classes
    all_vals = np.concatenate(p_list)
    vmin, vmax = np.percentile(all_vals, [1, 99]) 

    # 4. Plotting
    fig, axes = plt.subplots(1, len(p_list)+1, figsize=(15, 4))
    
    # Plot Absolute Power (Left and Right)
    for i, arr in enumerate(p_list):
        # FIX: used vlim instead of vmin/vmax, removed 'names'
        im, cn = mne.viz.plot_topomap(arr, info, axes=axes[i], show=False, 
                                      vlim=(vmin, vmax), cmap='Reds')
        axes[i].set_title(f"{labels[i]} (Absolute Power)")
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)

    # Plot Difference (Left - Right)
    if len(p_list) >= 2:
        diff = p_list[0] - p_list[1]
        limit = np.max(np.abs(diff))
        
        # FIX: used vlim instead of vmin/vmax, removed 'names'
        im, cn = mne.viz.plot_topomap(diff, info, axes=axes[-1], show=False, 
                                      vlim=(-limit, limit), cmap='RdBu_r')
        axes[-1].set_title(f"Difference ({labels[0]} - {labels[1]})")
        plt.colorbar(im, ax=axes[-1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    X, y, ch_names = load_features()
    if X is None:
        raise SystemExit(0)

    # shape checks
    if X.ndim != 3:
        print("[Error] Expected raw EEG shape (Trials, Channels, Samples). Found:", X.shape)
        # try to handle if spectrogram-like by back-converting? not supported here
        raise SystemExit(0)

    # quick print
    print("Final unique labels:", np.unique(y))

    while True:
        print("\n--- EEG (EEGNet) VISUALIZATION ---")
        print("1. Amplitude Histogram")
        print("2. Sample Waveforms (C3/C4)")
        print("3. Global PSD (Left vs Right)")
        print("4. Topomap (Beta band)")
        print("Q. Quit")
        choice = input("Selection: ").lower().strip()
        if choice == '1':
            amplitude_histogram(X)
        elif choice == '2':
            plot_sample_waveforms(X, y, ch_names)
        elif choice == '3':
            plot_psd(X, y)
        elif choice == '4':
            plot_topomap_beta(X, y, ch_names)
        elif choice == 'q':
            break
"""
Description: Generates feature space maps for EEGNet, ATCNet, or SpectrogramCNN.
Usage: Change 'MODEL_TYPE' at the top to switch models.
"""

import sys
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm
import pandas as pd

# Setup Path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.config import PROJECT_ROOT, SUBJECTS
from src.train import BCIDataset, get_model, EEGNET_CONFIG, ATCNET_CONFIG, CNN_CONFIG

# CONFIGURATION
# Options: 'eegnet', 'atcnet', 'cnn'
MODEL_TYPE = 'atcnet'

# Select Config based on type
if MODEL_TYPE == 'eegnet':
    CFG = EEGNET_CONFIG
    MODE = 'time_series'
elif MODEL_TYPE == 'atcnet':
    CFG = ATCNET_CONFIG
    MODE = 'time_series'
elif MODEL_TYPE == 'cnn':
    CFG = CNN_CONFIG
    MODE = 'spectrogram'
else:
    raise ValueError("Unknown Model Type")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = PROJECT_ROOT / "results" / f"best_{MODEL_TYPE}.pth"

def extract_features(model, loader):
    model.eval()
    features_list = []
    labels_list = []
    
    
    # Hooking the input to the Final Fully Connected Layer (fc).
    # This captures the "Features" right before the decision is made.
    activation = {}
    def get_activation(name):
        def hook(model, input, output):
            # input is a tuple, we want the first element (the feature vector)
            activation[name] = input[0].detach() 
        return hook

    # Attach hook to 'fc' layer (All your models have self.fc)
    handle = model.fc.register_forward_hook(get_activation('features'))

    print(f"Extracting features for {MODEL_TYPE}...")
    with torch.no_grad():
        for X, y in tqdm(loader):
            X = X.to(DEVICE)
            _ = model(X) # Forward pass triggers the hook
            
            feats = activation['features'].cpu().numpy()
            # Flatten just in case (Batch, Features)
            feats = feats.reshape(feats.shape[0], -1)
            
            features_list.append(feats)
            labels_list.append(y.numpy())
            
    handle.remove() # Clean up
    return np.concatenate(features_list), np.concatenate(labels_list)

def run_tsne():
    if not os.path.exists(MODEL_PATH):
        print(f"ERROR: Model file not found at {MODEL_PATH}")
        print(f"Did you train {MODEL_TYPE} yet?")
        return

    # Load test set data
    subs = list(SUBJECTS)
    import random
    random.seed(42)
    random.shuffle(subs)
    
    # Split logic must match train.py exactly
    n_train = int(len(subs) * 0.70)
    n_val = int(len(subs) * 0.15)
    test_subs = subs[n_train+n_val:]
    
    print(f"Loading Test Set: {len(test_subs)} subjects ({MODE})")
    test_ds = BCIDataset(test_subs, mode=MODE, augment=False)
    loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    # 2. Load Model
    print(f"Loading Model: {MODEL_PATH}")
    model = get_model(CFG).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    
    # 3. Extract
    features, labels = extract_features(model, loader)
    print(f"Feature Vector Shape: {features.shape}")

    # 4. Run t-SNE
    print("Computing t-SNE embeddings...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
    X_embedded = tsne.fit_transform(features)

    # 5. Plot
    plt.figure(figsize=(10, 8))
    df = pd.DataFrame(X_embedded, columns=['X', 'Y'])
    df['Label'] = ['Right' if l==1 else 'Left' for l in labels]
    
    # Colors: Left=Blue, Right=Red
    sns.scatterplot(data=df, x='X', y='Y', hue='Label', alpha=0.6, 
                    palette={'Left': '#4169E1', 'Right': '#DC143C'}, s=60)
    
    plt.title(f"Feature Space Visualization: {MODEL_TYPE.upper()} (Test Set)", fontsize=14)
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.legend(title="Motor Imagery")
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_file = PROJECT_ROOT / "results" / f"tsne_{MODEL_TYPE}.png"
    plt.savefig(save_file, dpi=300)
    print(f"Plot saved to: {save_file}")
    plt.show()

if __name__ == "__main__":
    run_tsne()
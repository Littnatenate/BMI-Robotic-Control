import sys
import os
import random
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
# UPDATED IMPORT: Added F1, Precision, Recall
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

# Project Imports
# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.config import PROJECT_ROOT, SUBJECTS
from src.models.eegnet import EEGNet
from src.models.atcnet import ATCNet
from src.models.spectrogram_cnn import SpectrogramCNN

# CONFIGURATION

EEGNET_CONFIG = {
    "model_name": "eegnet",
    "epochs": 150,
    "batch_size": 64,
    "lr": 0.001,
    "weight_decay": 0.01,
    "dropout": 0.5,
    "augment_type": "time_series",
    "mixup_alpha": 0.0,
    "time_shift": 10
}

ATCNET_CONFIG = {
    "model_name": "atcnet",
    "epochs": 150,
    "batch_size": 32,
    "lr": 0.0005,
    "weight_decay": 0.001,
    "dropout": 0.4,
    "augment_type": "time_series",
    "mixup_alpha": 0.0,
    "time_shift": 15
}

CNN_CONFIG = {
    "model_name": "cnn",
    "epochs": 100,
    "batch_size": 32,
    "lr": 0.001,
    "weight_decay": 0.001,
    "dropout": 0.5,
    "augment_type": "spectrogram",
    "mixup_alpha": 0.0,
    "time_shift": 0
}

# Select the model config to run
ACTIVE_CONFIG = ATCNET_CONFIG


class BCIDataset(Dataset):
    
    #Custom PyTorch Dataset for loading processed EEG/Spectrogram data.
    
    def __init__(self, subject_ids, mode='time_series', augment=False):
        self.X = []
        self.y = []
        self.mode = mode
        self.augment = augment
        
        # Determine file paths based on mode
        if mode == 'spectrogram':
            data_dir = PROJECT_ROOT / "Datasets" / "processed"
            suffix = "_spectrograms.pkl"
        else:
            data_dir = PROJECT_ROOT / "Datasets" / "processed_eegnet"
            suffix = "_eegnet_features.pkl"

        print(f"Loading {len(subject_ids)} subjects ({mode})...")
        
        for sub in tqdm(subject_ids, desc="Loading Data"):
            sub_str = f"S{sub:03d}"
            fpath = data_dir / sub_str / f"{sub_str}{suffix}"
            
            if not fpath.exists():
                continue

            try:
                with open(fpath, 'rb') as f:
                    data = pickle.load(f)
                
                # Filter: Keep only Imagined Left (2) and Imagined Right (3)
                mask = np.isin(data['y'], [2, 3])
                
                if len(mask) == 0:
                    continue
                
                self.X.append(data['X'][mask])
                
                # Remap Labels: 2 -> 0 (Left), 3 -> 1 (Right)
                # This ensures the model sees binary labels (0, 1)
                remapped_y = np.where(data['y'][mask] == 2, 0, 1)
                self.y.append(remapped_y)
                
            except Exception as e:
                print(f"Error loading {sub_str}: {e}")

        if not self.X:
            raise RuntimeError("No data found! Check paths or subject IDs.")
            
        self.X = np.concatenate(self.X, axis=0)
        self.y = np.concatenate(self.y, axis=0)
        
        # Add Channel Dimension for Time Series Models: (N, 64, 320) -> (N, 1, 64, 320)
        if mode == 'time_series': 
            self.X = self.X[:, np.newaxis, :, :]

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx].copy()
        y = self.y[idx]
        
        if self.augment:
            x = self._apply_augmentation(x)

        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long)

    def _apply_augmentation(self, x):
        #Helper method to apply data augmentation
        if self.mode == 'time_series':
            # 1. Gaussian Noise
            x += np.random.normal(0, 0.2, x.shape) 
            # 2. Amplitude Scaling
            x *= np.random.uniform(0.95, 1.05)
            # 3. Time Shift
            shift_limit = ACTIVE_CONFIG['time_shift']
            shift = np.random.randint(-shift_limit, shift_limit)
            if shift != 0:
                x = np.roll(x, shift, axis=-1)
        
        elif self.mode == 'spectrogram':
            # Frequency Masking
            if random.random() < 0.5:
                freqs = x.shape[1]
                mask_width = random.randint(2, 6)
                f0 = random.randint(0, freqs - mask_width)
                x[:, f0:f0+mask_width, :] = 0.0

            # Light Gaussian Noise
            if random.random() < 0.5:
                noise = np.random.normal(0, 0.05, x.shape)
                x += noise
        
        return x


def get_model(cfg):
    # Factory function to initialize the selected model.
    name = cfg['model_name']
    drop = cfg['dropout']
    
    if name == 'eegnet':
        # F1=8 is standard EEGNet. D=2 makes F2=16.
        return EEGNet(dropoutRate=drop, F1=8, D=2, kernLength=64)
    
    if name == 'atcnet':
        return ATCNet(dropout=drop)
    
    if name == 'cnn':
        return SpectrogramCNN(dropout_rate=drop)
    
    raise ValueError(f"Unknown Model Name: {name}")


def train():
    # Main Training Loop
    # Initialize W&B
    wandb.init(
        project="BMI-Robotic-Control", 
        config=ACTIVE_CONFIG, 
        name=ACTIVE_CONFIG['model_name']
    )
    cfg = wandb.config
    
    # Data Split
    subs = list(SUBJECTS)
    
    # Random Shuffle with Fixed Seed
    # Ensures consistent splits across runs but shuffles subjects
    random.seed(42) 
    random.shuffle(subs)
    
    # 70% Train, 15% Val, 15% Test
    n_train = int(len(subs) * 0.70)
    n_val = int(len(subs) * 0.15)
    
    train_subs = subs[:n_train]
    val_subs = subs[n_train : n_train + n_val]
    test_subs = subs[n_train + n_val :]
    
    print(f"SPLIT: Train {len(train_subs)} | Val {len(val_subs)} | Test {len(test_subs)}")
    
    # Load Data
    if cfg.model_name == 'cnn':
        mode = 'spectrogram'
    else:
        mode = 'time_series'
    
    train_ds = BCIDataset(train_subs, mode, augment=True)
    val_ds = BCIDataset(val_subs, mode, augment=False)
    test_ds = BCIDataset(test_subs, mode, augment=False)
    
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size)
    
    # Setup Model & Optimiser
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(cfg).to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    criterion = nn.CrossEntropyLoss()
    
    best_acc = 0.0
    save_path = PROJECT_ROOT / "results" / f"best_{cfg.model_name}.pth"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Training Loop
    print(f"Starting Training ({cfg.model_name})...")
    
    for epoch in range(cfg.epochs):
        model.train()
        t_loss = 0
        t_corr = 0
        t_tot = 0
        
        for X, y in train_loader:
            X, y = X.to(device), y.to(device)
            
            # MixUp Augmentation
            if cfg.mixup_alpha > 0:
                lam = np.random.beta(cfg.mixup_alpha, cfg.mixup_alpha)
                idx = torch.randperm(X.size(0)).to(device)
                
                mixed_X = lam * X + (1 - lam) * X[idx, :]
                y_a, y_b = y, y[idx]
                
                out = model(mixed_X)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                out = model(X)
                loss = criterion(out, y)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            t_loss += loss.item()
            t_corr += (out.argmax(1) == y).sum().item()
            t_tot += y.size(0)
            
        # Update Scheduler
        scheduler.step()
        
        # Validation Step
        model.eval()
        v_corr = 0
        v_tot = 0
        
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(device), y.to(device)
                v_corr += (model(X).argmax(1) == y).sum().item()
                v_tot += y.size(0)
        
        # Calculate Metrics
        train_acc = t_corr / t_tot
        val_acc = v_corr / v_tot
        avg_loss = t_loss / len(train_loader)
        
        # Logging
        if (epoch + 1) % 5 == 0:
            print(f"Ep {epoch+1:03d} | Train: {train_acc:.2%} | Val: {val_acc:.2%}")
            
        wandb.log({
            "train_acc": train_acc, 
            "val_acc": val_acc, 
            "loss": avg_loss
        })
        
        # Save Best Model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    # Final Evaluation
    print("\n--- FINAL TEST EVALUATION ---")
    model.load_state_dict(torch.load(save_path))
    model.eval()
    
    preds = []
    targets = []
    
    with torch.no_grad():
        for X, y in test_loader:
            X, y = X.to(device), y.to(device)
            preds.extend(model(X).argmax(1).cpu().numpy())
            targets.extend(y.cpu().numpy())
    
    # Metrics
    final_acc = accuracy_score(targets, preds)
    final_f1 = f1_score(targets, preds, average='weighted')
    final_prec = precision_score(targets, preds, average='weighted')
    final_rec = recall_score(targets, preds, average='weighted')
    
    print(f"FINAL TEST ACCURACY:  {final_acc:.2%}")
    print(f"FINAL TEST F1-SCORE:  {final_f1:.4f}")
    print(f"FINAL TEST PRECISION: {final_prec:.4f}")
    print(f"FINAL TEST RECALL:    {final_rec:.4f}")
    
    wandb.log({
        "test_acc": final_acc,
        "test_f1": final_f1,
        "test_precision": final_prec,
        "test_recall": final_rec
    })
    
    # Plot Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(5, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f"Confusion Matrix ({cfg.model_name})")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    
    wandb.log({"confusion_matrix": wandb.Image(plt)})
    wandb.finish()

if __name__ == "__main__":
    train()
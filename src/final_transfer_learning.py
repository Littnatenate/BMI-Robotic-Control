"""
STEP 2: USER CALIBRATION (TRANSFER LEARNING)
============================================
Description:
    Fine-tunes the general model for a specific user to boost accuracy.
    - 'Unleashed' mode: Updates all weights (Best for EEGNet/ATCNet).
    - 'Safe' mode: Freezes feature extractor, updates classifier only (Best for CNN).

Author: Nathan Yu
"""

import sys
import os
import csv
import random
import pickle
import gc
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import wandb

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# Project Imports
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.append(parent_dir)

from src.config import PROJECT_ROOT, SUBJECTS, RESULTS_DIR
from src.models.eegnet import EEGNet
from src.models.atcnet import ATCNet
from src.models.spectrogram_cnn import SpectrogramCNN

#              CONFIGURATION

# Select Model Type
MODEL_TYPE = 'atcnet'  # Options: 'eegnet', 'atcnet', 'cnn'

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PRETRAINED_PATH = RESULTS_DIR / f"best_{MODEL_TYPE}.pth"

# Transfer Learning Hyperparameters
TL_CONFIG = {
    'eegnet': {
        'lr': 0.00001,
        'epochs': 40,
        'batch_size': 6,
        'suffix': '_eegnet_features.pkl',
        'folder': 'processed_eegnet',
        'freeze': False
    },
    'atcnet': {
        'lr': 0.0001,
        'epochs': 40,
        'batch_size': 6,
        'suffix': '_eegnet_features.pkl',
        'folder': 'processed_eegnet',
        'freeze': False
    },
    'cnn': {
        'lr': 0.0005,
        'epochs': 20,
        'batch_size': 8,
        'suffix': '_spectrograms.pkl',
        'folder': 'processed',
        'freeze': True   # Safe Mode
    }
}

# Load specific config
CFG = TL_CONFIG[MODEL_TYPE]


class CalibrationDataset(Dataset):
    
    # Dataset for fine-tuning. Includes specific augmentations for small data regimes.
    def __init__(self, X, y, mode, augment=False):
        self.X = X
        self.y = y
        self.mode = mode
        self.augment = augment
        
        # Add Channel Dimension for Time Series if missing
        if self.mode == 'time_series' and self.X.ndim == 3:
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
        #Applies noise, scaling, and shift to prevent overfitting on small data.
        if self.mode == 'time_series':
            # 1. Gaussian Noise
            x += np.random.normal(0, 0.02, x.shape)
            # 2. Amplitude Scaling
            x *= np.random.uniform(0.95, 1.05) # Multiply
            # 3. Time Shift
            shift = np.random.randint(-10, 10)
            if shift != 0:
                x = np.roll(x, shift, axis=-1)
        elif self.mode == 'spectrogram':
            x += np.random.normal(0, 0.05, x.shape)
        return x


def load_subject_data(sub_id):
    #Loads and filters data for a specific subject.
    folder = f"S{sub_id:03d}"
    path = PROJECT_ROOT / "Datasets" / CFG['folder'] / folder / f"{folder}{CFG['suffix']}"
    
    if not path.exists():
        return None, None
        
    try:
        with open(path, "rb") as f:
            data = pickle.load(f)
            
        # Filter: Keep only Imagined Movement (Classes 2 and 3)
        mask = np.isin(data['y'], [2, 3])
        X = data['X'][mask]
        
        # Remap: 2->0, 3->1
        y = np.where(data['y'][mask] == 2, 0, 1)
        
        return X, y
        
    except Exception as e:
        print(f"Error loading {folder}: {e}")
        return None, None


def get_fresh_model():
    #Initializes a new model instance based on global config.
    if MODEL_TYPE == 'eegnet':
        return EEGNet(dropoutRate=0.5, F1=8, D=2, kernLength=64)
    if MODEL_TYPE == 'atcnet':
        return ATCNet(dropout=0.5)
    if MODEL_TYPE == 'cnn':
        return SpectrogramCNN(dropout_rate=0.5)
    return None


def evaluate_model(model, loader):
    #Helper to calculate metrics.
    model.eval()
    preds, targets = [], []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)
            preds.extend(model(inputs).argmax(1).cpu().numpy())
            targets.extend(labels.cpu().numpy())
            
    return preds, targets


def main():
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"\n--- STARTING CALIBRATION: {MODEL_TYPE.upper()} ---")
    
    if not PRETRAINED_PATH.exists():
        print(f"Error: Pre-trained model not found at {PRETRAINED_PATH}")
        return

    # Load Baseline Weights
    print("Loading Pre-trained Weights...")
    base_weights = torch.load(PRETRAINED_PATH, map_location=DEVICE)

    # Setup Split (Use Test Subjects only)
    all_subs = list(SUBJECTS)
    random.seed(42)
    random.shuffle(all_subs)
    
    n_train = int(0.7 * len(all_subs))
    n_val = int(0.15 * len(all_subs))
    test_subs = all_subs[n_train + n_val :]
    
    print(f"Targeting {len(test_subs)} Test Subjects for Calibration.")

    # Initialize W&B
    wandb.init(
        project="BMI-Robotic-Control", 
        name=f"Calibration_{MODEL_TYPE.upper()}", 
        config=CFG
    )
    
    results_table = []
    
    # Calibration Loop
    for sub_id in tqdm(test_subs, desc="Calibrating Users"):
        
        # Load Data
        X, y = load_subject_data(sub_id)
        if X is None or len(X) < 20:
            continue 

        # 50/50 Split (Calibration Set / Evaluation Set)
        split = int(len(X) * 0.5)
        X_cal, y_cal = X[:split], y[:split]
        X_eval, y_eval = X[split:], y[split:]
        
        # Baseline Metrics (Zero-Shot)
        model = get_fresh_model().to(DEVICE)
        model.load_state_dict(base_weights)
        
        if MODEL_TYPE == 'cnn': data_mode = 'spectrogram'
        else: data_mode = 'time_series'
            
        eval_ds = CalibrationDataset(X_eval, y_eval, data_mode, augment=False)
        eval_loader = DataLoader(eval_ds, batch_size=CFG['batch_size'], shuffle=False)
        
        # Calculate Baseline Accuracy
        preds_base, targets_base = evaluate_model(model, eval_loader)
        baseline_acc = accuracy_score(targets_base, preds_base)

        # Setup Transfer Learning
        if CFG['freeze']:
            for param in model.parameters(): param.requires_grad = False
            for param in model.fc.parameters(): param.requires_grad = True
            optimizer = optim.Adam(model.fc.parameters(), lr=CFG['lr'])
        else:
            for param in model.parameters(): param.requires_grad = True
            optimizer = optim.AdamW(model.parameters(), lr=CFG['lr'], weight_decay=0.01)
        
        # Fine-Tune Loop
        cal_ds = CalibrationDataset(X_cal, y_cal, data_mode, augment=True)
        cal_loader = DataLoader(cal_ds, batch_size=CFG['batch_size'], shuffle=True)
        criterion = nn.CrossEntropyLoss()
        
        for epoch in range(CFG['epochs']):
            model.train()
            if CFG['batch_size'] < 16:
                for module in model.modules():
                    if isinstance(module, (nn.BatchNorm2d, nn.BatchNorm1d)): module.eval()

            for inputs, labels in cal_loader:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(model(inputs), labels)
                loss.backward()
                optimizer.step()
        
        # Final Metrics
        preds_final, targets_final = evaluate_model(model, eval_loader)
        
        final_acc = accuracy_score(targets_final, preds_final)
        final_f1 = f1_score(targets_final, preds_final, average='weighted')
        
        delta = final_acc - baseline_acc
        
        # Store results
        results_table.append([f"S{sub_id:03d}", baseline_acc, final_acc, final_f1, delta])

        # Save Calibrated Model
        if final_acc > 0.70:
            save_dir = RESULTS_DIR / "calibrated_models"
            save_dir.mkdir(exist_ok=True, parents=True)
            torch.save(model.state_dict(), save_dir / f"S{sub_id:03d}_{MODEL_TYPE}_calibrated.pth")

    # REPORTING
    if not results_table:
        print("No subjects processed successfully.")
        return

    # Sort by Final Accuracy
    results_table.sort(key=lambda x: x[2], reverse=True)
    
    mean_final_acc = np.mean([r[2] for r in results_table])
    mean_final_f1 = np.mean([r[3] for r in results_table])
    mean_delta = np.mean([r[4] for r in results_table])
    
    print("\n" + "="*40)
    print(f"CALIBRATION RESULTS: {MODEL_TYPE.upper()}")
    print(f"Mean Accuracy:   {mean_final_acc:.2%}")
    print(f"Mean F1 Score:   {mean_final_f1:.4f}")
    print(f"Avg Improvement: {mean_delta:+.2%}")
    print("="*40)
    
    # Save CSV Report
    csv_path = RESULTS_DIR / "calibrated_models" / f"{MODEL_TYPE}_leaderboard.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Subject", "Baseline_Acc", "Calibrated_Acc", "Calibrated_F1", "Improvement"])
        writer.writerows(results_table)
    
    # Log to WandB
    wandb.log({
        "final_mean_accuracy": mean_final_acc,
        "final_mean_f1": mean_final_f1,
        "avg_improvement": mean_delta,
        "calibration_results": wandb.Table(
            data=results_table, 
            columns=["Subject", "Baseline", "Acc", "F1", "Improvement"]
        )
    })
    wandb.finish()

if __name__ == "__main__":
    main()
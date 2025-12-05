"""
REAL-TIME INFERENCE: ROBOT CONTROL SIMULATION
=============================================
Description:
    1. Loads the trained ATCNet model.
    2. Simulates a "Live Stream" of EEG data from a test file.
    3. Processes data in 2-second chunks (sliding window).
    4. Outputs Motor Commands (LEFT / RIGHT) to the terminal.
"""

import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from pathlib import Path
import logging

# --- CONFIGURATION ---
MODEL_PATH = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\results\models\atcnet_v4\best_atcnet_v4.pth")
# Pick a random subject to simulate (e.g., S108)
TEST_SUBJECT_FILE = Path(r"C:\Users\524yu\OneDrive\Documents\VSCODEE\BMI-Robotic-Control\Datasets\processed_eegnet\S108\S108_eegnet_features.pkl")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONFIDENCE_THRESHOLD = 0.70  # Only move robot if AI is 70% sure

# --- ARCHITECTURE (Must match saved model) ---
class TCNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dropout):
        super(TCNBlock, self).__init__()
        self.padding = (kernel_size - 1) // 2 
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=self.padding)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=self.padding)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x):
        res = self.residual(x)
        out = F.elu(self.bn1(self.conv1(x)))
        out = self.dropout1(out)
        out = F.elu(self.bn2(self.conv2(out)))
        out = self.dropout2(out)
        if out.shape[2] != res.shape[2]:
            target = min(out.shape[2], res.shape[2])
            out = out[:, :, :target]
            res = res[:, :, :target]
        return F.elu(out + res)

class AttentionBlock(nn.Module):
    def __init__(self, input_dim):
        super(AttentionBlock, self).__init__()
        self.query = nn.Linear(input_dim, input_dim)
        self.key = nn.Linear(input_dim, input_dim)
        self.value = nn.Linear(input_dim, input_dim)
        self.scale = input_dim ** -0.5

    def forward(self, x):
        Q = self.query(x); K = self.key(x); V = self.value(x)
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        weights = F.softmax(scores, dim=-1)
        return torch.matmul(weights, V)

class ATCNet(nn.Module):
    def __init__(self, nb_classes=2):
        super(ATCNet, self).__init__()
        # Fixed params to match training
        Chans, Samples, dropoutRate = 64, 320, 0.5
        F1, T_KERNEL = 16, 64
        
        self.conv_temp = nn.Conv2d(1, F1, (1, T_KERNEL), padding=(0, T_KERNEL//2), bias=False)
        self.bn_temp = nn.BatchNorm2d(F1)
        self.conv_spatial = nn.Conv2d(F1, F1, (Chans, 1), groups=F1, bias=False)
        self.bn_spatial = nn.BatchNorm2d(F1)
        self.pool = nn.AvgPool2d((1, 8))
        self.dropout = nn.Dropout(dropoutRate)
        self.tcn1 = TCNBlock(F1, 32, kernel_size=31, dropout=dropoutRate)
        self.tcn2 = TCNBlock(32, 32, kernel_size=31, dropout=dropoutRate)
        self.att = AttentionBlock(32)
        with torch.no_grad():
            dummy = torch.zeros(1, 1, Chans, Samples)
            x = self.forward_features(dummy)
            flat_size = x.view(1, -1).size(1)
        self.fc = nn.Linear(flat_size, nb_classes)
        
    def forward_features(self, x):
        x = self.conv_temp(x); x = self.bn_temp(x)
        x = self.conv_spatial(x); x = self.bn_spatial(x); x = F.elu(x)
        x = self.pool(x); x = self.dropout(x)
        x = x.squeeze(2) 
        x = self.tcn1(x); x = self.tcn2(x)
        x = x.transpose(1, 2)
        x = self.att(x)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = x.reshape(x.size(0), -1)
        x = self.fc(x)
        return x

# --- ROBOT CONTROLLER ---
def simulate_robot_control():
    print("="*60)
    print("Initializing BCI Robot Controller...")
    
    # 1. Load Brain
    model = ATCNet(nb_classes=2).to(DEVICE)
    try:
        model.load_state_dict(torch.load(MODEL_PATH))
        model.eval()
        print("✅ AI Model Loaded Successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}"); return

    # 2. Load Data Stream
    print(f"📡 Connecting to data stream: {TEST_SUBJECT_FILE.name}")
    with open(TEST_SUBJECT_FILE, 'rb') as f:
        data = pickle.load(f)
    
    # Filter for imagined only
    mask = np.isin(data['y'], [2, 3])
    stream_X = data['X'][mask]
    stream_y = data['y'][mask] # (2=Left, 3=Right)
    
    print(f"🔴 Stream Active. Processing {len(stream_X)} packets...")
    print("="*60)
    print(f"{'TIMESTAMP':<10} | {'PREDICTION':<15} | {'CONFIDENCE':<10} | {'ACTION'}")
    print("-" * 60)
    
    # 3. Real-Time Loop
    correct_cmds = 0
    
    for i in range(len(stream_X)):
        # Simulate 0.1s delay between packets
        time.sleep(0.05) 
        
        # Get "Live" chunk (1, 64, 320)
        raw_chunk = stream_X[i]
        true_label = stream_y[i] # 2 or 3
        
        # Preprocess (Add batch dims)
        # Input: (64, 320) -> Tensor: (1, 1, 64, 320)
        tensor_chunk = torch.tensor(raw_chunk, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)
        
        # Inference
        with torch.no_grad():
            outputs = model(tensor_chunk)
            probs = F.softmax(outputs, dim=1) # [0.1, 0.9]
            
        conf, pred_idx = torch.max(probs, 1)
        confidence = conf.item()
        prediction = pred_idx.item() # 0 or 1
        
        # Map Prediction to Action
        # Training mapped 2->0 (Left), 3->1 (Right)
        cmd_str = "LEFT  <<" if prediction == 0 else "RIGHT >>"
        
        # Check Ground Truth
        ground_truth = 0 if true_label == 2 else 1
        is_correct = (prediction == ground_truth)
        if is_correct: correct_cmds += 1
        
        # Logic Gate: Only move if confident
        status = "HOLD"
        if confidence > CONFIDENCE_THRESHOLD:
            status = f"** MOVE {cmd_str.strip()} **"
        
        # Visualize
        color = "\033[92m" if is_correct else "\033[91m" # Green/Red
        reset = "\033[0m"
        
        print(f"{i*2.0:.1f}s       | {cmd_str:<15} | {confidence:.1%}    | {color}{status}{reset}")

    print("="*60)
    print(f"Session Ended. Command Accuracy: {correct_cmds/len(stream_X):.1%}")

if __name__ == "__main__":
    simulate_robot_control()
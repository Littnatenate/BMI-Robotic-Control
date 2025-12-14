"""
FINAL DEMO: NEURO-ROVER SIMULATION
==================================
A real-time visualization of the BCI system controlling a virtual rover.

KEY FEATURES:
1. Isometric View: Prevents graphical glitches on High-DPI screens.
2. Safety Logic: The rover only steers if confidence > 70%.
3. Data Stream: Loads unseen test data from the Subject 29 dataset.
"""

import sys
import os
import ctypes
import torch
import numpy as np
import pickle
import random
from ursina import *
from torch.utils.data import DataLoader, Dataset

# --- 1. SYSTEM SETUP ---------------------------------------------------------

# FIX: High-DPI Scaling (Prevents "Half-Screen" glitch on laptops)
try:
    ctypes.windll.user32.SetProcessDPIAware()
except: 
    pass

# PATH SETUP: Allow importing from 'src' folder
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.config import PROJECT_ROOT
from src.models.eegnet import EEGNet

# CONFIGURATION
TARGET_SUBJECT = 29
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "results/calibrated_models" / f"S{TARGET_SUBJECT:03d}_eegnet_calibrated.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# PHYSICS SETTINGS
CONFIDENCE_THRESHOLD = 0.70  # Signals below 70% are treated as noise/idle
CENTERING_SPEED = 2.0        # How fast the rover straightens out
STEERING_SPEED = 4.0         # How responsive the turning is

# --- 2. DATA MANAGEMENT ------------------------------------------------------

class CalibrationDataset(Dataset):
    """
    Standard PyTorch Dataset to handle EEG Data.
    """
    def __init__(self, X, y):
        self.X = X
        self.y = y
        # Ensure data is 4D: (Batch, Channel, Electrodes, Time)
        if self.X.ndim == 3: 
            self.X = self.X[:, np.newaxis, :, :]

    def __len__(self): 
        return len(self.X)

    def __getitem__(self, idx): 
        return (torch.tensor(self.X[idx], dtype=torch.float32), 
                torch.tensor(self.y[idx], dtype=torch.long))

def load_test_data():
    """
    Loads the processed features (.pkl) for the target subject.
    Returns: X (Features), y (Labels)
    """
    folder = f"S{TARGET_SUBJECT:03d}"
    path = PROJECT_ROOT / "Datasets" / "processed_eegnet" / folder / f"{folder}_eegnet_features.pkl"
    
    try:
        with open(path, "rb") as f: 
            data = pickle.load(f)
        
        # Filter for Imagined Movement only (Classes 2 and 3)
        mask = np.isin(data['y'], [2, 3])
        X = data['X'][mask]
        
        # Remap labels: 2 -> 0 (Left), 3 -> 1 (Right)
        y = np.where(data['y'][mask] == 2, 0, 1)
        
        return X, y
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

# --- 3. APP INITIALIZATION ---------------------------------------------------

app = Ursina()
window.title = "Neuro-Rover Simulator"
window.color = color.rgb(20, 20, 25) # Professional Dark Grey
window.borderless = False
window.fullscreen = False
window.size = (1280, 720) # Standard HD Resolution
window.center_on_screen()

# --- 4. BUILDING THE 3D WORLD ------------------------------------------------

# THE PLAYER (ROVER)
# We build the car using basic shapes so no external files are needed.
player = Entity(position=(0, 0, 0))

# Car Body
body = Entity(parent=player, model='cube', scale=(1.2, 0.6, 2.5), y=0.5, color=color.dark_gray)
roof = Entity(parent=player, model='cube', scale=(1.0, 0.5, 1.5), y=1.0, z=-0.2, color=color.black66)

# Headlights (We change this color to show status)
lights = Entity(parent=player, model='cube', scale=(1.0, 0.1, 0.1), 
                position=(0, 0.6, 1.25), color=color.yellow, unlit=True)

# Wheels (4 Cylinders)
wheel_positions = [
    (-0.6, 0.3, 0.8),  (0.6, 0.3, 0.8),   # Front Left, Front Right
    (-0.6, 0.3, -0.8), (0.6, 0.3, -0.8)   # Rear Left, Rear Right
]
for pos in wheel_positions:
    Entity(parent=player, model='cylinder', scale=(0.6, 0.2, 0.6), 
           rotation_z=90, position=pos, color=color.black)

# THE ENVIRONMENT
# We create floating lines that move backward to simulate speed.
lines = []
for i in range(25):
    e = Entity(model='cube', scale=(0.1, 0.1, 4), color=color.gray, 
               position=(random.randint(-15, 15), 0, random.randint(-20, 20)))
    lines.append(e)

# THE CAMERA
# Isometric View (Like SimCity) - Stable and clean.
camera.orthographic = True
camera.fov = 15
camera.position = (20, 20, -20)
camera.look_at(player)

# HUD (UI)
Text(text="NEURAL NAVIGATION ONLINE", position=(-0.85, 0.45), color=color.cyan)
status_bg = Entity(parent=camera.ui, model='quad', scale=(0.8, 0.1), 
                   position=(0, 0.4), color=color.black66)
status_text = Text(text="SYSTEM IDLE", position=(0, 0.41), origin=(0,0), 
                   scale=1.5, color=color.white)

# --- 5. LOGIC & LOOP ---------------------------------------------------------

# Load Data
print("Loading AI Model & Data...")
X, y = load_test_data()
if X is None: sys.exit("Data Error")

# Use last 50% of data (Unseen Test Set)
split = int(len(X) * 0.5)
dataset = CalibrationDataset(X[split:], y[split:])
loader = DataLoader(dataset, batch_size=1, shuffle=False)
data_iter = iter(loader)

# Load Model
model = EEGNet(nb_classes=2, Chans=64, Samples=320).to(DEVICE)
try: 
    model.load_state_dict(torch.load(CALIBRATED_MODEL_PATH, map_location=DEVICE))
    model.eval()
except: 
    sys.exit("Model Weights Not Found")

# Global State Variables
target_x = 0
current_state = "REST"
inference_timer = 0

def update():
    """
    Runs every frame (60 times per second).
    Handles animation and physics.
    """
    global target_x, inference_timer, current_state
    
    # 1. ANIMATE ENVIRONMENT (Fake Speed)
    for line in lines:
        line.z -= time.dt * 20
        if line.z < -10:
            line.z = 20
            line.x = random.randint(-15, 15)

    # 2. DYNAMICS LOGIC
    if current_state == "REST":
        # Cruise Control: Gently center the car
        target_x = lerp(target_x, 0, time.dt * CENTERING_SPEED)
        lights.color = color.yellow
    else:
        # Active Steering: Lights indicate direction
        lights.color = color.cyan if current_state == "RIGHT" else color.green

    # 3. APPLY MOVEMENT (Smooth Interpolation)
    player.x = lerp(player.x, target_x, time.dt * STEERING_SPEED)
    
    # Tilt the car when turning (Visual Polish)
    tilt_amount = (player.x - target_x) * 15
    player.rotation_z = lerp(player.rotation_z, tilt_amount, time.dt * 5)

    # 4. TRIGGER BCI (Every 1.2 seconds)
    inference_timer += time.dt
    if inference_timer > 1.2:
        inference_timer = 0
        try:
            run_bci_prediction()
        except StopIteration:
            status_text.text = "SESSION COMPLETE"
            status_text.color = color.white

def run_bci_prediction():
    """
    Runs one batch of EEG data through the AI.
    Updates the car's target destination.
    """
    global target_x, current_state
    
    # Get next trial
    X_batch, y_true = next(data_iter)
    X_batch = X_batch.to(DEVICE)
    
    # Run Model
    with torch.no_grad():
        out = model(X_batch)
        conf, pred = torch.max(torch.softmax(out, dim=1), 1)
        
    cls = pred.item()
    confidence = conf.item()
    
    # --- SAFETY LOGIC ---
    if confidence < CONFIDENCE_THRESHOLD:
        current_state = "REST"
        status_text.text = f"CRUISING (Signal Weak: {confidence:.0%})"
        status_text.color = color.yellow
        
    elif cls == 0: # LEFT
        current_state = "LEFT"
        status_text.text = f"◀ TURNING LEFT ({confidence:.0%})"
        status_text.color = color.green
        target_x -= 3
        
    else: # RIGHT
        current_state = "RIGHT"
        status_text.text = f"TURNING RIGHT ({confidence:.0%}) ▶"
        status_text.color = color.cyan
        target_x += 3

# Start the Engine
app.run()
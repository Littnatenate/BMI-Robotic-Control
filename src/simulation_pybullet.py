"""
FINAL DEMO: CINEMATIC BCI RACECAR
=================================
Visuals:
- Robot: MIT Racecar (Sleek, high-tech look).
- Floor: Solid Matte Dark Grey (No checkerboard).
- World: Random debris/obstacles to show movement.
- Camera: Low-angle 'Chase Cam'.
"""

import os
import sys
import time
import pickle
import random
import numpy as np
import torch
import pybullet as p
import pybullet_data
from torch.utils.data import DataLoader, Dataset

# --- PATH SETUP ---
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from src.config import PROJECT_ROOT
from src.models.eegnet import EEGNet

# --- CONFIGURATION ---
TARGET_SUBJECT = 29
CALIBRATED_MODEL_PATH = PROJECT_ROOT / "results/calibrated_models" / f"S{TARGET_SUBJECT:03d}_eegnet_calibrated.pth"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# DRIVING PHYSICS (Racecar is faster)
BASE_SPEED = 20.0
TURN_SPEED = 0.5 

# --- HELPER CLASSES ---
class CalibrationDataset(Dataset):
    def __init__(self, X, y, mode='time_series', augment=False):
        self.X = X; self.y = y; self.mode = mode
        if self.mode == 'time_series' and self.X.ndim == 3: self.X = self.X[:, np.newaxis, :, :]
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

def load_subject_data(sub_id):
    folder = f"S{sub_id:03d}"
    path = PROJECT_ROOT / "Datasets" / "processed_eegnet" / folder / f"{folder}_eegnet_features.pkl"
    if not path.exists(): sys.exit(f"❌ Error: Data not found: {path}")
    try:
        with open(path, "rb") as f: data = pickle.load(f)
        mask = np.isin(data['y'], [2, 3])
        return data['X'][mask], np.where(data['y'][mask] == 2, 0, 1)
    except: return None, None

# --- 1. SETUP VISUALS (THE MAKEOVER) ---
physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# HIDE UI
p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1) # Enable Shadows!

# CUSTOM FLOOR (Solid Color)
# We create a massive box instead of the default plane to hide the grid
visualShapeId = p.createVisualShape(shapeType=p.GEOM_BOX, halfExtents=[100, 100, 0.1], rgbaColor=[0.15, 0.15, 0.18, 1])
collisionShapeId = p.createCollisionShape(shapeType=p.GEOM_BOX, halfExtents=[100, 100, 0.1])
floorId = p.createMultiBody(baseMass=0, baseCollisionShapeIndex=collisionShapeId, baseVisualShapeIndex=visualShapeId, basePosition=[0, 0, -0.1])

# SPAWN OBSTACLES (To give a sense of speed)
# We place random cubes around so you can see the car moving past them
for i in range(30):
    x = random.uniform(-20, 20)
    y = random.uniform(-5, 100) # Ahead of the car
    col = [random.uniform(0.3, 0.5) for _ in range(3)] + [1]
    
    vis = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5], rgbaColor=col)
    col_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5, 0.5, 0.5])
    p.createMultiBody(0, col_shape, vis, [x, y, 0.5])

# --- LOAD RACECAR ---
# MIT Racecar is sleek and looks better than the Husky
carId = p.loadURDF("racecar/racecar.urdf", [0, 0, 0.2], useFixedBase=False)

# Joint Indices for Racecar (Steering: 4, 6 | Driving: 2, 3, 5, 7)
steering_joints = [4, 6]
drive_joints = [2, 3, 5, 7]

# TEXT DASHBOARD
txt_status = p.addUserDebugText("SYSTEM ONLINE", [0, 0, 2], [1, 1, 1], textSize=2.0)

# --- 2. LOAD BRAIN ---
print(f"Loading S{TARGET_SUBJECT}...")
X, y = load_subject_data(TARGET_SUBJECT)
split_idx = int(len(X) * 0.5)
test_loader = DataLoader(CalibrationDataset(X[split_idx:], y[split_idx:]), batch_size=1, shuffle=False)

model = EEGNet(nb_classes=2, Chans=64, Samples=320).to(DEVICE)
try:
    model.load_state_dict(torch.load(CALIBRATED_MODEL_PATH, map_location=DEVICE))
    model.eval()
except: sys.exit("❌ Model load failed.")

# --- 3. DRIVING LOGIC ---
def drive_racecar(steering_angle, speed):
    # Apply Steering
    for joint in steering_joints:
        p.setJointMotorControl2(carId, joint, p.POSITION_CONTROL, targetPosition=steering_angle)
    # Apply Speed
    for joint in drive_joints:
        p.setJointMotorControl2(carId, joint, p.VELOCITY_CONTROL, targetVelocity=speed, force=10)

# --- 4. MAIN LOOP ---
print("Starting Cinematic Racecar Demo...")

steering_target = 0.0

for i, (X_batch, y_batch) in enumerate(test_loader):
    X_batch = X_batch.to(DEVICE)
    y_true = y_batch.item()
    
    # Inference
    with torch.no_grad():
        out = model(X_batch)
        conf, pred = torch.max(torch.softmax(out, dim=1), 1)
    
    pred_cls = pred.item()
    confidence = conf.item()
    
    # LOGIC
    if pred_cls == 0: # LEFT
        status_text = f"◀ LEFT ({confidence:.0%})"
        color = [0, 1, 0] # Green
        steering_target = 0.5 # Turn wheels left
    else: # RIGHT
        status_text = f"RIGHT ({confidence:.0%}) ▶"
        color = [1, 0, 0] # Red
        steering_target = -0.5 # Turn wheels right
    
    # Update UI (Floating above car)
    car_pos, car_ori = p.getBasePositionAndOrientation(carId)
    p.addUserDebugText(status_text, [car_pos[0], car_pos[1], car_pos[2]+1.5], color, textSize=2.0, replaceItemUniqueId=txt_status, lifeTime=1.5)

    # EXECUTE (Drive for 1.5 seconds)
    for _ in range(100): 
        p.stepSimulation()
        drive_racecar(steering_target, -BASE_SPEED) # Negative speed because racecar model is flipped
        
        # CHASE CAMERA LOGIC
        car_pos, car_ori = p.getBasePositionAndOrientation(carId)
        # Calculate camera position behind the car
        yaw = p.getEulerFromQuaternion(car_ori)[2] * (180/3.14159)
        
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=yaw-90, cameraPitch=-20, cameraTargetPosition=car_pos)
        
        time.sleep(1./240.)

    # Straighten out briefly between trials
    drive_racecar(0, -BASE_SPEED)
    for _ in range(20):
        p.stepSimulation()
        car_pos, car_ori = p.getBasePositionAndOrientation(carId)
        yaw = p.getEulerFromQuaternion(car_ori)[2] * (180/3.14159)
        p.resetDebugVisualizerCamera(cameraDistance=2.5, cameraYaw=yaw-90, cameraPitch=-20, cameraTargetPosition=car_pos)
        time.sleep(1./240.)

print("Demo Complete.")
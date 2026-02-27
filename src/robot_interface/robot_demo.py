"""
Scientific Integrity:
    - Loads the Calibrated Model (S029).
    - SKIPS the first 50% of data (Calibration set).
    - USES only the last 50% (Unseen set).
"""

import sys
import os
import torch
import numpy as np
import time
import turtle 
from torch.utils.data import DataLoader, Subset

# PATH SETUP
# 1. Get the folder this script is in: .../src/robot_interface
current_dir = os.path.dirname(os.path.abspath(__file__))
# 2. Get the 'src' folder
src_dir = os.path.dirname(current_dir)
# 3. Get the Project Root (BMI-Robotic-Control)
project_root = os.path.dirname(src_dir)

# 4. Add Project Root to Python Path so 'from src.config' works
if project_root not in sys.path:
    sys.path.append(project_root)

from src.config import PROJECT_ROOT
from src.train import BCIDataset
from src.models.eegnet import EEGNet
from src.models.atcnet import ATCNet
from src.models.spectrogram_cnn import SpectrogramCNN

# DEMO SETTINGS
TARGET_SUBJECT = 29
MODEL_TYPE = 'eegnet'
CONFIDENCE_THRESHOLD = 0.70    
STABILITY_COUNT = 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = PROJECT_ROOT / "results" / "calibrated_models" / f"S{TARGET_SUBJECT:03d}_{MODEL_TYPE}_calibrated.pth"

# ROBOT CONTROLLER LOGIC
class RobotController:
    def __init__(self):
        self.buffer = []
        self.last_command = "STOP"

    def process_prediction(self, probability, predicted_class):
        # 1. Safety Check
        if probability < CONFIDENCE_THRESHOLD:
            self.buffer = [] 
            return "HOLD"

        # 2. Add to buffer
        command = "LEFT" if predicted_class == 0 else "RIGHT"
        self.buffer.append(command)

        # Keep buffer small
        if len(self.buffer) > STABILITY_COUNT:
            self.buffer.pop(0)

        # 3. Stability Check (Debouncing)
        if len(self.buffer) == STABILITY_COUNT and all(x == command for x in self.buffer):
            return command
        
        return "HOLD"

# Visualisation
def setup_visuals():
    screen = turtle.Screen()
    screen.title(f"BCI Robot Control | Subject S{TARGET_SUBJECT:03d} | Model: {MODEL_TYPE.upper()}")
    screen.bgcolor("black")
    screen.setup(width=800, height=600)
    
    # Arrows
    robot = turtle.Turtle()
    robot.shape("arrow")
    robot.color("#00FF00") 
    robot.shapesize(4, 4, 4) 
    robot.speed(0)
    robot.setheading(90) # Face Up
    
    # UI Text Writer
    writer = turtle.Turtle()
    writer.hideturtle()
    writer.color("white")
    writer.penup()
    writer.goto(0, 200)
    
    return screen, robot, writer

def get_model_architecture():
    if MODEL_TYPE == 'eegnet': return EEGNet(dropoutRate=0.5, F1=8, D=2, kernLength=64)
    if MODEL_TYPE == 'atcnet': return ATCNet(dropout=0.5)
    if MODEL_TYPE == 'cnn': return SpectrogramCNN(dropout_rate=0.5)
    return None

def run_simulation():
    if not MODEL_PATH.exists():
        print(f"CRITICAL ERROR: No calibrated model found at: {MODEL_PATH}")
        print("Run final_transfer_learning.py first!")
        return

    # Load the specific calibrated brain
    print(f"Loading Brain: {MODEL_PATH.name}...")
    model = get_model_architecture().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH))
    model.eval()

    # Load Data (Force Unseen Split)
    mode = 'spectrogram' if MODEL_TYPE == 'cnn' else 'time_series'
    print(f"Loading Data for S{TARGET_SUBJECT:03d}...")
    ds = BCIDataset([TARGET_SUBJECT], mode=mode, augment=False)
    
    # SPLIT: Skip the first 50%
    total_len = len(ds)
    split_idx = int(total_len * 0.5)
    unseen_indices = list(range(split_idx, total_len))
    unseen_ds = Subset(ds, unseen_indices)
    
    print(f"Total Trials: {total_len} | Simulation Trials: {len(unseen_ds)} (Unseen)")

    loader = DataLoader(unseen_ds, batch_size=1, shuffle=False)

    # Setup Graphics
    try:
        screen, robot, writer = setup_visuals()
    except:
        print("Error: No display detected.")
        return

    controller = RobotController()
    
    print("\n--- STARTING SIMULATION ---")
    print("Press Ctrl+C in terminal to stop.\n")
    
    with torch.no_grad():
        for i, (X, y_true) in enumerate(loader):
            X = X.to(DEVICE)
            
            # AI Inference
            outputs = model(X)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred_cls = torch.max(probs, 1)
            
            conf_val = confidence.item()
            pred_val = pred_cls.item()
            true_val = y_true.item()
            
            # Robot Logic
            action = controller.process_prediction(conf_val, pred_val)
            
            # visual update
            target_text = "LEFT" if true_val == 0 else "RIGHT"
            
            # 1. Clear Screen
            writer.clear()
            
            # 2. Set Colors
            # If AI matches User -> Green. If Wrong -> Red. If Hold -> Yellow.
            if action == target_text:
                status_color = "#00FF00" # Green
                status_msg = "SUCCESS"
            elif action == "HOLD":
                status_color = "yellow"
                status_msg = "BUFFERING"
            else:
                status_color = "red"
                status_msg = "MISMATCH"

            # 3. Write Text
            display_text = (f"Trial: {i+1}/{len(unseen_ds)}\n\n"
                            f"USER INTENT: {target_text}\n"
                            f"AI PREDICTION: {action}\n"
                            f"CONFIDENCE: {conf_val:.1%}\n\n"
                            f"STATUS: {status_msg}")
            
            writer.color(status_color)
            writer.write(display_text, align="center", font=("Arial", 18, "bold"))
            
            # 4. Animate Robot
            if action == "LEFT":
                robot.setheading(180) # Face Left
                robot.forward(50)     # Move
                time.sleep(0.2)
                robot.backward(50)    # Return to center
            elif action == "RIGHT":
                robot.setheading(0)   # Face Right
                robot.forward(50)
                time.sleep(0.2)
                robot.backward(50)
            
            # Console Log
            print(f"Frame {i:02d} | User: {target_text} | Robot: {action} | Conf: {conf_val:.2f}")
            
            # Sleep 1.5 seconds so the video viewer can read the text
            time.sleep(1.5) 

    # End
    writer.clear()
    writer.color("white")
    writer.write("SIMULATION COMPLETE", align="center", font=("Arial", 24, "bold"))
    time.sleep(3)
    screen.bye()

if __name__ == "__main__":
    run_simulation()
"""
CONFIG.PY
=========
Central Configuration for the BMI Robotic Control Project.
Target Accuracy: ~75% (Reproducible)

Structure:
- PATHS: Input/Output directories.
- EXPERIMENT: Subject ranges and Task definitions.
- PREPROCESSING: Filtering and ICA settings.
"""

from pathlib import Path

# FILE PATHS
# Using relative paths
# Assumes this file is in 'src/', so we go up two levels to find 'Datasets'.
PROJECT_ROOT = Path(__file__).parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "Datasets" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "Datasets" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# EXPERIMENT SETTINGS
SUBJECTS = range(1, 110) 

# PhysioNet Run Mapping (Standard for Motor Imagery)
# Runs 3, 7, 11 = Actual Movement (Fists/Feet)
# Runs 4, 8, 12 = Imagined Movement (Fists/Feet)
TASKS = {
    'imagined_movement': [4, 8, 12],
    'actual_movement': [3, 7, 11]
}

# PREPROCESSING CONFIG
# These settings produced the clean data for your 75% model.
CONFIG_PREPROC = {
    # Resampling: Standardize to 160Hz for EEGNet
    "target_sfreq": 160,
    
    # Filtering: 1-40Hz captures Mu (8-12) and Beta (13-30) 
    # while removing DC drift and High Freq noise.
    "l_freq": 1.0,
    "h_freq": 40.0,
    "notch_freq": 50.0,  # Thailand Mains Frequency
    
    # ICA (Artifact Removal)
    "ica_method": "fastica",
    "ica_n_components": None, # 'None' lets MNE decide based on rank (usually ~64)
    "ica_max_iter": 1500,     # High iteration count for convergence
    
    # Artifact Thresholds (Your custom heuristics)
    "eog_threshold": 2.5,     # Aggressive Z-score for Blinks
    "muscle_threshold": 1.5,  # Ratio for Muscle Noise
    "max_exclude": 8          # Safety: Never remove more than 8 components
}

# LABEL MAPPING
# Standardizes T1 (Left) and T2 (Right) to integers
LABEL_MAP = {
    'T1': 0,  # Left Fist (Base Label)
    'T2': 1   # Right Fist (Base Label)
}
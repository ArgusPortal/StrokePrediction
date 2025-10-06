"""
Configuration module for Stroke Prediction v3
Defines all paths, constants, and global settings
"""

from pathlib import Path
import random
import numpy as np

# ========== SEED FOR REPRODUCIBILITY ==========
SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# ========== PROJECT PATHS ==========
BASE_DIR = Path(__file__).resolve().parent.parent  # StrokePrediction/

# Data directories
DATA_DIR = BASE_DIR / "data"
RAW_PATH = DATA_DIR / "raw"
INTERIM_PATH = DATA_DIR / "interim"  # ✅ ADICIONAR
PROC_PATH = DATA_DIR / "processed"   # ✅ ADICIONAR

# Model and results directories
MODELS_PATH = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results"

# Logs directory
LOGS_PATH = BASE_DIR / "logs"

# Create directories if they don't exist
for path in [RAW_PATH, INTERIM_PATH, PROC_PATH, MODELS_PATH, RESULTS_PATH, LOGS_PATH]:
    path.mkdir(parents=True, exist_ok=True)

# ========== MODEL CONFIGURATION ==========
MODEL_VERSION = "3.0.0"
MODEL_NAME = "stroke_prediction_v3"

# ========== FEATURE ENGINEERING SETTINGS ==========
BMI_BINS = [0, 18.5, 25, 30, 35, 100]
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese1', 'obese2']

AGE_BINS = [0, 30, 45, 60, 75, 100]
AGE_LABELS = ['young', 'adult', 'middle', 'senior', 'elderly']

GLUCOSE_BINS = [0, 100, 126, 200, 500]
GLUCOSE_LABELS = ['normal', 'prediabetic', 'diabetic', 'severe']

# ========== MODELING PARAMETERS ==========
TEST_SIZE = 0.20
VAL_SIZE = 0.1875  # 15% of total
CV_FOLDS = 10

# ========== THRESHOLD OPTIMIZATION ==========
DEFAULT_THRESHOLD = 0.15
THRESHOLD_RANGE = (0.01, 0.50)

# ========== FAIRNESS THRESHOLDS ==========
FAIRNESS_GAP_THRESHOLD = 0.10  # Max acceptable gap

# ========== DRIFT MONITORING ==========
PSI_MODERATE_THRESHOLD = 0.10
PSI_CRITICAL_THRESHOLD = 0.25
CONCEPT_DRIFT_PCT_THRESHOLD = 10.0  # 10% degradation

# ========== CALIBRATION TARGETS ==========
TARGET_ECE = 0.05  # Expected Calibration Error
TARGET_BSS = 0.10  # Brier Skill Score

# ========== LOGGING ==========
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# ========== EXPORT ALL CONFIGS ==========
__all__ = [
    # Paths
    'BASE_DIR',
    'DATA_DIR',
    'RAW_PATH',
    'INTERIM_PATH',
    'PROC_PATH',
    'MODELS_PATH',
    'RESULTS_PATH',
    'LOGS_PATH',
    
    # Global settings
    'SEED',
    'MODEL_VERSION',
    'MODEL_NAME',
    
    # Feature engineering
    'BMI_BINS',
    'BMI_LABELS',
    'AGE_BINS',
    'AGE_LABELS',
    'GLUCOSE_BINS',
    'GLUCOSE_LABELS',
    
    # Modeling
    'TEST_SIZE',
    'VAL_SIZE',
    'CV_FOLDS',
    
    # Thresholds
    'DEFAULT_THRESHOLD',
    'THRESHOLD_RANGE',
    'FAIRNESS_GAP_THRESHOLD',
    
    # Drift monitoring
    'PSI_MODERATE_THRESHOLD',
    'PSI_CRITICAL_THRESHOLD',
    'CONCEPT_DRIFT_PCT_THRESHOLD',
    
    # Calibration
    'TARGET_ECE',
    'TARGET_BSS',
    
    # Logging
    'LOG_LEVEL',
    'LOG_FORMAT'
]

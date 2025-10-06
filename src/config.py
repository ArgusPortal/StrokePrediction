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

# ========== ADVANCED MODEL SUPPORT ==========
ADVANCED_LIBS = True

MODEL_REGISTRY = {
    'logistic_l2': {
        'type': 'linear',
        'description': 'Logistic Regression with L2 regularization and class balancing'
    },
    'gradient_boosting': {
        'type': 'ensemble',
        'description': 'Gradient Boosting with tuned depth and subsampling'
    },
    'random_forest': {
        'type': 'ensemble',
        'description': 'Random Forest with class balancing'
    },
    'xgboost': {
        'type': 'boosting',
        'description': 'XGBoost classifier with balanced settings'
    },
    'lightgbm': {
        'type': 'boosting',
        'description': 'LightGBM classifier with balanced weights'
    }
}

# ========== FEATURE ENGINEERING SETTINGS ==========
BMI_BINS = [0, 18.5, 25, 30, 35, 100]
BMI_LABELS = ['underweight', 'normal', 'overweight', 'obese1', 'obese2']

AGE_BINS = [0, 30, 45, 60, 75, 100]
AGE_LABELS = ['young', 'adult', 'middle', 'senior', 'elderly']

GLUCOSE_BINS = [0, 100, 126, 200, 500]
GLUCOSE_LABELS = ['normal', 'prediabetic', 'diabetic', 'severe']

# Aggregate feature settings for downstream modules
FEATURE_CONFIG = {
    'age_bins': AGE_BINS,
    'age_labels': AGE_LABELS,
    'bmi_bins': BMI_BINS,
    'bmi_labels': BMI_LABELS,
    'glucose_bins': GLUCOSE_BINS,
    'glucose_labels': GLUCOSE_LABELS
}

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

# ========== VISUALIZATION CONFIGURATION ==========
VIZ_CONFIG = {
    'style': 'seaborn-v0_8-darkgrid',
    'color_palette': ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c', '#e67e22'],
    'figure_dpi': 300,
    'font_size': 10,
    'title_size': 14,
    'label_size': 12
}

# ========== OPTIMIZED MODEL PARAMETERS (v3.1) ==========
XGBOOST_OPTIMIZED = {
    'n_estimators': 500,  # Atualizado baseado em resultados
    'max_depth': 6,
    'learning_rate': 0.03,
    'subsample': 0.85,
    'colsample_bytree': 0.85,
    'scale_pos_weight': 25,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}

SAMPLING_STRATEGY = 'smote_tomek'  # Melhor estratégia
OPTIMAL_THRESHOLD_METHOD = 'multi_objective'  # vs 'fixed'

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
    'ADVANCED_LIBS',
    'MODEL_REGISTRY',
    
    # Feature engineering
    'BMI_BINS',
    'BMI_LABELS',
    'AGE_BINS',
    'AGE_LABELS',
    'GLUCOSE_BINS',
    'GLUCOSE_LABELS',
    'FEATURE_CONFIG',
    
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
    
    # Visualization
    'VIZ_CONFIG',
    
    # Optimized parameters
    'XGBOOST_OPTIMIZED',
    'SAMPLING_STRATEGY',
    'OPTIMAL_THRESHOLD_METHOD',
    
    # Logging
    'LOG_LEVEL',
    'LOG_FORMAT'
]

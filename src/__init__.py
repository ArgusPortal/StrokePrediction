"""
Stroke Prediction v3 - Módulos principais
"""

# === CONFIGURAÇÕES ===
from .config import (
    SEED,
    DATA_DIR,
    RAW_PATH,
    INTERIM_PATH,  # ✅ Já presente
    PROC_PATH,     # ✅ Já presente
    MODELS_PATH,
    RESULTS_PATH,
    BASE_DIR
)

# === FUNÇÕES PRINCIPAIS ===
from .data_loader import load_and_validate_data
from .feature_engineering import engineer_medical_features
from .preprocessing import create_preprocessing_pipeline
from .model_training import train_model_suite
from .evaluation import evaluate_model_comprehensive
from .fairness import analyze_fairness
from .calibration import analyze_calibration
from .drift_monitoring import monitor_drift
from .utils import save_model_with_metadata, load_model_with_metadata

__all__ = [
    # Configs
    'SEED',
    'DATA_DIR',
    'RAW_PATH',
    'INTERIM_PATH',
    'PROC_PATH',
    'MODELS_PATH',
    'RESULTS_PATH',
    'BASE_DIR',
    
    # Funções
    'load_and_validate_data',
    'engineer_medical_features',
    'create_preprocessing_pipeline',
    'train_model_suite',
    'evaluate_model_comprehensive',
    'analyze_fairness',
    'analyze_calibration',
    'monitor_drift',
    'save_model_with_metadata',
    'load_model_with_metadata'
]

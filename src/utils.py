"""
Utility functions
Helper functions for model persistence and general operations
"""

import logging
import json
import joblib
import hashlib
from datetime import datetime, UTC
from pathlib import Path
from .config import MODELS_PATH, VERSION

logger = logging.getLogger(__name__)


def save_model_with_metadata(model, model_name, metadata, output_dir=None):
    """
    Save model with complete metadata
    
    Args:
        model: Trained model
        model_name: Name for the model
        metadata: Dict with model metadata
        output_dir: Optional output directory
        
    Returns:
        tuple: (model_path, metadata_path)
    """
    
    if output_dir is None:
        output_dir = MODELS_PATH
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_filename = f"{model_name}_v{VERSION}.joblib"
    model_path = output_dir / model_filename
    
    joblib.dump(model, model_path, compress=3)
    file_size_mb = model_path.stat().st_size / (1024 * 1024)
    
    logger.info(f"Model saved: {model_filename} ({file_size_mb:.2f} MB)")
    
    # Prepare metadata
    full_metadata = {
        'model_info': {
            'name': model_name,
            'version': VERSION,
            'filename': model_filename,
            'file_size_mb': round(file_size_mb, 2),
            'creation_date': datetime.now(UTC).isoformat()
        },
        'performance': metadata.get('val_metrics', {}),
        'training': metadata.get('training_info', {})
    }
    
    # Save metadata
    metadata_filename = f"{model_name}_metadata_v{VERSION}.json"
    metadata_path = output_dir / metadata_filename
    
    with open(metadata_path, 'w') as f:
        json.dump(full_metadata, f, indent=2)
    
    logger.info(f"Metadata saved: {metadata_filename}")
    
    return model_path, metadata_path


def load_model_with_metadata(model_path):
    """
    Load model and its metadata
    
    Returns:
        tuple: (model, metadata)
    """
    
    model_path = Path(model_path)
    
    # Load model
    model = joblib.load(model_path)
    logger.info(f"Model loaded from: {model_path.name}")
    
    # Load metadata
    metadata_path = model_path.parent / model_path.name.replace('.joblib', '_metadata.json')
    
    if metadata_path.exists():
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        logger.info(f"Metadata loaded from: {metadata_path.name}")
    else:
        metadata = {}
        logger.warning("No metadata file found")
    
    return model, metadata


def calculate_data_hash(df):
    """Calculate hash of dataframe for integrity checking"""
    data_str = str(df.shape) + str(df.values.tobytes())
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

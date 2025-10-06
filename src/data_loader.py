"""
Data loading and validation module
Handles CSV loading with multiple fallbacks and integrity checks
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from .config import RAW_PATH, SEED
import hashlib

logger = logging.getLogger(__name__)

def calculate_checksum(df):
    """Calculate SHA256 checksum for data integrity"""
    data_str = str(df.shape) + str(df.values.tobytes())
    return hashlib.sha256(data_str.encode()).hexdigest()[:16]

def load_and_validate_data():
    """
    Loads stroke dataset with validation and checksums
    
    Returns:
        pd.DataFrame: Validated dataframe
        dict: Metadata with checksums and stats
    """
    
    logger.info("=" * 70)
    logger.info("Step 1: Data Loading & Validation")
    logger.info("-" * 70)
    
    # Candidate filenames in priority order
    candidates = [
        "healthcare-dataset-stroke-data.csv",
        "strokedata.csv", 
        "stroke.csv",
        "stroke_data.csv"
    ]
    
    df = None
    
    # Try each candidate
    for filename in candidates:
        filepath = RAW_PATH / filename
        if filepath.exists():
            logger.info(f"Found: {filename}")
            df = pd.read_csv(filepath)
            break
    
    # Auto-detect if not found
    if df is None:
        csvs = list(RAW_PATH.glob("*.csv"))
        if len(csvs) == 1:
            logger.info(f"Auto-detected: {csvs[0].name}")
            df = pd.read_csv(csvs[0])
        else:
            raise FileNotFoundError(
                f"No stroke dataset found in {RAW_PATH}\n"
                f"Expected one of: {candidates}\n"
                f"Found: {[c.name for c in csvs]}"
            )
    
    # === VALIDATION CHECKS ===
    
    logger.info("\nValidation checks:")
    
    # 1. Required columns
    required_cols = ['stroke', 'age', 'gender', 'hypertension', 
                     'heart_disease', 'bmi', 'avg_glucose_level']
    
    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    logger.info(f"  ✓ All required columns present")
    
    # 2. Target variable
    if not set(df['stroke'].dropna().unique()).issubset({0, 1}):
        raise ValueError("Target 'stroke' must be binary (0, 1)")
    logger.info(f"  ✓ Target variable valid")
    
    # 3. Class balance
    class_counts = df['stroke'].value_counts()
    imbalance_ratio = class_counts[0] / class_counts[1]
    logger.info(f"  ✓ Class distribution: {class_counts.to_dict()}")
    logger.info(f"  ✓ Imbalance ratio: {imbalance_ratio:.1f}:1")
    
    # 4. Data quality
    total_nulls = df.isnull().sum().sum()
    null_pct = (total_nulls / df.size) * 100
    logger.info(f"  ✓ Missing values: {total_nulls} ({null_pct:.2f}%)")
    
    # 5. Duplicates
    n_duplicates = df.duplicated().sum()
    if n_duplicates > 0:
        logger.warning(f"  ⚠️ Found {n_duplicates} duplicate rows - removing")
        df = df.drop_duplicates()
    
    # 6. Checksum
    checksum = calculate_checksum(df)
    logger.info(f"  ✓ Data checksum: {checksum}")
    
    # Metadata
    metadata = {
        'shape': df.shape,
        'prevalence': df['stroke'].mean(),
        'imbalance_ratio': float(imbalance_ratio),
        'missing_pct': float(null_pct),
        'checksum': checksum
    }
    
    logger.info(f"\n✅ Data loaded successfully: {df.shape}")
    logger.info(f"   Prevalence: {metadata['prevalence']:.3%}")
    
    return df, metadata

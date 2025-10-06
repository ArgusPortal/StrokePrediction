"""
Medical feature engineering module
Creates clinically-informed features based on domain knowledge
"""

import pandas as pd
import numpy as np
import logging
from .config import FEATURE_CONFIG

logger = logging.getLogger(__name__)

def engineer_medical_features(df):
    """
    Creates medically-informed features with deterministic binning
    
    CHANGES FROM V2:
    - Reduced age groups: 5 → 3 (fairness improvement)
    - Fixed bin edges (reproducibility)
    - Categorical encoding standardized
    
    Args:
        df: Input dataframe
        
    Returns:
        pd.DataFrame: Enhanced dataframe with new features
    """
    
    logger.info("\nStep 2: Medical Feature Engineering")
    logger.info("-" * 70)
    
    df = df.copy()
    
    # Clean initial data
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        logger.info("  Removed 'id' column")
    
    # Fix known data quality issues
    if 'work_type' in df.columns:
        df['work_type'] = df['work_type'].replace({'Govt_jov': 'Govt_job'})
    
    # === 1. HANDLE MISSING VALUES ===
    
    if 'bmi' in df.columns:
        bmi_missing = df['bmi'].isnull().sum()
        if bmi_missing > 0:
            # FIX: Usar .loc[] para evitar SettingWithCopyWarning
            df.loc[:, 'bmi'] = df['bmi'].fillna(df['bmi'].median())
            logger.info(f"  Imputed {bmi_missing} missing BMI values (median)")
    
    # === 2. CARDIOVASCULAR RISK SCORE ===
    
    df['cardio_risk_score'] = (
        df.get('hypertension', 0) * 2 +
        df.get('heart_disease', 0) * 3 +
        (df.get('age', 0) > 65).astype(int) * 2 +
        (df.get('avg_glucose_level', 0) > 140).astype(int)
    )
    
    # === 3. AGE FEATURES (REDUCED GROUPS FOR FAIRNESS) ===
    
    if 'age' in df.columns:
        df['age_group'] = pd.cut(
            df['age'], 
            bins=FEATURE_CONFIG['age_bins'],
            labels=FEATURE_CONFIG['age_labels'],
            include_lowest=True
        )
        
        df['is_elderly'] = (df['age'] > 65).astype(int)
        df['age_squared'] = df['age'] ** 2
        
        logger.info(f"  Created age groups: {FEATURE_CONFIG['age_labels']}")
    
    # === 4. BMI FEATURES ===
    
    if 'bmi' in df.columns:
        df['bmi_category'] = pd.cut(
            df['bmi'],
            bins=FEATURE_CONFIG['bmi_bins'],
            labels=FEATURE_CONFIG['bmi_labels'],
            include_lowest=True
        )
        
        df['bmi_risk'] = (df['bmi'] > 30).astype(int)
        df['bmi_extreme'] = (df['bmi'] > 35).astype(int)
        
        if 'age' in df.columns:
            df['bmi_age_interaction'] = df['bmi'] * df['age'] / 100
    
    # === 5. GLUCOSE FEATURES ===
    
    if 'avg_glucose_level' in df.columns:
        df['glucose_category'] = pd.cut(
            df['avg_glucose_level'],
            bins=FEATURE_CONFIG['glucose_bins'],
            labels=FEATURE_CONFIG['glucose_labels'],
            include_lowest=True
        )
        
        df['is_diabetic'] = (df['avg_glucose_level'] > 126).astype(int)
        df['is_prediabetic'] = (
            (df['avg_glucose_level'] >= 100) & 
            (df['avg_glucose_level'] < 126)
        ).astype(int)
        
        if 'age' in df.columns:
            df['glucose_age_risk'] = df['avg_glucose_level'] * df['age'] / 1000
    
    # === 6. SMOKING FEATURES ===
    
    if 'smoking_status' in df.columns:
        # Merge 'Unknown' with 'formerly smoked'
        df['smoking_status_clean'] = df['smoking_status'].replace({
            'Unknown': 'formerly smoked'
        })
        
        smoking_map = {
            'never smoked': 0, 
            'formerly smoked': 1, 
            'smokes': 2
        }
        df['smoking_risk'] = df['smoking_status_clean'].map(smoking_map).fillna(1)
        df['is_smoker'] = (df['smoking_status_clean'] == 'smokes').astype(int)
    
    # === 7. GENDER FEATURES ===
    
    if 'gender' in df.columns:
        df['gender_clean'] = df['gender'].replace({'Other': 'Female'})
        df['gender_risk'] = df['gender_clean'].map({'Female': 0, 'Male': 1}).fillna(0)
    
    # === 8. WORK STRESS ===
    
    if 'work_type' in df.columns:
        df['high_stress_job'] = df['work_type'].isin(['Private', 'Self-employed']).astype(int)
    
    # === 9. COMPOSITE RISK SCORE ===
    
    risk_components = [
        'cardio_risk_score', 'is_elderly', 'bmi_risk', 
        'is_diabetic', 'smoking_risk', 'gender_risk'
    ]
    available_components = [c for c in risk_components if c in df.columns]
    df['total_risk_score'] = df[available_components].sum(axis=1)
    
    # === 10. INTERACTION TERMS ===
    
    if 'age' in df.columns and 'hypertension' in df.columns:
        df['age_hypertension_int'] = df['age'] * df['hypertension']
    
    if 'bmi' in df.columns and 'avg_glucose_level' in df.columns:
        df['metabolic_syndrome'] = (
            (df['bmi'] > 30) & (df['avg_glucose_level'] > 100)
        ).astype(int)
    
    # === SUMMARY ===
    
    new_features = len([c for c in df.columns if any(
        kw in c for kw in ['risk', 'score', 'interaction', 'syndrome', 'category', 'group']
    )])
    
    logger.info(f"  ✓ Created {new_features} new features")
    logger.info(f"  ✓ Final shape: {df.shape}")
    
    return df

"""
Drift monitoring module
Detects data drift (PSI) and concept drift (performance degradation)
"""

import logging
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

logger = logging.getLogger(__name__)


def calculate_psi(expected, actual, bins=10):
    """
    Calculate Population Stability Index (PSI)
    
    PSI Interpretation:
    - <0.10: No significant change
    - 0.10-0.25: Moderate change
    - >0.25: Significant change (retrain)
    
    Returns:
        float: PSI value
    """
    
    expected = np.array(expected)[~np.isnan(expected)]
    actual = np.array(actual)[~np.isnan(actual)]
    
    if len(expected) == 0 or len(actual) == 0:
        return 0.0
    
    # Create bins from expected distribution
    breakpoints = np.quantile(expected, np.linspace(0, 1, bins + 1))
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 2:
        return 0.0
    
    # Bin both distributions
    expected_bins = np.digitize(expected, breakpoints[1:-1])
    actual_bins = np.digitize(actual, breakpoints[1:-1])
    
    # Calculate proportions
    expected_props = np.bincount(expected_bins, minlength=len(breakpoints)) / len(expected)
    actual_props = np.bincount(actual_bins, minlength=len(breakpoints)) / len(actual)
    
    # Add epsilon to avoid division by zero
    expected_props = np.where(expected_props == 0, 0.0001, expected_props)
    actual_props = np.where(actual_props == 0, 0.0001, actual_props)
    
    # Calculate PSI
    psi = np.sum((actual_props - expected_props) * np.log(actual_props / expected_props))
    
    return psi


def monitor_drift(X_baseline, X_current, feature_names=None, psi_threshold=0.25):
    """
    Monitor data drift across all features
    
    Returns:
        dict: Drift report with PSI per feature
    """
    
    logger.info("Monitoring data drift (PSI)...")
    
    if feature_names is None:
        feature_names = [f'feature_{i}' for i in range(X_baseline.shape[1])]
    
    drift_report = {}
    
    for i, feature in enumerate(feature_names):
        if i < X_baseline.shape[1] and i < X_current.shape[1]:
            psi = calculate_psi(X_baseline[:, i], X_current[:, i])
            
            status = 'stable' if psi < 0.10 else 'moderate' if psi < psi_threshold else 'critical'
            
            drift_report[feature] = {
                'psi': psi,
                'status': status
            }
            
            if status == 'critical':
                logger.warning(f"Feature '{feature}' has critical drift: PSI={psi:.4f}")
    
    # Summary
    critical_count = sum(1 for v in drift_report.values() if v['status'] == 'critical')
    moderate_count = sum(1 for v in drift_report.values() if v['status'] == 'moderate')
    
    logger.info(f"Drift summary: {critical_count} critical, {moderate_count} moderate")
    
    return drift_report


def detect_concept_drift(y_true_baseline, y_proba_baseline,
                         y_true_current, y_proba_current,
                         threshold_pct=10):
    """
    Detect concept drift by comparing performance metrics
    
    Returns:
        dict: Drift status for each metric
    """
    
    logger.info("Detecting concept drift...")
    
    # Calculate metrics for both periods
    metrics_baseline = {
        'pr_auc': average_precision_score(y_true_baseline, y_proba_baseline),
        'roc_auc': roc_auc_score(y_true_baseline, y_proba_baseline),
        'brier': brier_score_loss(y_true_baseline, y_proba_baseline)
    }
    
    metrics_current = {
        'pr_auc': average_precision_score(y_true_current, y_proba_current),
        'roc_auc': roc_auc_score(y_true_current, y_proba_current),
        'brier': brier_score_loss(y_true_current, y_proba_current)
    }
    
    # Calculate degradation
    concept_drift = {}
    
    for metric in ['pr_auc', 'roc_auc']:
        pct_change = ((metrics_baseline[metric] - metrics_current[metric]) / 
                      metrics_baseline[metric]) * 100
        
        concept_drift[metric] = {
            'baseline': metrics_baseline[metric],
            'current': metrics_current[metric],
            'pct_change': pct_change,
            'drift_detected': pct_change > threshold_pct
        }
    
    # Brier (lower is better)
    pct_change_brier = ((metrics_current['brier'] - metrics_baseline['brier']) / 
                        metrics_baseline['brier']) * 100
    
    concept_drift['brier'] = {
        'baseline': metrics_baseline['brier'],
        'current': metrics_current['brier'],
        'pct_change': pct_change_brier,
        'drift_detected': pct_change_brier > threshold_pct
    }
    
    # Check if any metric drifted
    any_drift = any(v['drift_detected'] for v in concept_drift.values())
    
    if any_drift:
        logger.warning("Concept drift detected - model retraining recommended")
    else:
        logger.info("No significant concept drift detected")
    
    return concept_drift

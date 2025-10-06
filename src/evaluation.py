"""
Model evaluation module
Comprehensive evaluation metrics and reporting
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    precision_recall_curve, roc_curve,
    brier_score_loss
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


def evaluate_model_comprehensive(model, X_test, y_test, threshold=0.5):
    """
    Comprehensive model evaluation
    
    Returns:
        dict: Complete evaluation metrics
    """
    
    logger.info("Performing comprehensive evaluation...")
    
    # Get predictions
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate metrics
    metrics = {
        'roc_auc': roc_auc_score(y_test, y_proba),
        'pr_auc': average_precision_score(y_test, y_proba),
        'brier_score': brier_score_loss(y_test, y_proba),
        'confusion_matrix': cm,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'f1_score': 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0,
        'threshold': threshold
    }
    
    # Calibration
    fraction_pos, mean_pred = calibration_curve(y_test, y_proba, n_bins=10)
    metrics['calibration_error'] = np.mean(np.abs(fraction_pos - mean_pred))
    
    # PR curve data
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    metrics['pr_curve'] = {
        'precision': precision,
        'recall': recall,
        'thresholds': pr_thresholds
    }
    
    # ROC curve data
    fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba)
    metrics['roc_curve'] = {
        'fpr': fpr,
        'tpr': tpr,
        'thresholds': roc_thresholds
    }
    
    logger.info(f"Evaluation complete:")
    logger.info(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC: {metrics['pr_auc']:.4f}")
    logger.info(f"  Recall: {metrics['recall']:.4f}")
    logger.info(f"  Precision: {metrics['precision']:.4f}")
    logger.info(f"  Calibration Error: {metrics['calibration_error']:.4f}")
    
    return metrics


def find_optimal_threshold(y_true, y_proba, target_recall=0.70):
    """
    Find optimal threshold for target recall
    
    Returns:
        dict: Optimal threshold and associated metrics
    """
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    
    # Find threshold that achieves target recall
    recall_mask = recall[:-1] >= target_recall
    
    if recall_mask.any():
        optimal_idx = np.where(recall_mask)[0][0]
        optimal_threshold = thresholds[optimal_idx]
        
        y_pred = (y_proba >= optimal_threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        return {
            'threshold': optimal_threshold,
            'recall': recall[optimal_idx],
            'precision': precision[optimal_idx],
            'f1_score': 2 * precision[optimal_idx] * recall[optimal_idx] / 
                       (precision[optimal_idx] + recall[optimal_idx]),
            'true_positives': int(tp),
            'false_positives': int(fp),
            'nnc': (tp + fp) / tp if tp > 0 else float('inf')
        }
    else:
        logger.warning(f"Cannot achieve target recall {target_recall}")
        return None

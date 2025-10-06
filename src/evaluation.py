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
    brier_score_loss, fbeta_score, balanced_accuracy_score
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


def optimize_thresholds_multiobjective(
    y_true,
    y_proba,
    betas=(1.0, 2.0),
    min_precision=None,
    min_recall=None,
    max_threshold=None
):
    """
    Generate a threshold sweep and pick candidates via multiple F-beta objectives.
    
    Args:
        y_true (array-like): Ground-truth labels.
        y_proba (array-like): Predicted probabilities.
        betas (tuple[float]): F-beta scores to optimise (1.0 => F1, 2.0 => recall-focused).
        min_precision (float | None): Optional precision floor for candidate filtering.
        min_recall (float | None): Optional recall floor for candidate filtering.
    
    Returns:
        dict: {
            'grid': pd.DataFrame,
            'best_by_beta': dict[beta, dict],
            'constraints': {'min_precision': value, 'min_recall': value}
        }
    """
    
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    # Skip the first point (threshold -> inf) to align lengths
    thresholds = thresholds
    precision = precision[1:]
    recall = recall[1:]
    
    candidates = []
    for idx, threshold in enumerate(thresholds):
        prec = precision[idx]
        rec = recall[idx]
        
        if min_precision is not None and prec < min_precision:
            continue
        if min_recall is not None and rec < min_recall:
            continue
        if max_threshold is not None and threshold > max_threshold:
            continue
        
        y_pred = (y_proba >= threshold).astype(int)
        
        metrics = {
            'threshold': float(threshold),
            'precision': float(prec),
            'recall': float(rec),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'support_pos': int(y_pred.sum()),
            'support_total': int(y_pred.shape[0])
        }
        
        for beta in betas:
            metrics[f'fbeta_{beta:.1f}'] = fbeta_score(
                y_true, y_pred,
                beta=beta,
                zero_division=0
            )
        
        candidates.append(metrics)
    
    if not candidates:
        logger.warning("No thresholds satisfied the provided constraints.")
        return {
            'grid': pd.DataFrame(),
            'best_by_beta': {},
            'constraints': {'min_precision': min_precision, 'min_recall': min_recall}
        }
    
    grid_df = pd.DataFrame(candidates).sort_values('threshold').reset_index(drop=True)
    
    best_by_beta = {}
    for beta in betas:
        column = f'fbeta_{beta:.1f}'
        best_row = grid_df.sort_values(column, ascending=False).iloc[0]
        best_by_beta[beta] = best_row.to_dict()
    
    return {
        'grid': grid_df,
        'best_by_beta': best_by_beta,
        'constraints': {
            'min_precision': min_precision,
            'min_recall': min_recall,
            'max_threshold': max_threshold
        }
    }


def summarize_threshold_performance(y_true, y_proba, threshold: float):
    """
    Compute key metrics for a specific decision threshold.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    y_pred = (y_proba >= threshold).astype(int)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    
    return {
        'threshold': float(threshold),
        'confusion_matrix': cm,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp),
        'precision': precision,
        'recall': recall,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'f1_score': f1,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred)
    }


def summarize_threshold_grid(y_true, y_proba, thresholds):
    """
    Generate precision/recall summary for a list of thresholds.
    
    Returns:
        pd.DataFrame: metrics ordered by threshold.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)
    records = []
    
    for thr in thresholds:
        metrics = summarize_threshold_performance(y_true, y_proba, thr)
        records.append({
            'threshold': metrics['threshold'],
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1_score': metrics['f1_score'],
            'balanced_accuracy': metrics['balanced_accuracy'],
            'tp': metrics['true_positives'],
            'fp': metrics['false_positives'],
            'fn': metrics['false_negatives'],
            'tn': metrics['true_negatives']
        })
    
    df = pd.DataFrame(records).sort_values('threshold').reset_index(drop=True)
    return df

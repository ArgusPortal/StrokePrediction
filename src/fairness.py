"""
Fairness analysis module
Detects and mitigates algorithmic bias
"""

import logging
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, average_precision_score, confusion_matrix
)

logger = logging.getLogger(__name__)


def analyze_fairness(model, X, y, sensitive_attrs=['gender', 'Residence_type'], threshold=0.5):
    """
    Comprehensive fairness analysis
    
    Returns:
        dict: Fairness metrics by group
    """
    
    logger.info("Performing fairness analysis...")
    
    results = {}
    
    for attr in sensitive_attrs:
        if attr not in X.columns:
            logger.warning(f"Sensitive attribute '{attr}' not found")
            continue
        
        logger.info(f"Analyzing fairness for: {attr}")
        
        group_metrics = {}
        
        for group_value in X[attr].dropna().unique():
            mask = X[attr] == group_value
            
            if mask.sum() < 10:
                continue
            
            X_group = X[mask]
            y_group = y[mask]
            
            y_proba = model.predict_proba(X_group)[:, 1]
            y_pred = (y_proba >= threshold).astype(int)
            
            # Calculate metrics
            roc_auc = roc_auc_score(y_group, y_proba)
            pr_auc = average_precision_score(y_group, y_proba)
            
            cm = confusion_matrix(y_group, y_pred)
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            else:
                tpr = fpr = 0
            
            group_metrics[group_value] = {
                'n': mask.sum(),
                'TPR': tpr,
                'FPR': fpr,
                'ROC_AUC': roc_auc,
                'PR_AUC': pr_auc
            }
        
        # Calculate gaps
        if len(group_metrics) >= 2:
            tpr_values = [m['TPR'] for m in group_metrics.values()]
            fpr_values = [m['FPR'] for m in group_metrics.values()]
            pr_auc_values = [m['PR_AUC'] for m in group_metrics.values()]
            
            gaps = {
                'TPR_gap': max(tpr_values) - min(tpr_values),
                'FPR_gap': max(fpr_values) - min(fpr_values),
                'PR_AUC_gap': max(pr_auc_values) - min(pr_auc_values)
            }
        else:
            gaps = {'TPR_gap': 0, 'FPR_gap': 0, 'PR_AUC_gap': 0}
        
        results[attr] = {
            'metrics': group_metrics,
            'gaps': gaps
        }
        
        # Log results
        for metric, value in gaps.items():
            status = '✓ OK' if value < 0.10 else '⚠ ATTENTION' if value < 0.15 else '✗ CRITICAL'
            logger.info(f"  {metric}: {value:.4f} {status}")
    
    return results

# -*- coding: utf-8 -*-
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, balanced_accuracy_score

RESULTS_DIR = Path('results')
THRESH_FILE = RESULTS_DIR / 'threshold.json'

art = joblib.load('analysis_artifacts.joblib')
model = art.get('best_model_calibrated') or art['best_model']
X_val = art['X_val']
y_val = art['y_val'].astype(int)
X_test = art['X_test']
y_test = art['y_test'].astype(int)

threshold_payload = {
    'value': 0.5,
    'source': 'default',
    'rationale': 'not provided'
}
if THRESH_FILE.exists():
    threshold_payload.update(json.loads(THRESH_FILE.read_text(encoding='utf-8')))

threshold = float(threshold_payload['value'])
print(f"Using decision threshold: {threshold:.2f} (source={threshold_payload.get('source')})")

# Helper metrics

def true_positive_rate(y_true, y_pred):
    mask = y_true == 1
    denom = mask.sum()
    if denom == 0:
        return np.nan
    return (y_pred[mask] == 1).sum() / denom

def false_positive_rate(y_true, y_pred):
    mask = y_true == 0
    denom = mask.sum()
    if denom == 0:
        return np.nan
    return (y_pred[mask] == 1).sum() / denom

def positive_predictive_value(y_true, y_pred):
    mask = y_pred == 1
    denom = mask.sum()
    if denom == 0:
        return np.nan
    return (y_true[mask] == 1).sum() / denom

sensitive_attrs = ['gender', 'Residence_type', 'smoking_status', 'work_type', 'ever_married', 'is_elderly']

# Compute metrics on validation + test

def evaluate_split(name, X, y_true, proba):
    y_pred = (proba >= threshold).astype(int)
    print(f"\n=== Fairness metrics ({name}) ===")
    overall = {}
    records = []
    for attr in sensitive_attrs:
        feature = X[attr]
        metrics = MetricFrame(
            metrics={
                'TPR': true_positive_rate,
                'FPR': false_positive_rate,
                'PPV': positive_predictive_value,
                'selection_rate': selection_rate,
            },
            y_true=y_true,
            y_pred=y_pred,
            sensitive_features=feature
        )
        df = metrics.by_group.reset_index().rename(columns={'index': 'group'})
        tpr_gap = df['TPR'].max() - df['TPR'].min()
        valid_tpr = df['TPR'].dropna()
        if not valid_tpr.empty and valid_tpr.min() >= 0 and valid_tpr.max() < 1:
            odds = valid_tpr / (1 - valid_tpr)
            odds_ratio = odds.max() / odds.min() if odds.min() > 0 else np.inf
        else:
            odds_ratio = np.nan
        records.append({
            'attribute': attr,
            'TPR_gap': float(tpr_gap) if not np.isnan(tpr_gap) else None,
            'odds_ratio': float(odds_ratio) if not np.isnan(odds_ratio) else None,
            'metrics': df.to_dict(orient='records')
        })
    overall['groups'] = records
    overall['constraints_met'] = all((r['TPR_gap'] is not None) and (r['TPR_gap'] <= 0.10) for r in records)
    return overall

proba_val = model.predict_proba(X_val)[:, 1]
proba_test = model.predict_proba(X_test)[:, 1]

val_report = evaluate_split('validation', X_val, y_val, proba_val)
test_report = evaluate_split('test', X_test, y_test, proba_test)

if not test_report['constraints_met']:
    print("\n[ALERT] Constraints not met on test set. Residual gaps:")
    for entry in test_report['groups']:
        print(f"  - {entry['attribute']}: TPR-gap={entry['TPR_gap']:.3f} | odds_ratio={entry['odds_ratio']}")

# Mitigation plan using ThresholdOptimizer por atributo
print("\n=== Mitigation Plan (Equalized Odds, validation) ===")
mitigation_plan = {}
for attr in ["Residence_type", "smoking_status", "work_type", "is_elderly"]:
    feature = X_val[attr].astype("category")
    valid_values = [value for value in feature.unique() if y_val[feature == value].nunique() == 2]
    feature_valid = feature.where(feature.isin(valid_values), "__OUTROS__")
    if y_val[feature_valid == "__OUTROS__"].nunique() < 2:
        print(f"  Skipping {attr}: insuficientes classes após agregação.")
        continue
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
    )
    try:
        optimizer.fit(X_val, y_val, sensitive_features=feature_valid)
        thresholds_plan = optimizer.interpolated_thresholder_.thresholds_
        mitigation_plan[attr] = {
            group: {"threshold": info["threshold"]} for group, info in thresholds_plan.items()
        }
        print(f"  {attr}: {len(thresholds_plan)} thresholds sugeridos.")
    except ValueError as exc:
        print(f"  ThresholdOptimizer falhou para {attr}: {exc}")

plan_payload = {
    'threshold': threshold,
    'source': threshold_payload.get('source'),
    'rationale': threshold_payload.get('rationale'),
    'validation_report': val_report,
    'test_report': test_report,
    'mitigation_plan': mitigation_plan,
}

Path('results/fairness_audit.json').write_text(json.dumps(plan_payload, indent=2), encoding='utf-8')
print("\nFairness audit saved to results/fairness_audit.json")

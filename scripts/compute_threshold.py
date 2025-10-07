import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, balanced_accuracy_score, confusion_matrix
import joblib

RESULTS_DIR = Path('results')
RESULTS_DIR.mkdir(exist_ok=True)
THRESH_PATH = RESULTS_DIR / 'threshold.json'

art = joblib.load('analysis_artifacts.joblib')
model = art.get('best_model_calibrated') or art['best_model']
X_val = art['X_val']
y_val = art['y_val'].astype(int)
X_test = art['X_test']
y_test = art['y_test'].astype(int)

y_proba_val = model.predict_proba(X_val)[:, 1]
y_proba_test = model.predict_proba(X_test)[:, 1]

thresholds = np.round(np.arange(0.01, 0.50 + 1e-9, 0.01), 2)
records = []
for thr in thresholds:
    y_pred = (y_proba_val >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_val, y_pred, average='binary', zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_val, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    records.append({
        'threshold': float(thr),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'balanced_accuracy': float(bal_acc),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    })

sweep_df = pd.DataFrame(records)
constraints_mask = (sweep_df['recall'] >= 0.70) & (sweep_df['precision'] >= 0.15)
filtered = sweep_df[constraints_mask]
if not filtered.empty:
    # select by highest f1, then highest balanced accuracy, then lowest threshold
    best_row = filtered.sort_values(['f1', 'balanced_accuracy', 'threshold'], ascending=[False, False, True]).iloc[0]
    rationale = 'threshold selected maximizing F1 under precision>=0.15 and recall>=0.70'
else:
    # fallback lexicographic: max recall, then max precision, then min threshold
    best_row = sweep_df.sort_values(['recall', 'precision', 'threshold'], ascending=[False, False, True]).iloc[0]
    rationale = 'fallback: no threshold met precision>=0.15 & recall>=0.70; chose highest recall then precision'

best_threshold = float(best_row['threshold'])

# Evaluate on validation and test at selected threshold
def evaluate_split(y_true, y_prob, thr):
    y_pred = (y_prob >= thr).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='binary', zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'balanced_accuracy': float(bal_acc),
        'tp': int(tp),
        'fp': int(fp),
        'tn': int(tn),
        'fn': int(fn)
    }

val_metrics = evaluate_split(y_val, y_proba_val, best_threshold)
test_metrics = evaluate_split(y_test, y_proba_test, best_threshold)

payload = {
    'value': best_threshold,
    'source': 'validation_calibrated',
    'rationale': rationale,
    'validation_metrics': val_metrics,
    'test_metrics': test_metrics,
    'grid': sweep_df.to_dict(orient='records')
}

THRESH_PATH.write_text(json.dumps(payload, indent=2), encoding='utf-8')

print(f"Selected threshold: {best_threshold:.2f} ({rationale})")
print('Validation metrics:', val_metrics)
print('Test metrics:', test_metrics)
print(f'Threshold sweep saved to {THRESH_PATH}')

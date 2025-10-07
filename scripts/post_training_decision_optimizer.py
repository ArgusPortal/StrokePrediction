"""Post-training decision threshold optimizer.

This script freezes a single operating threshold based on calibrated
validation probabilities, respecting business constraints on precision
and recall. The selected threshold is then reported (without re-tuning)
on the hold-out test set.
"""

from __future__ import annotations

import json
from math import sqrt
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


PRECISION_MIN = 0.15
RECALL_MIN = 0.70
RESULTS_DIR = Path("results")
THRESH_JSON_PATH = RESULTS_DIR / "threshold.json"
THRESH_GRID_PATH = RESULTS_DIR / "threshold_grid_val.csv"
VAL_METRICS_PATH = RESULTS_DIR / "metrics_val.json"
TEST_METRICS_PATH = RESULTS_DIR / "metrics_test.json"
POLICY_TEXT = "maximize F1 under precision>=0.15 & recall>=0.70 (fallback if needed)"


def load_artifacts(path: str = "analysis_artifacts.joblib") -> dict:
    return joblib.load(path)


def ensure_array(name: str, artifacts: dict, fallback_callable):
    arr = artifacts.get(name)
    if arr is None:
        arr = fallback_callable()
    return np.asarray(arr)


def compute_confusion_terms(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return int(tp), int(fp), int(fn), int(tn)


def sweep_thresholds(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns precision/recall with an extra leading element.
    thresholds_candidates = thresholds
    records = []
    support_total = int(len(y_true))

    for idx, thr in enumerate(thresholds_candidates):
        preds = (y_proba >= thr).astype(int)
        tp, fp, fn, tn = compute_confusion_terms(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, preds)
        records.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "balanced_accuracy": float(bal_acc),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support": support_total,
            }
        )

    # Include extreme operating points for completeness.
    for thr in (0.0, 1.0):
        preds = (y_proba >= thr).astype(int)
        tp, fp, fn, tn = compute_confusion_terms(y_true, preds)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, preds)
        records.append(
            {
                "threshold": float(thr),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
                "balanced_accuracy": float(bal_acc),
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "support": support_total,
            }
        )

    df = pd.DataFrame(records).drop_duplicates(subset=["threshold"]).sort_values("threshold").reset_index(drop=True)
    return df


def select_threshold(grid: pd.DataFrame) -> tuple[pd.Series, Optional[str]]:
    feasible = grid[
        (grid["precision"] >= PRECISION_MIN) &
        (grid["recall"] >= RECALL_MIN)
    ]

    if not feasible.empty:
        # Maximize F1, break ties with higher recall, then lower threshold.
        best = feasible.sort_values(
            ["f1", "recall", "threshold"],
            ascending=[False, False, True],
        ).iloc[0]
        return best, None

    # Fallback 1: lexicographic on recall within precision constraint.
    precision_ok = grid[grid["precision"] >= PRECISION_MIN]
    if not precision_ok.empty:
        best = precision_ok.sort_values(
            ["recall", "threshold"],
            ascending=[False, True],
        ).iloc[0]
        reason = "fallback_precision_constraint"
        return best, reason

    # Fallback 2: minimize distance to constraint box.
    def distance(row):
        prec_gap = max(0.0, PRECISION_MIN - row["precision"])
        rec_gap = max(0.0, RECALL_MIN - row["recall"])
        return sqrt(prec_gap ** 2 + rec_gap ** 2)

    best = grid.assign(_distance=grid.apply(distance, axis=1)).sort_values(
        ["_distance", "threshold"]
    ).iloc[0]
    best = best.drop(labels="_distance")
    reason = "fallback_distance_to_constraints"
    return best, reason


def evaluate_split(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> dict:
    preds = (y_proba >= threshold).astype(int)
    tp, fp, fn, tn = compute_confusion_terms(y_true, preds)
    return {
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(len(y_true)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()

    y_val = np.asarray(artifacts["y_val"]).astype(int)
    y_test = np.asarray(artifacts["y_test"]).astype(int)

    model_calibrated = artifacts.get("best_model_calibrated")
    if model_calibrated is None:
        raise ValueError("Calibrated model ('best_model_calibrated') not found in artifacts.")

    X_val = artifacts["X_val"]
    X_test = artifacts["X_test"]

    y_proba_val_cal = ensure_array(
        "y_proba_val_cal",
        artifacts,
        lambda: model_calibrated.predict_proba(X_val)[:, 1],
    )
    y_proba_test_cal = ensure_array(
        "y_proba_test_cal",
        artifacts,
        lambda: model_calibrated.predict_proba(X_test)[:, 1],
    )

    grid_df = sweep_thresholds(y_val, y_proba_val_cal)
    grid_df.to_csv(THRESH_GRID_PATH, index=False)

    selected_row, fallback_reason = select_threshold(grid_df)
    threshold_value = float(selected_row["threshold"])

    val_metrics = evaluate_split(y_val, y_proba_val_cal, threshold_value)
    test_metrics = evaluate_split(y_test, y_proba_test_cal, threshold_value)

    summary_payload = {
        "value": threshold_value,
        "source": "validation_calibrated",
        "policy": POLICY_TEXT,
    }
    if fallback_reason is not None:
        summary_payload["fallback_reason"] = fallback_reason
    THRESH_JSON_PATH.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    VAL_METRICS_PATH.write_text(json.dumps(val_metrics, indent=2), encoding="utf-8")
    TEST_METRICS_PATH.write_text(json.dumps(test_metrics, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()


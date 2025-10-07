"""Decision threshold agent operating on calibrated validation probabilities."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


THRESHOLD_MIN = 0.01
THRESHOLD_MAX = 0.99
THRESHOLD_STEP = 0.005
PRECISION_MIN = 0.15
RECALL_MIN = 0.70
RESULTS_DIR = Path("results")


def load_artifacts(path: str = "analysis_artifacts.joblib") -> dict:
    return joblib.load(path)


def ensure_probabilities(name: str, artifacts: dict, fallback_callable):
    values = artifacts.get(name)
    if values is None:
        values = fallback_callable()
    return np.asarray(values)


def build_threshold_grid(y_true: np.ndarray, y_proba: np.ndarray) -> pd.DataFrame:
    n_steps = int(round((THRESHOLD_MAX - THRESHOLD_MIN) / THRESHOLD_STEP)) + 1
    thresholds = np.linspace(THRESHOLD_MIN, THRESHOLD_MAX, n_steps)
    rows = []
    for thr in thresholds:
        preds = (y_proba >= thr).astype(int)
        rows.append(
            {
                "threshold": float(thr),
                "precision": float(precision_score(y_true, preds, zero_division=0)),
                "recall": float(recall_score(y_true, preds, zero_division=0)),
                "f1": float(f1_score(y_true, preds, zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def select_threshold(grid: pd.DataFrame) -> tuple[float, bool]:
    feasible = grid[
        (grid["precision"] >= PRECISION_MIN) &
        (grid["recall"] >= RECALL_MIN)
    ]

    if not feasible.empty:
        max_f1 = feasible["f1"].max()
        top = feasible[feasible["f1"] == max_f1]
        chosen = top.sort_values("threshold", ascending=False).iloc[0]
        return float(chosen["threshold"]), False

    max_f1 = grid["f1"].max()
    top = grid[grid["f1"] == max_f1]
    chosen = top.sort_values("threshold", ascending=False).iloc[0]
    return float(chosen["threshold"]), True


def evaluate_metrics(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> dict:
    preds = (y_proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": float(thr),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1": float(f1_score(y_true, preds, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, preds)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support": int(len(y_true)),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts()
    model_cal = artifacts.get("best_model_calibrated")
    if model_cal is None:
        raise ValueError("Calibrated model (isotonic) is required but missing.")

    X_val = artifacts["X_val"]
    X_test = artifacts["X_test"]
    y_val = np.asarray(artifacts["y_val"]).astype(int)
    y_test = np.asarray(artifacts["y_test"]).astype(int)

    y_proba_val_cal = ensure_probabilities(
        "y_proba_val_cal",
        artifacts,
        lambda: model_cal.predict_proba(X_val)[:, 1],
    )
    y_proba_test_cal = ensure_probabilities(
        "y_proba_test_cal",
        artifacts,
        lambda: model_cal.predict_proba(X_test)[:, 1],
    )

    grid = build_threshold_grid(y_val, y_proba_val_cal)
    threshold_value, constraints_unmet = select_threshold(grid)

    payload = {
        "threshold": threshold_value,
        "source": "validation_calibrated",
    }
    if constraints_unmet:
        payload["constraints_unmet"] = True
    (RESULTS_DIR / "threshold.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    val_metrics = evaluate_metrics(y_val, y_proba_val_cal, threshold_value)
    test_metrics = evaluate_metrics(y_test, y_proba_test_cal, threshold_value)

    pd.DataFrame([val_metrics]).to_csv(RESULTS_DIR / "metrics_threshold_val.csv", index=False)
    pd.DataFrame([test_metrics]).to_csv(RESULTS_DIR / "metrics_threshold_test.csv", index=False)

    print("Threshold selected (validation calibrated):", threshold_value)
    if constraints_unmet:
        print("Constraints unmet: precision>=0.15 & recall>=0.70 not satisfied.")
    print("Validation metrics:", val_metrics)
    print("Test metrics:", test_metrics)


if __name__ == "__main__":
    main()


"""Abstention zone analysis for calibrated triage model.

Computes expected metrics when deferring decisions in a probability band.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

ABSTENTION_LOW = 0.07
ABSTENTION_HIGH = 0.10


def load_artifacts():
    return joblib.load("analysis_artifacts.joblib")


def split_counts(y_true: np.ndarray, y_proba: np.ndarray, low: float, high: float) -> Tuple[int, int, int, int]:
    mask_mid = (y_proba >= low) & (y_proba <= high)
    mid_total = int(mask_mid.sum())
    mid_pos = int(y_true[mask_mid].sum())

    mask_low = y_proba < low
    mask_high = y_proba > high

    low_total = int(mask_low.sum())
    high_total = int(mask_high.sum())
    low_pos = int(y_true[mask_low].sum())
    high_pos = int(y_true[mask_high].sum())

    return (mid_total, mid_pos, low_total, low_pos, high_total, high_pos)


def summarize_split(name: str, y_true: np.ndarray, y_proba: np.ndarray) -> dict:
    mid_total, mid_pos, low_total, low_pos, high_total, high_pos = split_counts(
        y_true, y_proba, ABSTENTION_LOW, ABSTENTION_HIGH
    )
    total = len(y_true)
    total_pos = int(y_true.sum())
    coverage = (low_total + high_total) / total if total else 0.0
    expected_ppv = high_pos / high_total if high_total else float("nan")
    expected_tpr = high_pos / total_pos if total_pos else float("nan")
    return {
        "split": name,
        "total": total,
        "total_positives": total_pos,
        "alert_low_total": low_total,
        "alert_low_positives": low_pos,
        "alert_high_total": high_total,
        "alert_high_positives": high_pos,
        "abstained_total": mid_total,
        "abstained_positives": mid_pos,
        "abstention_rate": mid_total / total if total else 0.0,
        "coverage_after_abstain": coverage,
        "expected_ppv_post_abstain": expected_ppv,
        "expected_tpr_post_abstain": expected_tpr,
    }


def main():
    art = load_artifacts()
    model = art["best_model_calibrated"]
    X_val = pd.DataFrame(art["X_val"])
    y_val = np.asarray(art["y_val"]).astype(int)
    X_test = pd.DataFrame(art["X_test"])
    y_test = np.asarray(art["y_test"]).astype(int)
    y_proba_val = art.get("y_proba_val_cal")
    if y_proba_val is None:
        y_proba_val = model.predict_proba(X_val)[:, 1]
    y_proba_test = art.get("y_proba_test_cal")
    if y_proba_test is None:
        y_proba_test = model.predict_proba(X_test)[:, 1]

    summary = [
        summarize_split("validation", y_val, y_proba_val),
        summarize_split("test", y_test, y_proba_test),
    ]
    df = pd.DataFrame(summary)
    (RESULTS_DIR / "abstention_summary.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    print(df)


if __name__ == "__main__":
    main()


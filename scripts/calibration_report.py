"""Calibration evaluation script.

Computes ECE, Brier Score, and Brier Skill Score on validation and test sets,
then saves a reliability diagram for visual inspection.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss


DEFAULT_BINS = 10
RESULTS_PATH = Path("results")
PLOT_PATH = RESULTS_PATH / "calibration_reliability.png"


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = DEFAULT_BINS
) -> float:
    """Compute Expected Calibration Error (ECE) with equal-width bins."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    total = y_true.size
    ece = 0.0

    for i in range(n_bins):
        left = bins[i]
        right = bins[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= left) & (y_prob <= right)
        else:
            mask = (y_prob >= left) & (y_prob < right)
        bin_count = mask.sum()
        if bin_count == 0:
            continue
        avg_conf = y_prob[mask].mean()
        avg_acc = y_true[mask].mean()
        ece += (bin_count / total) * abs(avg_conf - avg_acc)

    return ece


def brier_skill_score(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float, float]:
    """Return (brier_score, baseline_brier, bss)."""
    brier = brier_score_loss(y_true, y_prob)
    baseline_prob = y_true.mean()
    baseline_brier = np.mean((y_true - baseline_prob) ** 2)
    if math.isclose(baseline_brier, 0.0):
        bss = 0.0
    else:
        bss = 1.0 - (brier / baseline_brier)
    return brier, baseline_brier, bss


def load_artifacts(path: str = "analysis_artifacts.joblib") -> Dict[str, object]:
    return joblib.load(path)


def ensure_results_dir():
    RESULTS_PATH.mkdir(parents=True, exist_ok=True)


def extract_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Return positive class probabilities."""
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)
        # Handle binary classification outputs
        if probs.ndim == 2 and probs.shape[1] == 2:
            return probs[:, 1]
        return probs.ravel()
    raise AttributeError("Model does not support predict_proba.")


def build_metrics_table(metrics: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    return (
        pd.DataFrame(metrics)
        .T[["ECE", "Brier", "BSS"]]
        .rename_axis("Split")
        .reset_index()
    )


def main():
    ensure_results_dir()
    artifacts = load_artifacts()

    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]
    x_val = artifacts["X_val"]
    y_val = artifacts["y_val"]
    x_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    prob_val = extract_probabilities(model, x_val)
    prob_test = extract_probabilities(model, x_test)

    metrics = {}
    plot_data = []

    for split_name, y_true, y_prob in [
        ("Validation", y_val, prob_val),
        ("Test", y_test, prob_test),
    ]:
        ece = expected_calibration_error(y_true, y_prob)
        brier, baseline_brier, bss = brier_skill_score(y_true, y_prob)
        metrics[split_name] = {
            "ECE": ece,
            "Brier": brier,
            "BaselineBrier": baseline_brier,
            "BSS": bss,
        }
        frac_pos, mean_pred = calibration_curve(
            y_true, y_prob, n_bins=DEFAULT_BINS, strategy="uniform"
        )
        plot_data.append((split_name, frac_pos, mean_pred))

    # Save metrics table
    table = build_metrics_table(metrics)
    table_path = RESULTS_PATH / "calibration_metrics.csv"
    table.to_csv(table_path, index=False)

    # Reliability diagram
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Ideal")
    for split_name, frac_pos, mean_pred in plot_data:
        ax.plot(mean_pred, frac_pos, marker="o", label=split_name)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(PLOT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print("Calibration metrics (ECE, Brier, BSS):")
    print(table.to_string(index=False))
    print(f"\nSaved reliability diagram to: {PLOT_PATH}")
    print(f"Metrics table saved to: {table_path}")


if __name__ == "__main__":
    main()

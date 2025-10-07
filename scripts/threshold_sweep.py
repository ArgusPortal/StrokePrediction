"""Threshold sweep for decision optimization.

Varrermos thresholds em validação, aplicando restrições de negócio
e reportamos métricas no teste usando o limiar congelado.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


THRESHOLD_START = 0.01
THRESHOLD_STOP = 0.50
THRESHOLD_STEP = 0.01
PRECISION_MIN = 0.15
RECALL_MIN = 0.70
RESULTS_DIR = Path("results")


def load_artifacts(path: str = "analysis_artifacts.joblib"):
    return joblib.load(path)


def extract_probs(model, X):
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    raise AttributeError("Model missing predict_proba.")


def sweep_thresholds(y_true, y_prob) -> pd.DataFrame:
    thresholds = np.arange(THRESHOLD_START, THRESHOLD_STOP + THRESHOLD_STEP / 2, THRESHOLD_STEP)
    rows = []
    for thr in thresholds:
        preds = (y_prob >= thr).astype(int)
        precision = precision_score(y_true, preds, zero_division=0)
        recall = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        bal_acc = balanced_accuracy_score(y_true, preds)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        rows.append(
            {
                "threshold": round(thr, 2),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "balanced_accuracy": bal_acc,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "tp": tp,
            }
        )
    df = pd.DataFrame(rows)
    return df


def select_threshold(df: pd.DataFrame) -> pd.Series | None:
    feasible = df[
        (df["precision"] >= PRECISION_MIN) &
        (df["recall"] >= RECALL_MIN)
    ]
    if feasible.empty:
        return None
    best_idx = feasible["f1"].idxmax()
    return feasible.loc[best_idx]


def evaluate_at_threshold(y_true, y_prob, threshold: float) -> dict:
    preds = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, preds, zero_division=0),
        "recall": recall_score(y_true, preds, zero_division=0),
        "f1": f1_score(y_true, preds, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]

    X_val = artifacts["X_val"]
    y_val = artifacts["y_val"]
    X_test = artifacts["X_test"]
    y_test = artifacts["y_test"]

    prob_val = extract_probs(model, X_val)
    prob_test = extract_probs(model, X_test)

    sweep_df = sweep_thresholds(y_val, prob_val)
    sweep_path = RESULTS_DIR / "threshold_sweep_validation.csv"
    sweep_df.to_csv(sweep_path, index=False)

    selected = select_threshold(sweep_df)
    if selected is None:
        print("Nenhum threshold atende às restrições (Recall ≥ 70%, Precision ≥ 15%).")
        print(f"Resultados completos salvos em: {sweep_path}")
        return

    chosen_thr = float(selected["threshold"])
    print("Threshold sweep (validação):")
    print(sweep_df.to_string(index=False, formatters={"threshold": "{:.2f}".format}))
    print(f"\nThreshold escolhido (validação): {chosen_thr:.2f}")
    print(
        f"Métricas no limiar escolhido -> Precision: {selected['precision']:.3f}, "
        f"Recall: {selected['recall']:.3f}, F1: {selected['f1']:.3f}, "
        f"Balanced Acc.: {selected['balanced_accuracy']:.3f}"
    )

    test_metrics = evaluate_at_threshold(y_test, prob_test, chosen_thr)
    print("\nRelato (teste) no limiar congelado:")
    print(
        f"Precision: {test_metrics['precision']:.3f}, "
        f"Recall: {test_metrics['recall']:.3f}, F1: {test_metrics['f1']:.3f}, "
        f"Balanced Acc.: {test_metrics['balanced_accuracy']:.3f}"
    )
    print(
        f"Matriz de confusão (teste): TN={test_metrics['tn']}, FP={test_metrics['fp']}, "
        f"FN={test_metrics['fn']}, TP={test_metrics['tp']}"
    )
    print(f"\nTabela do sweep salva em: {sweep_path}")


if __name__ == "__main__":
    main()

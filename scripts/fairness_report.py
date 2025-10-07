"""Fairness auditing script using Fairlearn MetricFrame."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate


RESULTS_DIR = Path("results")
METRICS_PATH = RESULTS_DIR / "fairness_metrics.csv"
INTERSECTION_PATH = RESULTS_DIR / "fairness_metrics_intersection.csv"

SENSITIVE_ATTRS = [
    "gender",
    "Residence_type",
    "smoking_status",
    "work_type",
    "ever_married",
    "is_elderly",
]

INTERSECTION_ATTRS = ["gender", "Residence_type", "smoking_status"]
MIN_GROUP_SIZE = 30
THRESHOLD = 0.06  # business-approved threshold
THRESHOLD_FILE = Path("results/threshold.json")


def load_artifacts(path: str = "analysis_artifacts.joblib"):
    return joblib.load(path)


def extract_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] == 2:
            return proba[:, 1]
        return proba.ravel()
    raise AttributeError("Model lacks predict_proba.")


def derive_predictions(proba: np.ndarray, threshold: float) -> np.ndarray:
    return (proba >= threshold).astype(int)


def true_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask_pos = y_true == 1
    total_pos = mask_pos.sum()
    if total_pos == 0:
        return np.nan
    return (y_pred[mask_pos] == 1).sum() / total_pos


def false_positive_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mask_neg = y_true == 0
    total_neg = mask_neg.sum()
    if total_neg == 0:
        return np.nan
    return (y_pred[mask_neg] == 1).sum() / total_neg


def precision_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    predicted_pos = y_pred == 1
    total_pred_pos = predicted_pos.sum()
    if total_pred_pos == 0:
        return np.nan
    return (y_true[predicted_pos] == 1).sum() / total_pred_pos


def metric_frame_for_attribute(
    y_true: pd.Series,
    y_pred: np.ndarray,
    sensitive_feature: pd.Series,
) -> MetricFrame:
    metrics = {
        "TPR": true_positive_rate,
        "FPR": false_positive_rate,
        "PPV": precision_score,
        "selection_rate": selection_rate,
    }
    return MetricFrame(
        metrics=metrics,
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_feature,
    )


def metricframe_to_table(
    name: str,
    mf: MetricFrame,
    counts: pd.Series,
    positive_counts: pd.Series,
) -> pd.DataFrame:
    per_group = mf.by_group.rename_axis("group").reset_index()
    per_group["n"] = per_group["group"].map(counts)
    per_group["n_pos"] = per_group["group"].map(positive_counts).fillna(0).astype(int)
    per_group["TPR_gap"] = np.nan
    per_group["FPR_gap"] = np.nan
    per_group["PPV_gap"] = np.nan
    per_group["selection_rate_gap"] = np.nan
    overall = pd.DataFrame(
        {
            "group": ["OVERALL"],
            "TPR": [mf.overall["TPR"]],
            "FPR": [mf.overall["FPR"]],
            "PPV": [mf.overall["PPV"]],
            "selection_rate": [mf.overall["selection_rate"]],
            "n": [counts.sum()],
            "n_pos": [positive_counts.sum()],
            "TPR_gap": [np.nan],
            "FPR_gap": [np.nan],
            "PPV_gap": [np.nan],
            "selection_rate_gap": [np.nan],
        }
    )
    table = pd.concat([per_group, overall], ignore_index=True)
    group_rows = table[table["group"] != "OVERALL"]

    def _gap(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return float("nan")
        return float(valid.max() - valid.min())

    gaps = {
        "TPR_gap": _gap(group_rows["TPR"]),
        "FPR_gap": _gap(group_rows["FPR"]),
        "PPV_gap": _gap(group_rows["PPV"]),
        "selection_rate_gap": _gap(group_rows["selection_rate"]),
    }
    gap_row = {
        "group": "GAP",
        "TPR": np.nan,
        "FPR": np.nan,
        "PPV": np.nan,
        "selection_rate": np.nan,
        "n": np.nan,
        "n_pos": np.nan,
        "TPR_gap": gaps["TPR_gap"],
        "FPR_gap": gaps["FPR_gap"],
        "PPV_gap": gaps["PPV_gap"],
        "selection_rate_gap": gaps["selection_rate_gap"],
    }
    table = pd.concat([table, pd.DataFrame([gap_row])], ignore_index=True)
    return table


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]

    X_val = artifacts["X_val"].copy()
    y_val = artifacts["y_val"]

    threshold = THRESHOLD
    if THRESHOLD_FILE.exists():
        try:
            payload = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
            threshold = float(payload.get("value", threshold))
        except Exception:
            pass
    print(f"Using threshold {threshold:.3f} for fairness report")

    prob_val = extract_probabilities(model, X_val)
    preds_val = derive_predictions(prob_val, threshold=threshold)

    tables = []
    alerts: list[str] = []

    for attr in SENSITIVE_ATTRS:
        counts = X_val[attr].value_counts()
        positive_counts = X_val.loc[y_val == 1, attr].value_counts()
        mf = metric_frame_for_attribute(y_val, preds_val, X_val[attr])
        table = metricframe_to_table(attr, mf, counts, positive_counts)
        table.insert(0, "attribute", attr)

        gap_value = table.loc[table["group"] == "GAP", "TPR_gap"].iloc[0]
        if gap_value >= 0.10:
            alerts.append(f"{attr}: TPR_gap={gap_value:.3f}")

        for _, row in table[(table["group"] != "GAP") & (table["group"] != "OVERALL")].iterrows():
            if row.get("n_pos", 0) < 5:
                alerts.append(f"{attr}/{row['group']}: apenas {int(row.get('n_pos', 0))} positivos")

        tables.append(table)

    all_tables = pd.concat(tables, ignore_index=True)
    all_tables.to_csv(METRICS_PATH, index=False)

    inter_col = X_val[INTERSECTION_ATTRS].astype(str).agg("|".join, axis=1)
    inter_counts = inter_col.value_counts()
    large_groups = inter_counts[inter_counts >= MIN_GROUP_SIZE]
    inter_feature = inter_col.where(inter_col.isin(large_groups.index), "__OTHER__")
    mf_inter = metric_frame_for_attribute(y_val, preds_val, inter_feature)
    counts_inter = inter_feature.value_counts()
    pos_inter = inter_feature[y_val == 1].value_counts()
    inter_table = metricframe_to_table("intersection", mf_inter, counts_inter, pos_inter)
    inter_table.insert(0, "attribute", "intersection")
    inter_table.to_csv(INTERSECTION_PATH, index=False)

    print(f"Fairness metrics saved to {METRICS_PATH}")
    print(f"Intersectional metrics saved to {INTERSECTION_PATH}")
    if alerts:
        print("Fairness alerts:")
        for msg in alerts:
            print(f"  - {msg}")
    else:
        print("No fairness alerts triggered (TPR_gap < 0.10 and n_pos >= 5).")


if __name__ == "__main__":
    main()

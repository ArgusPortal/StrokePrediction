"""Heuristic group threshold adjustment for fairness with business constraints."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fairlearn.metrics import MetricFrame, selection_rate
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

RESULTS_DIR = Path("results")
THRESHOLD_TABLE_PATH = RESULTS_DIR / "attribute_thresholds.csv"
CLASS_REPORT_PATH = RESULTS_DIR / "attribute_thresholds_metrics.csv"
FAIRNESS_VAL_PATH = RESULTS_DIR / "fairness_post_val.csv"
FAIRNESS_TEST_PATH = RESULTS_DIR / "fairness_post_test.csv"
THRESHOLD_FILE = RESULTS_DIR / "threshold.json"

DEFAULT_THRESHOLD = 0.06
ACTIVE_BASE_THRESHOLD = DEFAULT_THRESHOLD
THRESHOLD_GRID = np.round(np.arange(0.02, 0.51, 0.01), 2)
TARGET_UPPER = 0.80
TARGET_LOWER = 0.70
BUSINESS_PRECISION = 0.15
BUSINESS_RECALL = 0.70
MIN_GROUP_SIZE = 30
FAIRNESS_ATTRS = [
    "smoking_status",
    "work_type",
    "ever_married",
    "Residence_type",
    "gender",
    "is_elderly",
]
MIN_GAIN = 0.01  # minimum TPR-gap reduction to accept change


@dataclass
class Candidate:
    attribute: str
    level: Any
    count: int
    base_tpr: float
    proposed_threshold: float
    proposed_tpr: float
    direction: str  # 'raise' or 'lower'


def load_artifacts(path: str = "analysis_artifacts.joblib"):
    return joblib.load(path)


def extract_prob(model, X: pd.DataFrame) -> np.ndarray:
    proba = model.predict_proba(X)
    if proba.ndim == 2 and proba.shape[1] == 2:
        return proba[:, 1]
    return proba.ravel()


def tpr(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> float:
    mask = y_true == 1
    total = mask.sum()
    if total == 0:
        return np.nan
    return (scores[mask] >= threshold).sum() / total


def generate_threshold_candidates(
    y_true: np.ndarray,
    scores: np.ndarray,
) -> Tuple[float, List[Tuple[float, float, str]]]:
    """Return base TPR and candidate thresholds (threshold, new_tpr, direction)."""
    base = tpr(y_true, scores, ACTIVE_BASE_THRESHOLD)
    if np.isnan(base):
        return base, []

    proposals: List[Tuple[float, float, str]] = []

    if base > TARGET_UPPER:
        high_options = []
        for thr in THRESHOLD_GRID:
            if thr < ACTIVE_BASE_THRESHOLD:
                continue
            new_tpr = tpr(y_true, scores, thr)
            if np.isnan(new_tpr) or new_tpr > base:
                continue
            if new_tpr <= TARGET_UPPER:
                high_options.append((abs(TARGET_UPPER - new_tpr), thr, new_tpr))
        if high_options:
            _, thr, new_tpr = min(high_options, key=lambda item: (item[0], item[1]))
            proposals.append((float(thr), float(new_tpr), "raise"))

    if base < TARGET_LOWER:
        low_options = []
        for thr in THRESHOLD_GRID:
            if thr > ACTIVE_BASE_THRESHOLD:
                continue
            new_tpr = tpr(y_true, scores, thr)
            if np.isnan(new_tpr) or new_tpr < base:
                continue
            low_options.append((abs(TARGET_LOWER - new_tpr), -thr, thr, new_tpr))
        if low_options:
            _, _, thr, new_tpr = min(low_options, key=lambda item: (item[0], item[1]))
            proposals.append((float(thr), float(new_tpr), "lower"))

    return base, proposals


def build_threshold_arrays(df: pd.DataFrame, mappings: Dict[str, Dict[str, float]]) -> np.ndarray:
    values = np.full(len(df), ACTIVE_BASE_THRESHOLD, dtype=float)
    for attr, attr_map in mappings.items():
        if not attr_map:
            continue
        mapped = df[attr].map(attr_map).fillna(ACTIVE_BASE_THRESHOLD).to_numpy()
        values = np.maximum(values, mapped)
    return values


def summarize(y_true: np.ndarray, y_pred: np.ndarray, split: str) -> Dict[str, float]:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "split": split,
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def tpr_gap(y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series) -> float:
    metric = MetricFrame(
        metrics={"TPR": lambda y, p: recall_score(y, p, zero_division=0)},
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive,
    )
    series = metric.by_group["TPR"].dropna()
    if series.empty:
        return 0.0
    return float(series.max() - series.min())


def fairness_table(y_true: np.ndarray, y_pred: np.ndarray, sensitive: pd.Series, name: str) -> pd.DataFrame:
    def _tpr(y, p):
        positives = (y == 1)
        denom = positives.sum()
        if denom == 0:
            return float("nan")
        return (p[positives] == 1).sum() / denom

    def _fpr(y, p):
        mask = y == 0
        denom = mask.sum()
        if denom == 0:
            return np.nan
        return (p[mask] == 1).sum() / denom

    metrics = {
        "TPR": _tpr,
        "FPR": _fpr,
        "PPV": lambda y, p: precision_score(y, p, zero_division=0),
        "selection_rate": lambda y, p: selection_rate(y, p),
    }
    frame = MetricFrame(metrics=metrics, y_true=y_true, y_pred=y_pred, sensitive_features=sensitive)
    counts = sensitive.value_counts()
    positive_counts = sensitive[y_true == 1].value_counts()
    table = frame.by_group.rename_axis("group").reset_index()
    table["attribute"] = name
    table["n"] = table["group"].map(counts)
    table["n_pos"] = table["group"].map(positive_counts).fillna(0).astype(int)
    overall = pd.DataFrame(
        {
            "attribute": [name],
            "group": ["OVERALL"],
            "TPR": [frame.overall["TPR"]],
            "FPR": [frame.overall["FPR"]],
            "PPV": [frame.overall["PPV"]],
            "selection_rate": [frame.overall["selection_rate"]],
            "n": [len(y_true)],
            "n_pos": [int((y_true == 1).sum())],
        }
    )
    table = pd.concat([table, overall], ignore_index=True)
    subset = table[table["group"] != "OVERALL"]

    def _gap(series: pd.Series) -> float:
        valid = series.dropna()
        if valid.empty:
            return float("nan")
        return float(valid.max() - valid.min())

    gaps = {
        "TPR_gap": _gap(subset["TPR"]),
        "FPR_gap": _gap(subset["FPR"]),
        "PPV_gap": _gap(subset["PPV"]),
        "selection_rate_gap": _gap(subset["selection_rate"]),
    }
    gap_row = {"attribute": name, "group": "GAP", "n": np.nan, "n_pos": np.nan}
    gap_row.update(gaps)
    table = pd.concat([table, pd.DataFrame([gap_row])], ignore_index=True)
    return table


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    artifacts = load_artifacts()
    model = artifacts.get("best_model_calibrated") or artifacts["best_model"]

    X_val = artifacts["X_val"].copy()
    y_val = artifacts["y_val"].to_numpy()
    X_test = artifacts["X_test"].copy()
    y_test = artifacts["y_test"].to_numpy()

    scores_val = extract_prob(model, X_val)
    scores_test = extract_prob(model, X_test)

    current_threshold = DEFAULT_THRESHOLD
    if THRESHOLD_FILE.exists():
        try:
            payload = json.loads(THRESHOLD_FILE.read_text(encoding='utf-8'))
            current_threshold = float(payload.get('value', current_threshold))
        except Exception:
            pass
    global ACTIVE_BASE_THRESHOLD
    ACTIVE_BASE_THRESHOLD = current_threshold
    print(f"Using base threshold {ACTIVE_BASE_THRESHOLD:.3f} for group adjustments")

    base_preds_val = (scores_val >= ACTIVE_BASE_THRESHOLD).astype(int)
    base_gaps = {attr: tpr_gap(y_val, base_preds_val, X_val[attr]) for attr in FAIRNESS_ATTRS}

    candidates: List[Candidate] = []
    for attr in FAIRNESS_ATTRS:
        counts = X_val[attr].value_counts()
        for level, count in counts.items():
            if count < MIN_GROUP_SIZE:
                continue
            mask = X_val[attr] == level
            y_group = y_val[mask]
            if (y_group == 1).sum() == 0 or (y_group == 0).sum() == 0:
                continue
            base_tpr = tpr(y_group, scores_val[mask], ACTIVE_BASE_THRESHOLD)
            base_tpr, proposals = generate_threshold_candidates(y_group, scores_val[mask])
            for thr, new_tpr, direction in proposals:
                if direction == "raise" and thr <= ACTIVE_BASE_THRESHOLD:
                    continue
                if direction == "lower" and thr >= ACTIVE_BASE_THRESHOLD:
                    continue
                candidates.append(
                    Candidate(
                        attribute=attr,
                        level=str(level),
                        count=int(count),
                        base_tpr=float(base_tpr),
                        proposed_threshold=float(thr),
                        proposed_tpr=float(new_tpr),
                        direction=direction,
                    )
                )

    candidates.sort(key=lambda c: (-c.base_tpr, c.count))

    combo_val_series = X_val[FAIRNESS_ATTRS].astype(str).agg("|".join, axis=1)
    base_inter_gap = tpr_gap(y_val, base_preds_val, combo_val_series)
    best_mappings: Dict[str, Dict[str, float]] = {attr: {} for attr in FAIRNESS_ATTRS}
    best_preds_val = base_preds_val
    best_metrics = summarize(y_val, best_preds_val, "validation")
    best_attr_gaps = base_gaps.copy()
    best_inter_gap = base_inter_gap
    best_max_gap = max(list(best_attr_gaps.values()) + [best_inter_gap])

    n = len(candidates)
    for mask in range(1, 1 << n):
        trial_mappings: Dict[str, Dict[str, float]] = {attr: {} for attr in FAIRNESS_ATTRS}
        for idx in range(n):
            if mask & (1 << idx):
                cand = candidates[idx]
                trial_mappings[cand.attribute][cand.level] = cand.proposed_threshold

        trial_thresholds = build_threshold_arrays(X_val, trial_mappings)
        trial_preds = (scores_val >= trial_thresholds).astype(int)
        metrics = summarize(y_val, trial_preds, "validation")
        if metrics["precision"] < BUSINESS_PRECISION or metrics["recall"] < BUSINESS_RECALL:
            continue

        attr_gaps = {attr: tpr_gap(y_val, trial_preds, X_val[attr]) for attr in FAIRNESS_ATTRS}
        inter_gap = tpr_gap(y_val, trial_preds, combo_val_series)
        max_gap = max(list(attr_gaps.values()) + [inter_gap])

        improved = False
        if max_gap < best_max_gap - MIN_GAIN:
            improved = True
        elif max_gap <= 0.10 and best_max_gap > 0.10:
            improved = True
        elif abs(max_gap - best_max_gap) <= 1e-6:
            if metrics["precision"] > best_metrics["precision"] + 1e-3:
                improved = True
            elif (
                abs(metrics["precision"] - best_metrics["precision"]) <= 1e-3
                and metrics["recall"] > best_metrics["recall"]
            ):
                improved = True

        if improved:
            best_mappings = trial_mappings
            best_preds_val = trial_preds
            best_metrics = metrics
            best_attr_gaps = attr_gaps
            best_inter_gap = inter_gap
            best_max_gap = max_gap

    val_thresholds = build_threshold_arrays(X_val, best_mappings)
    test_thresholds = build_threshold_arrays(X_test, best_mappings)

    y_val_post = (scores_val >= val_thresholds).astype(int)
    y_test_post = (scores_test >= test_thresholds).astype(int)

    val_metrics = summarize(y_val, y_val_post, "validation")
    test_metrics = summarize(y_test, y_test_post, "test")

    if (
        val_metrics["precision"] < BUSINESS_PRECISION
        or val_metrics["recall"] < BUSINESS_RECALL
        or test_metrics["precision"] < BUSINESS_PRECISION
        or test_metrics["recall"] < BUSINESS_RECALL
    ):
        print(
            "Warning: group threshold search could not satisfy business constraints; "
            "retaining base threshold without adjustments."
        )
        return

    records = []
    for attr, mapping in best_mappings.items():
        for level, threshold in mapping.items():
            mask = X_val[attr] == level
            direction = "raise" if threshold > ACTIVE_BASE_THRESHOLD else "lower" if threshold < ACTIVE_BASE_THRESHOLD else "none"
            records.append(
                {
                    "attribute": attr,
                    "group": str(level),
                    "threshold": threshold,
                    "base_tpr": tpr(y_val[mask], scores_val[mask], ACTIVE_BASE_THRESHOLD),
                    "new_tpr": tpr(y_val[mask], scores_val[mask], threshold),
                    "count_val": int(mask.sum()),
                    "count_pos_val": int((y_val[mask] == 1).sum()),
                    "direction": direction,
                }
            )
    pd.DataFrame(records).to_csv(THRESHOLD_TABLE_PATH, index=False)
    pd.DataFrame([val_metrics, test_metrics]).to_csv(CLASS_REPORT_PATH, index=False)

    alerts: list[str] = []
    fairness_val_tables = []
    for attr in FAIRNESS_ATTRS:
        table = fairness_table(y_val, y_val_post, X_val[attr], attr)
        fairness_val_tables.append(table)
        gap_val = table.loc[table["group"] == "GAP", "TPR_gap"].iloc[0]
        if not np.isnan(gap_val) and gap_val >= 0.10:
            alerts.append(f"{attr}: TPR_gap={gap_val:.3f}")
        for _, row in table[(table["group"] != "OVERALL") & (table["group"] != "GAP")].iterrows():
            if row.get("n_pos", 0) < 5:
                alerts.append(f"{attr}/{row['group']}: apenas {int(row.get('n_pos', 0))} positivos")
    inter_table_val = fairness_table(y_val, y_val_post, combo_val_series, "intersection")
    fairness_val_tables.append(inter_table_val)
    pd.concat(fairness_val_tables, ignore_index=True).to_csv(FAIRNESS_VAL_PATH, index=False)

    fairness_test_tables = []
    combo_test = X_test[FAIRNESS_ATTRS].astype(str).agg("|".join, axis=1)
    for attr in FAIRNESS_ATTRS:
        fairness_test_tables.append(fairness_table(y_test, y_test_post, X_test[attr], attr))
    fairness_test_tables.append(fairness_table(y_test, y_test_post, combo_test, "intersection"))
    pd.concat(fairness_test_tables, ignore_index=True).to_csv(FAIRNESS_TEST_PATH, index=False)

    print(f"Threshold adjustments saved to {THRESHOLD_TABLE_PATH}")
    print(f"Classification metrics saved to {CLASS_REPORT_PATH}")
    print(f"Fairness tables saved to {FAIRNESS_VAL_PATH} and {FAIRNESS_TEST_PATH}")
    if alerts:
        print("Fairness alerts:")
        for msg in alerts:
            print(f"  - {msg}")


if __name__ == "__main__":
    main()



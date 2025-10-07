"""Fairness and decision audit using a frozen calibrated threshold.

This agent:
1. Loads the calibrated decision threshold selected on validation.
2. Evaluates global metrics on validation and test splits.
3. Produces pre-mitigation group fairness tables with bootstrap CIs.
4. Applies post-processing mitigation per attribute:
   - Stage 1: Equal Opportunity (true positive rate parity).
   - Stage 2: Equalized Odds (optional; attempted only with sufficient support).
5. Writes post-mitigation tables, JSON audit, and alerts.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fairlearn.postprocessing import ThresholdOptimizer
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


RESULTS_DIR = Path("results")
THRESHOLD_FILE = RESULTS_DIR / "threshold.json"
METRICS_VAL_PATH = RESULTS_DIR / "metrics_threshold_val.csv"
METRICS_TEST_PATH = RESULTS_DIR / "metrics_threshold_test.csv"
FAIRNESS_PRE_VAL_PATH = RESULTS_DIR / "fairness_pre_val.csv"
FAIRNESS_PRE_TEST_PATH = RESULTS_DIR / "fairness_pre_test.csv"
FAIRNESS_POST_VAL_PATH = RESULTS_DIR / "fairness_post_val.csv"
FAIRNESS_POST_TEST_PATH = RESULTS_DIR / "fairness_post_test.csv"
FAIRNESS_AUDIT_PATH = RESULTS_DIR / "fairness_audit.json"

TARGET_ATTRIBUTES = ["Residence_type", "gender", "smoking_status", "work_type", "is_elderly"]

EO_MIN_POS = 5
EO_MIN_NEG = 5
EOD_MIN_POS = 10
EOD_MIN_NEG = 10
TPR_GAP_ALERT_THRESHOLD = 0.10
BOOTSTRAP_ITERATIONS = 1000
BOOTSTRAP_RANDOM_STATE = 42


def load_artifacts(path: str = "analysis_artifacts.joblib") -> dict:
    return joblib.load(path)


def load_threshold() -> float:
    if not THRESHOLD_FILE.exists():
        raise FileNotFoundError("results/threshold.json is required.")
    payload = json.loads(THRESHOLD_FILE.read_text(encoding="utf-8"))
    threshold = payload.get("threshold", payload.get("value"))
    if threshold is None:
        raise KeyError("Threshold payload missing 'threshold' or 'value'.")
    source = payload.get("source")
    if source != "validation_calibrated":
        raise ValueError("Threshold source must be 'validation_calibrated'.")
    return float(threshold)


def ensure_probabilities(name: str, artifacts: dict, fallback_callable):
    values = artifacts.get(name)
    if values is None:
        values = fallback_callable()
    return np.asarray(values, dtype=float)


def global_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support": int(len(y_true)),
    }


def write_global_metrics(val_metrics: dict, test_metrics: dict) -> None:
    pd.DataFrame([val_metrics]).to_csv(METRICS_VAL_PATH, index=False)
    pd.DataFrame([test_metrics]).to_csv(METRICS_TEST_PATH, index=False)


def aggregate_groups(series: pd.Series, y_true: np.ndarray, min_pos: int, min_neg: int) -> Tuple[pd.Series, Dict[str, str]]:
    raw = series.astype(str)
    stats = (
        pd.DataFrame({"group": raw, "y": y_true})
        .groupby("group")["y"]
        .agg(total="count", positives="sum")
        .assign(negatives=lambda df: df["total"] - df["positives"])
    )
    mapping: Dict[str, str] = {}
    for group, row in stats.iterrows():
        if row["positives"] < min_pos or row["negatives"] < min_neg:
            mapping[group] = "__OTHER__"
        else:
            mapping[group] = group
    aggregated = raw.map(mapping).fillna("__OTHER__")
    return aggregated, mapping


def apply_mapping(series: pd.Series, mapping: Dict[str, str]) -> pd.Series:
    return series.astype(str).map(mapping).fillna("__OTHER__")


def compute_group_stats(y_true: np.ndarray, y_pred: np.ndarray, groups: pd.Series, attribute: str, dataset: str, stage: str) -> pd.DataFrame:
    rows: List[dict] = []
    groups_arr = groups.to_numpy()
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    unique_groups = pd.unique(groups_arr)
    for group in unique_groups:
        mask = groups_arr == group
        y_t = y_true_arr[mask]
        y_p = y_pred_arr[mask]
        n_total = len(y_t)
        n_pos = int(y_t.sum())
        n_neg = n_total - n_pos
        tp = int(((y_t == 1) & (y_p == 1)).sum())
        fp = int(((y_t == 0) & (y_p == 1)).sum())
        fn = int(((y_t == 1) & (y_p == 0)).sum())
        tn = int(((y_t == 0) & (y_p == 0)).sum())
        pred_pos = tp + fp
        selection = pred_pos / n_total if n_total > 0 else np.nan
        tpr = tp / n_pos if n_pos > 0 else np.nan
        fpr = fp / n_neg if n_neg > 0 else np.nan
        ppv = tp / pred_pos if pred_pos > 0 else np.nan

        rows.append(
            {
                "attribute": attribute,
                "group": group,
                "dataset": dataset,
                "stage": stage,
                "n_total": int(n_total),
                "n_pos": n_pos,
                "n_neg": n_neg,
                "tp": tp,
                "fp": int(fp),
                "fn": fn,
                "tn": int(tn),
                "pred_pos": int(pred_pos),
                "pred_neg": int(n_total - pred_pos),
                "selection_rate": float(selection) if not np.isnan(selection) else np.nan,
                "TPR": float(tpr) if not np.isnan(tpr) else np.nan,
                "FPR": float(fpr) if not np.isnan(fpr) else np.nan,
                "PPV": float(ppv) if not np.isnan(ppv) else np.nan,
            }
        )
    df = pd.DataFrame(rows)
    df.sort_values("group", inplace=True)
    return df


def bootstrap_tpr_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    group_labels: List[str],
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = BOOTSTRAP_RANDOM_STATE,
) -> Tuple[Dict[str, Tuple[float, float]], Tuple[float, float]]:
    rng = np.random.default_rng(random_state)
    n = len(y_true)
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)
    groups_arr = groups.to_numpy()

    samples: Dict[str, List[float]] = {g: [] for g in group_labels}
    gap_samples: List[float] = []

    for _ in range(iterations):
        idx = rng.integers(0, n, size=n)
        y_t = y_true_arr[idx]
        y_p = y_pred_arr[idx]
        g = groups_arr[idx]

        stats = compute_group_stats(y_t, y_p, pd.Series(g), attribute="tmp", dataset="tmp", stage="tmp")
        tprs = {row["group"]: row["TPR"] for _, row in stats.iterrows()}
        valid_tprs = [v for v in tprs.values() if not np.isnan(v)]
        if valid_tprs:
            gap_samples.append(float(max(valid_tprs) - min(valid_tprs)))
        for group in group_labels:
            value = tprs.get(group, np.nan)
            if not np.isnan(value):
                samples[group].append(float(value))

    def percentile_bounds(values: List[float]) -> Tuple[float, float]:
        if not values:
            return (float("nan"), float("nan"))
        arr = np.asarray(values)
        return float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5))

    group_ci = {group: percentile_bounds(vals) for group, vals in samples.items()}
    gap_ci = percentile_bounds(gap_samples)
    return group_ci, gap_ci


def build_fairness_table(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    attribute: str,
    dataset: str,
    stage: str,
) -> Tuple[pd.DataFrame, float, Tuple[float, float]]:
    stats_df = compute_group_stats(y_true, y_pred, groups, attribute, dataset, stage)
    group_labels = stats_df["group"].tolist()
    group_ci, gap_ci = bootstrap_tpr_ci(y_true, y_pred, groups, group_labels)

    stats_df["tpr_ci_low"] = stats_df["group"].map(lambda g: group_ci.get(g, (np.nan, np.nan))[0])
    stats_df["tpr_ci_high"] = stats_df["group"].map(lambda g: group_ci.get(g, (np.nan, np.nan))[1])
    stats_df["tpr_gap"] = np.nan
    stats_df["tpr_gap_ci_low"] = np.nan
    stats_df["tpr_gap_ci_high"] = np.nan

    valid_tprs = stats_df["TPR"].dropna()
    if valid_tprs.empty:
        gap_value = float("nan")
    else:
        gap_value = float(valid_tprs.max() - valid_tprs.min())

    gap_row = {
        "attribute": attribute,
        "group": "__GAP__",
        "dataset": dataset,
        "stage": stage,
        "n_total": np.nan,
        "n_pos": np.nan,
        "n_neg": np.nan,
        "tp": np.nan,
        "fp": np.nan,
        "fn": np.nan,
        "tn": np.nan,
        "pred_pos": np.nan,
        "pred_neg": np.nan,
        "selection_rate": np.nan,
        "TPR": np.nan,
        "FPR": np.nan,
        "PPV": np.nan,
        "tpr_ci_low": np.nan,
        "tpr_ci_high": np.nan,
        "tpr_gap": gap_value,
        "tpr_gap_ci_low": gap_ci[0],
        "tpr_gap_ci_high": gap_ci[1],
    }

    stats_with_gap = pd.concat([stats_df, pd.DataFrame([gap_row])], ignore_index=True)
    return stats_with_gap, gap_value, gap_ci


@dataclass
class MitigationResult:
    stage: str
    val_table: pd.DataFrame
    test_table: pd.DataFrame
    val_gap: float
    val_gap_ci: Tuple[float, float]
    test_gap: float
    test_gap_ci: Tuple[float, float]
    history: List[Dict[str, object]] = field(default_factory=list)


def attempt_equal_opportunity(
    model,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    groups_val: pd.Series,
    groups_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="true_positive_rate_parity",
        objective="balanced_accuracy_score",
        prefit=True,
    )
    optimizer.fit(X_val, y_val, sensitive_features=groups_val)
    pred_val = optimizer.predict(X_val, sensitive_features=groups_val).astype(int)
    pred_test = optimizer.predict(X_test, sensitive_features=groups_test).astype(int)
    return pred_val, pred_test, {
        "constraint": "true_positive_rate_parity",
        "status": "applied",
    }


def attempt_equalized_odds(
    model,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    y_val: np.ndarray,
    y_test: np.ndarray,
    groups_val: pd.Series,
    groups_test: pd.Series,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, object]]:
    optimizer = ThresholdOptimizer(
        estimator=model,
        constraints="equalized_odds",
        objective="balanced_accuracy_score",
        prefit=True,
    )
    optimizer.fit(X_val, y_val, sensitive_features=groups_val)
    pred_val = optimizer.predict(X_val, sensitive_features=groups_val).astype(int)
    pred_test = optimizer.predict(X_test, sensitive_features=groups_test).astype(int)
    return pred_val, pred_test, {
        "constraint": "equalized_odds",
        "status": "applied",
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    artifacts = load_artifacts()
    model = artifacts.get("best_model_calibrated")
    if model is None:
        raise ValueError("Calibrated model ('best_model_calibrated') not found in artifacts.")

    threshold = load_threshold()

    X_val = artifacts["X_val"]
    X_test = artifacts["X_test"]
    y_val = np.asarray(artifacts["y_val"]).astype(int)
    y_test = np.asarray(artifacts["y_test"]).astype(int)

    y_proba_val = ensure_probabilities(
        "y_proba_val_cal",
        artifacts,
        lambda: model.predict_proba(X_val)[:, 1],
    )
    y_proba_test = ensure_probabilities(
        "y_proba_test_cal",
        artifacts,
        lambda: model.predict_proba(X_test)[:, 1],
    )

    y_pred_val_base = (y_proba_val >= threshold).astype(int)
    y_pred_test_base = (y_proba_test >= threshold).astype(int)

    val_metrics = global_metrics(y_val, y_proba_val, threshold)
    test_metrics = global_metrics(y_test, y_proba_test, threshold)
    write_global_metrics(val_metrics, test_metrics)

    pre_val_tables: List[pd.DataFrame] = []
    pre_test_tables: List[pd.DataFrame] = []
    post_val_tables: List[pd.DataFrame] = []
    post_test_tables: List[pd.DataFrame] = []
    audit_payload: Dict[str, object] = {
        "threshold": threshold,
        "threshold_source": "validation_calibrated",
        "attributes": [],
        "alerts": [],
    }

    alerts: List[str] = []

    for attribute in TARGET_ATTRIBUTES:
        series_val = X_val[attribute]
        groups_val, mapping = aggregate_groups(series_val, y_val, EO_MIN_POS, EO_MIN_NEG)
        groups_test = apply_mapping(X_test[attribute], mapping)

        baseline_val_table, baseline_val_gap, baseline_val_gap_ci = build_fairness_table(
            y_val, y_pred_val_base, groups_val, attribute, "validation", "baseline"
        )
        baseline_test_table, baseline_test_gap, baseline_test_gap_ci = build_fairness_table(
            y_test, y_pred_test_base, groups_test, attribute, "test", "baseline"
        )

        pre_val_tables.append(baseline_val_table)
        pre_test_tables.append(baseline_test_table)

        support_info = {
            "validation": {
                "min_pos": int(np.nanmin(baseline_val_table.loc[baseline_val_table["group"] != "__GAP__", "n_pos"])),
                "min_neg": int(np.nanmin(baseline_val_table.loc[baseline_val_table["group"] != "__GAP__", "n_neg"])),
            },
            "test": {
                "min_pos": int(np.nanmin(baseline_test_table.loc[baseline_test_table["group"] != "__GAP__", "n_pos"])),
                "min_neg": int(np.nanmin(baseline_test_table.loc[baseline_test_table["group"] != "__GAP__", "n_neg"])),
            },
        }

        mitigation_history: List[Dict[str, object]] = []
        final_result = MitigationResult(
            stage="baseline",
            val_table=baseline_val_table,
            test_table=baseline_test_table,
            val_gap=baseline_val_gap,
            val_gap_ci=baseline_val_gap_ci,
            test_gap=baseline_test_gap,
            test_gap_ci=baseline_test_gap_ci,
            history=[],
        )

        eo_supported = (
            (baseline_val_table.loc[baseline_val_table["group"] != "__GAP__", "n_pos"] >= EO_MIN_POS).all()
            and (baseline_val_table.loc[baseline_val_table["group"] != "__GAP__", "n_neg"] >= EO_MIN_NEG).all()
        )

        if eo_supported and not np.isnan(baseline_val_gap) and baseline_val_gap > TPR_GAP_ALERT_THRESHOLD:
            try:
                eo_val_pred, eo_test_pred, eo_record = attempt_equal_opportunity(
                    model, X_val, X_test, y_val, y_test, groups_val, groups_test
                )
                eo_val_table, eo_val_gap, eo_val_gap_ci = build_fairness_table(
                    y_val, eo_val_pred, groups_val, attribute, "validation", "post_equal_opportunity"
                )
                eo_test_table, eo_test_gap, eo_test_gap_ci = build_fairness_table(
                    y_test, eo_test_pred, groups_test, attribute, "test", "post_equal_opportunity"
                )
                eo_record.update(
                    {
                        "validation_tpr_gap": eo_val_gap,
                        "validation_tpr_gap_ci": list(eo_val_gap_ci),
                        "test_tpr_gap": eo_test_gap,
                        "test_tpr_gap_ci": list(eo_test_gap_ci),
                    }
                )
                mitigation_history.append(eo_record)
                final_result = MitigationResult(
                    stage="post_equal_opportunity",
                    val_table=eo_val_table,
                    test_table=eo_test_table,
                    val_gap=eo_val_gap,
                    val_gap_ci=eo_val_gap_ci,
                    test_gap=eo_test_gap,
                    test_gap_ci=eo_test_gap_ci,
                    history=[eo_record],
                )

                eod_supported = (
                    (eo_val_table.loc[eo_val_table["group"] != "__GAP__", "n_pos"] >= EOD_MIN_POS).all()
                    and (eo_val_table.loc[eo_val_table["group"] != "__GAP__", "n_neg"] >= EOD_MIN_NEG).all()
                )

                if eod_supported and not np.isnan(eo_val_gap) and eo_val_gap > TPR_GAP_ALERT_THRESHOLD:
                    try:
                        eod_val_pred, eod_test_pred, eod_record = attempt_equalized_odds(
                            model, X_val, X_test, y_val, y_test, groups_val, groups_test
                        )
                        eod_val_table, eod_val_gap, eod_val_gap_ci = build_fairness_table(
                            y_val, eod_val_pred, groups_val, attribute, "validation", "post_equalized_odds"
                        )
                        eod_test_table, eod_test_gap, eod_test_gap_ci = build_fairness_table(
                            y_test, eod_test_pred, groups_test, attribute, "test", "post_equalized_odds"
                        )
                        eod_record.update(
                            {
                                "validation_tpr_gap": eod_val_gap,
                                "validation_tpr_gap_ci": list(eod_val_gap_ci),
                                "test_tpr_gap": eod_test_gap,
                                "test_tpr_gap_ci": list(eod_test_gap_ci),
                            }
                        )
                        mitigation_history.append(eod_record)
                        final_result = MitigationResult(
                            stage="post_equalized_odds",
                            val_table=eod_val_table,
                            test_table=eod_test_table,
                            val_gap=eod_val_gap,
                            val_gap_ci=eod_val_gap_ci,
                            test_gap=eod_test_gap,
                            test_gap_ci=eod_test_gap_ci,
                            history=mitigation_history.copy(),
                        )
                    except ValueError as exc:
                        mitigation_history.append(
                            {
                                "constraint": "equalized_odds",
                                "status": "failed",
                                "reason": str(exc),
                            }
                        )
                else:
                    mitigation_history.append(
                        {
                            "constraint": "equalized_odds",
                            "status": "skipped_due_to_data",
                            "reason": "insufficient_support",
                        }
                    )
            except ValueError as exc:
                mitigation_history.append(
                    {
                        "constraint": "true_positive_rate_parity",
                        "status": "failed",
                        "reason": str(exc),
                    }
                )
        else:
            mitigation_history.append(
                {
                    "constraint": "true_positive_rate_parity",
                    "status": "skipped_due_to_data",
                    "reason": "insufficient_support" if eo_supported is False else "gap_below_threshold",
                }
            )

        final_result.history = mitigation_history
        post_val_tables.append(final_result.val_table)
        post_test_tables.append(final_result.test_table)

        if (
            not np.isnan(final_result.test_gap)
            and final_result.test_gap > TPR_GAP_ALERT_THRESHOLD
            and final_result.test_gap_ci[0] > 0
        ):
            alerts.append(
                f"{attribute}: post-stage {final_result.stage} test TPR-gap={final_result.test_gap:.3f} "
                f"[{final_result.test_gap_ci[0]:.3f}, {final_result.test_gap_ci[1]:.3f}]"
            )

        attribute_payload = {
            "attribute": attribute,
            "support": support_info,
            "baseline": {
                "validation": {
                    "tpr_gap": baseline_val_gap,
                    "tpr_gap_ci": list(baseline_val_gap_ci),
                    "metrics": baseline_val_table.to_dict(orient="records"),
                },
                "test": {
                    "tpr_gap": baseline_test_gap,
                    "tpr_gap_ci": list(baseline_test_gap_ci),
                    "metrics": baseline_test_table.to_dict(orient="records"),
                },
            },
            "final_stage": {
                "stage": final_result.stage,
                "validation_tpr_gap": final_result.val_gap,
                "validation_tpr_gap_ci": list(final_result.val_gap_ci),
                "test_tpr_gap": final_result.test_gap,
                "test_tpr_gap_ci": list(final_result.test_gap_ci),
                "metrics_validation": final_result.val_table.to_dict(orient="records"),
                "metrics_test": final_result.test_table.to_dict(orient="records"),
            },
            "history": mitigation_history,
        }
        audit_payload["attributes"].append(attribute_payload)

    audit_payload["alerts"] = alerts

    pd.concat(pre_val_tables, ignore_index=True).to_csv(FAIRNESS_PRE_VAL_PATH, index=False)
    pd.concat(pre_test_tables, ignore_index=True).to_csv(FAIRNESS_PRE_TEST_PATH, index=False)
    pd.concat(post_val_tables, ignore_index=True).to_csv(FAIRNESS_POST_VAL_PATH, index=False)
    pd.concat(post_test_tables, ignore_index=True).to_csv(FAIRNESS_POST_TEST_PATH, index=False)
    FAIRNESS_AUDIT_PATH.write_text(json.dumps(audit_payload, indent=2), encoding="utf-8")

    print(f"[INFO] Fairness audit saved to {FAIRNESS_AUDIT_PATH}")
    if alerts:
        print("[ALERTA] Condições críticas detectadas:")
        for alert in alerts:
            print(f"  - {alert}")
    else:
        print("[INFO] Nenhum alerta crítico pós-mitigação.")


if __name__ == "__main__":
    main()

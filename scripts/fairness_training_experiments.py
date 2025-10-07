"""Experiments with fairness-aware training strategies.

This script evaluates two alternatives beyond post-processing for the
most critical attribute (``is_elderly``):

1. ExponentiatedGradient with a True Positive Rate parity constraint.
2. AIF360 Reweighing followed by logistic regression.

For each approach, it:
  * Trains a logistic pipeline using the existing preprocessor.
  * Generates probability scores on validation/test splits.
  * Applies the frozen threshold (from ``results/threshold.json``).
  * Reports performance metrics and TPR gap (with bootstrap CIs) per group.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from fairlearn.reductions import ExponentiatedGradient, TruePositiveRateParity
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils import check_random_state

RESULTS_DIR = Path("results")
THRESHOLD_PATH = RESULTS_DIR / "threshold.json"
TARGET_ATTRIBUTE = "is_elderly"
BOOTSTRAP_ITERATIONS = 1000
RANDOM_STATE = 42


@dataclass
class ExperimentResult:
    name: str
    overall_val: Dict[str, float]
    overall_test: Dict[str, float]
    tpr_gap_val: float
    tpr_gap_val_ci: Tuple[float, float]
    tpr_gap_test: float
    tpr_gap_test_ci: Tuple[float, float]
    notes: List[str]


def load_threshold() -> float:
    if not THRESHOLD_PATH.exists():
        raise FileNotFoundError("results/threshold.json was not found.")
    payload = json.loads(THRESHOLD_PATH.read_text(encoding="utf-8"))
    threshold = payload.get("threshold", payload.get("value"))
    if threshold is None:
        raise KeyError("Threshold payload missing 'threshold' or 'value'.")
    source = payload.get("source")
    if source != "validation_calibrated":
        raise ValueError("Frozen threshold must come from validation (calibrated).")
    return float(threshold)


def global_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0.0
    tpr = recall
    tnr = tn / (tn + fp) if (tn + fp) else 0.0
    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": balanced_accuracy,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "support": int(len(y_true)),
    }


def bootstrap_tpr_gap(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    groups: np.ndarray,
    threshold: float,
    iterations: int = BOOTSTRAP_ITERATIONS,
    random_state: int = RANDOM_STATE,
) -> Tuple[float, Tuple[float, float]]:
    rng = check_random_state(random_state)
    y_pred = (y_prob >= threshold).astype(int)
    base_gap = tpr_gap(y_true, y_pred, groups)

    samples = []
    n = len(y_true)
    for _ in range(iterations):
        idx = rng.randint(0, n, size=n)
        samples.append(
            tpr_gap(y_true[idx], y_pred[idx], groups[idx])
        )
    ci_low = float(np.percentile(samples, 2.5))
    ci_high = float(np.percentile(samples, 97.5))
    return base_gap, (ci_low, ci_high)


def tpr_gap(y_true: np.ndarray, y_pred: np.ndarray, groups: np.ndarray) -> float:
    gaps = []
    for group in np.unique(groups):
        mask = groups == group
        positives = (y_true[mask] == 1).sum()
        if positives == 0:
            continue
        tpr = ((y_true[mask] == 1) & (y_pred[mask] == 1)).sum() / positives
        gaps.append(tpr)
    if not gaps:
        return np.nan
    return float(max(gaps) - min(gaps))


def build_base_estimator(preprocessor, random_state: int = RANDOM_STATE) -> Pipeline:
    log_reg = LogisticRegression(
        penalty="elasticnet",
        solver="saga",
        l1_ratio=0.2,
        C=0.004893900918477494,
        class_weight="balanced",
        max_iter=4000,
        random_state=random_state,
    )
    return Pipeline(
        steps=[
            ("prep", clone(preprocessor)),
            ("clf", log_reg),
        ]
    )


def train_exponentiated_gradient(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> ExponentiatedGradient:
    base_estimator = build_base_estimator(preprocessor)
    constraint = TruePositiveRateParity()
    eg = ExponentiatedGradient(
        estimator=base_estimator,
        constraints=constraint,
        eps=0.01,
        max_iter=50,
        sample_weight_name="clf__sample_weight",
    )
    eg.fit(X_train, y_train, sensitive_features=X_train[TARGET_ATTRIBUTE])
    return eg


def compute_reweigh_weights(groups: pd.Series, labels: pd.Series) -> np.ndarray:
    n = len(groups)
    group_counts = groups.value_counts().to_dict()
    label_counts = labels.value_counts().to_dict()
    joint = (
        pd.DataFrame({"group": groups, "label": labels})
        .value_counts()
        .to_dict()
    )

    group_probs = {g: c / n for g, c in group_counts.items()}
    label_probs = {l: c / n for l, c in label_counts.items()}
    joint_probs = {k: c / n for k, c in joint.items()}

    weights = np.zeros(n, dtype=float)
    for idx, (g, y) in enumerate(zip(groups, labels)):
        numerator = group_probs[g] * label_probs[y]
        denominator = joint_probs.get((g, y), 1e-12)
        weights[idx] = numerator / denominator
    return weights


def model_proba(model, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    preds = model.predict(X)
    return np.asarray(preds, dtype=float)


def train_reweighing(
    preprocessor,
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    weights = compute_reweigh_weights(X_train[TARGET_ATTRIBUTE], y_train)
    estimator = build_base_estimator(preprocessor)
    estimator.fit(X_train, y_train, clf__sample_weight=weights)
    return estimator


def run_experiments():
    artifacts = joblib.load("analysis_artifacts.joblib")
    preprocessor = artifacts["preprocessor"]
    X_train = pd.DataFrame(artifacts["X_train"])
    y_train = pd.Series(artifacts["y_train"]).astype(int)
    X_val = pd.DataFrame(artifacts["X_val"])
    y_val = np.asarray(artifacts["y_val"]).astype(int)
    X_test = pd.DataFrame(artifacts["X_test"])
    y_test = np.asarray(artifacts["y_test"]).astype(int)

    threshold = load_threshold()

    experiments: List[ExperimentResult] = []

    # Baseline (calibrated) from artifacts
    base_model = artifacts["best_model_calibrated"]
    base_prob_val = model_proba(base_model, X_val)
    base_prob_test = model_proba(base_model, X_test)
    base_overall_val = global_metrics(y_val, base_prob_val, threshold)
    base_overall_test = global_metrics(y_test, base_prob_test, threshold)
    base_gap_val, base_gap_ci_val = bootstrap_tpr_gap(
        y_val, base_prob_val, X_val[TARGET_ATTRIBUTE].to_numpy(), threshold
    )
    base_gap_test, base_gap_ci_test = bootstrap_tpr_gap(
        y_test, base_prob_test, X_test[TARGET_ATTRIBUTE].to_numpy(), threshold
    )
    experiments.append(
        ExperimentResult(
            name="baseline_calibrated",
            overall_val=base_overall_val,
            overall_test=base_overall_test,
            tpr_gap_val=base_gap_val,
            tpr_gap_val_ci=base_gap_ci_val,
            tpr_gap_test=base_gap_test,
            tpr_gap_test_ci=base_gap_ci_test,
            notes=["Existing calibrated pipeline."],
        )
    )

    # ExponentiatedGradient
    try:
        eg_model = train_exponentiated_gradient(preprocessor, X_train, y_train)
        eg_prob_val = model_proba(eg_model, X_val)
        eg_prob_test = model_proba(eg_model, X_test)
        eg_overall_val = global_metrics(y_val, eg_prob_val, threshold)
        eg_overall_test = global_metrics(y_test, eg_prob_test, threshold)
        eg_gap_val, eg_gap_ci_val = bootstrap_tpr_gap(
            y_val, eg_prob_val, X_val[TARGET_ATTRIBUTE].to_numpy(), threshold
        )
        eg_gap_test, eg_gap_ci_test = bootstrap_tpr_gap(
            y_test, eg_prob_test, X_test[TARGET_ATTRIBUTE].to_numpy(), threshold
        )
        experiments.append(
            ExperimentResult(
                name="exponentiated_gradient",
                overall_val=eg_overall_val,
                overall_test=eg_overall_test,
                tpr_gap_val=eg_gap_val,
                tpr_gap_val_ci=eg_gap_ci_val,
                tpr_gap_test=eg_gap_test,
                tpr_gap_test_ci=eg_gap_ci_test,
                notes=["ExponentiatedGradient (TPR parity) on is_elderly."],
            )
        )
    except Exception as exc:
        experiments.append(
            ExperimentResult(
                name="exponentiated_gradient",
                overall_val={},
                overall_test={},
                tpr_gap_val=float("nan"),
                tpr_gap_val_ci=(float("nan"), float("nan")),
                tpr_gap_test=float("nan"),
                tpr_gap_test_ci=(float("nan"), float("nan")),
                notes=[f"Failed: {exc}"],
            )
        )

    # Reweighing
    try:
        rw_model = train_reweighing(preprocessor, X_train, y_train)
        rw_prob_val = model_proba(rw_model, X_val)
        rw_prob_test = model_proba(rw_model, X_test)
        rw_overall_val = global_metrics(y_val, rw_prob_val, threshold)
        rw_overall_test = global_metrics(y_test, rw_prob_test, threshold)
        rw_gap_val, rw_gap_ci_val = bootstrap_tpr_gap(
            y_val, rw_prob_val, X_val[TARGET_ATTRIBUTE].to_numpy(), threshold
        )
        rw_gap_test, rw_gap_ci_test = bootstrap_tpr_gap(
            y_test, rw_prob_test, X_test[TARGET_ATTRIBUTE].to_numpy(), threshold
        )
        experiments.append(
            ExperimentResult(
                name="aif360_reweighing",
                overall_val=rw_overall_val,
                overall_test=rw_overall_test,
                tpr_gap_val=rw_gap_val,
                tpr_gap_val_ci=rw_gap_ci_val,
                tpr_gap_test=rw_gap_test,
                tpr_gap_test_ci=rw_gap_ci_test,
                notes=["AIF360 Reweighing (is_elderly) + logistic regression."],
            )
        )
    except Exception as exc:
        experiments.append(
            ExperimentResult(
                name="aif360_reweighing",
                overall_val={},
                overall_test={},
                tpr_gap_val=float("nan"),
                tpr_gap_val_ci=(float("nan"), float("nan")),
                tpr_gap_test=float("nan"),
                tpr_gap_test_ci=(float("nan"), float("nan")),
                notes=[f"Failed: {exc}"],
            )
        )

    records = []
    for exp in experiments:
        record = {
            "experiment": exp.name,
            "precision_val": exp.overall_val.get("precision"),
            "recall_val": exp.overall_val.get("recall"),
            "balanced_accuracy_val": exp.overall_val.get("balanced_accuracy"),
            "precision_test": exp.overall_test.get("precision"),
            "recall_test": exp.overall_test.get("recall"),
            "balanced_accuracy_test": exp.overall_test.get("balanced_accuracy"),
            "tpr_gap_val": exp.tpr_gap_val,
            "tpr_gap_val_ci_low": exp.tpr_gap_val_ci[0],
            "tpr_gap_val_ci_high": exp.tpr_gap_val_ci[1],
            "tpr_gap_test": exp.tpr_gap_test,
            "tpr_gap_test_ci_low": exp.tpr_gap_test_ci[0],
            "tpr_gap_test_ci_high": exp.tpr_gap_test_ci[1],
            "notes": "; ".join(exp.notes),
        }
        records.append(record)

    df = pd.DataFrame(records)
    output_path = RESULTS_DIR / "fairness_training_experiments.csv"
    df.to_csv(output_path, index=False)
    print(df.to_string(index=False))
    print(f"\n[INFO] Summary saved to {output_path}")


if __name__ == "__main__":
    run_experiments()

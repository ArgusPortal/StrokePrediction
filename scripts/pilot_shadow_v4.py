"""
Pilot shadow evaluation pipeline for Stroke Prediction v4.

Generates calibrated classification metrics, precision-recall curves,
decision curve analysis, and fairness audits using the production threshold.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
)

from src.fairness_audit import (
    audit_fairness_baseline,
    generate_fairness_report,
    mitigate_fairness_staged,
)

plt.switch_backend("Agg")


RESULTS_PATH = Path("results")
PRODUCTION_THRESHOLD_PATH = RESULTS_PATH / "threshold.json"
ARTIFACTS_PATH = Path("analysis_artifacts.joblib")
PR_CURVE_TEST_PATH = RESULTS_PATH / "pr_curve_test.png"
PR_CURVE_VAL_PATH = RESULTS_PATH / "pr_curve_val.png"
DCA_TEST_CSV = RESULTS_PATH / "dca_test.csv"
DCA_TEST_PNG = RESULTS_PATH / "dca_test.png"
DCA_VAL_CSV = RESULTS_PATH / "dca_val.csv"
DCA_VAL_PNG = RESULTS_PATH / "dca_val.png"
METRICS_TEST_CSV = RESULTS_PATH / "metrics_threshold_test.csv"
METRICS_VAL_CSV = RESULTS_PATH / "metrics_threshold_val.csv"


@dataclass
class ClassificationReport:
    threshold: float
    tp: int
    fp: int
    fn: int
    tn: int
    recall: float
    precision: float
    f1: float
    specificity: float
    balanced_accuracy: float
    signal_rate: float
    support: int

    def as_csv_row(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "balanced_accuracy": self.balanced_accuracy,
            "tp": self.tp,
            "fp": self.fp,
            "fn": self.fn,
            "tn": self.tn,
            "support": self.support,
            "selection_rate": self.signal_rate,
        }


def load_artifacts() -> Dict[str, object]:
    if not ARTIFACTS_PATH.exists():
        raise FileNotFoundError(f"Artifacts not found at {ARTIFACTS_PATH}")
    artifacts = joblib.load(ARTIFACTS_PATH)

    required_keys = [
        "X_val",
        "y_val",
        "X_test",
        "y_test",
        "y_proba_val_cal",
        "best_model_calibrated",
    ]
    missing = [key for key in required_keys if key not in artifacts]
    if missing:
        raise KeyError(f"Artifacts missing keys: {missing}")
    return artifacts


def load_production_threshold() -> float:
    if not PRODUCTION_THRESHOLD_PATH.exists():
        raise FileNotFoundError(f"Missing production threshold at {PRODUCTION_THRESHOLD_PATH}")
    with PRODUCTION_THRESHOLD_PATH.open("r", encoding="utf-8") as f:
        config = json.load(f)
    threshold = float(config["threshold"])
    if config.get("source") != "validation_calibrated":
        raise ValueError("Unexpected threshold source; expected 'validation_calibrated'")
    return threshold


def ensure_test_probabilities(artifacts: Dict[str, object]) -> np.ndarray:
    if "y_proba_test_cal" in artifacts:
        return np.asarray(artifacts["y_proba_test_cal"], dtype=float)
    model = artifacts.get("best_model_calibrated")
    if model is None:
        raise KeyError("Calibrated model not found in artifacts to compute test probabilities.")
    y_proba_test = model.predict_proba(artifacts["X_test"])[:, 1]
    return np.asarray(y_proba_test, dtype=float)


def compute_classification_report(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> ClassificationReport:
    y_pred = (y_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    balanced_acc = balanced_accuracy_score(y_true, y_pred)
    signal_rate = y_pred.mean()

    return ClassificationReport(
        threshold=threshold,
        tp=int(tp),
        fp=int(fp),
        fn=int(fn),
        tn=int(tn),
        recall=recall,
        precision=precision,
        f1=f1,
        specificity=specificity,
        balanced_accuracy=balanced_acc,
        signal_rate=signal_rate,
        support=len(y_true),
    )


def display_report(label: str, report: ClassificationReport) -> None:
    print(f"\n{label} @ threshold={report.threshold:.4f}")
    print(f"TP={report.tp}, FP={report.fp}, FN={report.fn}, TN={report.tn}")
    print(
        "Recall/TPR={:.3f}, Precision/PPV={:.3f}, F1={:.3f}, "
        "Specificity/TNR={:.3f}, Balanced Accuracy={:.3f}, % sinalizada={:.2f}%".format(
            report.recall,
            report.precision,
            report.f1,
            report.specificity,
            report.balanced_accuracy,
            report.signal_rate * 100,
        )
    )


def update_metrics_csv(path: Path, reports: Iterable[ClassificationReport]) -> None:
    rows = [r.as_csv_row() for r in reports]
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)


def generate_precision_recall_plot(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    threshold: float,
    output_path: Path,
    label: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    plt.figure(figsize=(7, 5))
    plt.plot(recall, precision, label="Modelo calibrado")
    plt.xlabel("Recall (Sensibilidade)")
    plt.ylabel("Precisão (PPV)")
    plt.title(f"Curva Precision-Recall - {label}")
    plt.ylim(0.0, 1.05)
    plt.xlim(0.0, 1.0)

    op_report = compute_classification_report(y_true, y_proba, threshold)
    plt.scatter([op_report.recall], [op_report.precision], color="red", label=f"Ponto operacional t={threshold:.3f}")
    plt.annotate(
        f"PPV={op_report.precision:.2f}\nRecall={op_report.recall:.2f}",
        (op_report.recall, op_report.precision),
        textcoords="offset points",
        xytext=(10, -15),
        ha="left",
        fontsize=9,
        color="red",
    )
    plt.legend(loc="best")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
    return precision, recall, pr_thresholds


def decision_curve_analysis(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    thresholds: Iterable[float],
) -> pd.DataFrame:
    y_true_arr = np.asarray(y_true, dtype=int)
    n = len(y_true_arr)
    prevalence = y_true_arr.mean()
    thresholds = np.asarray(list(thresholds))

    nb_model: List[float] = []
    nb_all: List[float] = []
    nb_none: List[float] = []

    for t in thresholds:
        if not 0 < t < 1:
            raise ValueError("Thresholds for DCA must be between 0 and 1.")
        weight = t / (1 - t)

        y_pred = (y_proba >= t).astype(int)
        tp = np.sum((y_pred == 1) & (y_true_arr == 1))
        fp = np.sum((y_pred == 1) & (y_true_arr == 0))
        nb_model.append((tp / n) - (fp / n) * weight)

        nb_all.append(prevalence - (1 - prevalence) * weight)
        nb_none.append(0.0)

    return pd.DataFrame(
        {
            "threshold": thresholds,
            "net_benefit_model": nb_model,
            "net_benefit_treat_all": nb_all,
            "net_benefit_treat_none": nb_none,
        }
    )


def plot_decision_curve(df: pd.DataFrame, output_path: Path, production_threshold: float, label: str) -> None:
    plt.figure(figsize=(7, 5))
    plt.plot(df["threshold"], df["net_benefit_model"], label="Modelo")
    plt.plot(df["threshold"], df["net_benefit_treat_all"], label="Tratar todos", linestyle="--")
    plt.plot(df["threshold"], df["net_benefit_treat_none"], label="Tratar ninguém", linestyle=":")
    plt.axvline(production_threshold, color="red", linestyle="-.", label=f"Limiar produção {production_threshold:.3f}")
    plt.xlabel("Probabilidade-limiar")
    plt.ylabel("Net benefit")
    plt.title(f"Decision Curve Analysis - {label}")
    plt.legend(loc="best")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300)
    plt.close()


def main() -> None:
    artifacts = load_artifacts()
    production_threshold = load_production_threshold()
    y_proba_test = ensure_test_probabilities(artifacts)
    y_proba_val = np.asarray(artifacts["y_proba_val_cal"], dtype=float)

    X_val = artifacts["X_val"]
    y_val = np.asarray(artifacts["y_val"], dtype=int)
    X_test = artifacts["X_test"]
    y_test = np.asarray(artifacts["y_test"], dtype=int)

    # Task 1: classification metrics on test split
    report_test = compute_classification_report(y_test, y_proba_test, production_threshold)
    display_report("TEST", report_test)
    update_metrics_csv(METRICS_TEST_CSV, [report_test])

    # Optional Task 2: validation at alternative threshold
    alt_threshold_val = 0.070
    report_val_prod = compute_classification_report(y_val, y_proba_val, production_threshold)
    report_val_alt = compute_classification_report(y_val, y_proba_val, alt_threshold_val)
    display_report("VAL (produção)", report_val_prod)
    display_report(f"VAL (t={alt_threshold_val:.3f})", report_val_alt)
    update_metrics_csv(METRICS_VAL_CSV, [report_val_prod, report_val_alt])

    # Task 3: Precision-Recall curves
    generate_precision_recall_plot(y_test, y_proba_test, production_threshold, PR_CURVE_TEST_PATH, "Teste")
    generate_precision_recall_plot(y_val, y_proba_val, alt_threshold_val, PR_CURVE_VAL_PATH, "Validação")

    # Task 4: Decision Curve Analysis
    dca_thresholds = np.arange(0.06, 0.1201, 0.005)
    dca_test_df = decision_curve_analysis(y_test, y_proba_test, dca_thresholds)
    dca_test_df.to_csv(DCA_TEST_CSV, index=False)
    plot_decision_curve(dca_test_df, DCA_TEST_PNG, production_threshold, "Teste")

    dca_val_df = decision_curve_analysis(y_val, y_proba_val, dca_thresholds)
    dca_val_df.to_csv(DCA_VAL_CSV, index=False)
    plot_decision_curve(dca_val_df, DCA_VAL_PNG, alt_threshold_val, "Validação")

    # Task 5: Fairness audit and mitigation
    sensitive_attrs = ["is_elderly", "gender", "smoking_status", "Residence_type", "work_type"]
    baseline_val = audit_fairness_baseline(
        X=X_val,
        y=y_val,
        y_proba=y_proba_val,
        threshold=production_threshold,
        sensitive_attrs=sensitive_attrs,
        dataset_name="validation",
        results_path=RESULTS_PATH,
        n_boot=500,
    )
    baseline_test = audit_fairness_baseline(
        X=X_test,
        y=y_test,
        y_proba=y_proba_test,
        threshold=production_threshold,
        sensitive_attrs=sensitive_attrs,
        dataset_name="test",
        results_path=RESULTS_PATH,
        n_boot=500,
    )

    mitigation = mitigate_fairness_staged(
        X_val=X_val,
        y_val=y_val,
        y_proba_val=y_proba_val,
        X_test=X_test,
        y_test=y_test,
        y_proba_test=y_proba_test,
        sensitive_attrs=sensitive_attrs,
        threshold_base=production_threshold,
        n_boot=500,
        results_path=RESULTS_PATH,
    )

    generate_fairness_report(
        baseline_val=baseline_val,
        baseline_test=baseline_test,
        mitigation=mitigation,
        results_path=RESULTS_PATH,
    )


if __name__ == "__main__":
    main()

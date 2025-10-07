"""Advanced modeling experiments for clinical stroke triage.

Implements:
1. Regularized Logistic Regression + isotonic calibration.
2. Monotonic XGBoost (numeric features constrained to increase risk) + calibration.
3. Super Learner (stacked ensemble) with calibration.

Outputs consolidated metrics and probability files under results/.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingClassifier, StackingClassifier
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

THRESHOLD = 0.085
RANDOM_STATE = 42


def load_artifacts() -> Dict:
    return joblib.load("analysis_artifacts.joblib")


def evaluation_metrics(y_true: np.ndarray, y_proba: np.ndarray, threshold: float) -> Dict[str, float]:
    y_pred = (y_proba >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary", zero_division=0
    )
    bal_acc = balanced_accuracy_score(y_true, y_pred)
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    return {
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def prepare_monotonic_constraints(preprocessor, numeric_monotonic: set[str]) -> str:
    # preprocessor is fitted ColumnTransformer
    feature_names = preprocessor.get_feature_names_out()
    constraints = []
    for name in feature_names:
        constraints.append(1 if name in numeric_monotonic else 0)
    return "(" + ",".join(str(int(c)) for c in constraints) + ")"


def main():
    artifacts = load_artifacts()
    X_train = pd.DataFrame(artifacts["X_train"])
    y_train = pd.Series(artifacts["y_train"]).astype(int)
    X_val = pd.DataFrame(artifacts["X_val"])
    y_val = pd.Series(artifacts["y_val"]).astype(int)
    X_test = pd.DataFrame(artifacts["X_test"])
    y_test = pd.Series(artifacts["y_test"]).astype(int)
    base_pipeline = artifacts["best_model"]
    preprocessor_template = base_pipeline.named_steps["prep"]

    results_payload = {}

    # Fit template preprocessor to training data for consistent feature order
    preprocessor_fitted = clone(preprocessor_template).fit(X_train, y_train)
    numeric_monotonic = {
        "age",
        "avg_glucose_level",
        "bmi",
        "cardio_risk_score",
        "age_squared",
        "bmi_age_interaction",
        "glucose_age_risk",
        "smoking_risk",
        "total_risk_score",
        "age_hypertension_int",
    }
    monotonic_constraints = prepare_monotonic_constraints(preprocessor_fitted, numeric_monotonic)

    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # 1. Regularized Logistic Regression
    logistic_pipe = Pipeline(
        steps=[
            ("prep", clone(preprocessor_template)),
            (
                "clf",
                LogisticRegression(
                    penalty="elasticnet",
                    solver="saga",
                    l1_ratio=0.25,
                    C=0.1,
                    max_iter=4000,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    logistic_calibrated = CalibratedClassifierCV(
        estimator=logistic_pipe,
        method="isotonic",
        cv=cv5,
        n_jobs=-1,
    )
    logistic_calibrated.fit(X_train, y_train)
    y_val_lr = logistic_calibrated.predict_proba(X_val)[:, 1]
    y_test_lr = logistic_calibrated.predict_proba(X_test)[:, 1]
    results_payload["logistic_regularized"] = {
        "val": evaluation_metrics(y_val.to_numpy(), y_val_lr, THRESHOLD),
        "test": evaluation_metrics(y_test.to_numpy(), y_test_lr, THRESHOLD),
    }

    # 2. Monotonic XGBoost
    xgb_estimator = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        seed=RANDOM_STATE,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=400,
        monotone_constraints=monotonic_constraints,
        reg_lambda=1.0,
        reg_alpha=0.1,
    )
    xgb_pipe = Pipeline(steps=[("prep", clone(preprocessor_template)), ("clf", xgb_estimator)])
    xgb_pipe.fit(X_train, y_train)
    xgb_calibrated = CalibratedClassifierCV(
        estimator=xgb_pipe,
        method="isotonic",
        cv="prefit",
    )
    xgb_calibrated.fit(X_val, y_val)
    y_val_xgb = xgb_calibrated.predict_proba(X_val)[:, 1]
    y_test_xgb = xgb_calibrated.predict_proba(X_test)[:, 1]
    results_payload["xgboost_monotonic"] = {
        "val": evaluation_metrics(y_val.to_numpy(), y_val_xgb, THRESHOLD),
        "test": evaluation_metrics(y_test.to_numpy(), y_test_xgb, THRESHOLD),
    }

    # 3. Super Learner (Stacking)
    base_lr = Pipeline(
        steps=[
            ("prep", clone(preprocessor_template)),
            (
                "clf",
                LogisticRegression(
                    penalty="l2",
                    solver="lbfgs",
                    max_iter=3000,
                    class_weight="balanced",
                    C=1.0,
                ),
            ),
        ]
    )
    base_hgb = Pipeline(
        steps=[
            ("prep", clone(preprocessor_template)),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=0.05,
                    max_depth=5,
                    max_iter=400,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    stacking = StackingClassifier(
        estimators=[("lr", base_lr), ("hgb", base_hgb)],
        final_estimator=LogisticRegression(max_iter=2000, class_weight="balanced"),
        passthrough=False,
        cv=cv5,
        n_jobs=-1,
    )
    stacking_calibrated = CalibratedClassifierCV(
        estimator=stacking,
        method="isotonic",
        cv=cv5,
        n_jobs=-1,
    )
    stacking_calibrated.fit(X_train, y_train)
    y_val_stack = stacking_calibrated.predict_proba(X_val)[:, 1]
    y_test_stack = stacking_calibrated.predict_proba(X_test)[:, 1]
    results_payload["super_learner"] = {
        "val": evaluation_metrics(y_val.to_numpy(), y_val_stack, THRESHOLD),
        "test": evaluation_metrics(y_test.to_numpy(), y_test_stack, THRESHOLD),
    }

    # Persist probability curves for manual inspection if needed
    proba_payload = {
        "logistic_regularized": {
            "val": y_val_lr.tolist(),
            "test": y_test_lr.tolist(),
        },
        "xgboost_monotonic": {
            "val": y_val_xgb.tolist(),
            "test": y_test_xgb.tolist(),
        },
        "super_learner": {
            "val": y_val_stack.tolist(),
            "test": y_test_stack.tolist(),
        },
    }

    (RESULTS_DIR / "model_next_steps_metrics.json").write_text(
        json.dumps(results_payload, indent=2), encoding="utf-8"
    )
    (RESULTS_DIR / "model_next_steps_probabilities.json").write_text(
        json.dumps(proba_payload), encoding="utf-8"
    )
    print("Advanced modeling experiments completed. Metrics saved to results/model_next_steps_metrics.json")


if __name__ == "__main__":
    main()

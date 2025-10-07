"""Fairness audit utilities with bootstrap confidence intervals and staged mitigation."""

import logging
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

logger = logging.getLogger(__name__)

try:
    from fairlearn.metrics import MetricFrame, selection_rate, true_positive_rate, false_positive_rate
    from fairlearn.postprocessing import ThresholdOptimizer
    FAIRLEARN_AVAILABLE = True
except ImportError:
    logger.warning("Fairlearn not available. Install with: pip install fairlearn")
    MetricFrame = None  # type: ignore
    ThresholdOptimizer = None  # type: ignore
    FAIRLEARN_AVAILABLE = False


class PrecomputedProbabilityEstimator:
    """Lightweight estimator that serves precomputed probabilities to ThresholdOptimizer."""

    def __init__(self, probas: Dict[str, np.ndarray]):
        self._probas = {key: np.asarray(values, dtype=float) for key, values in probas.items()}
        self._current_key: Optional[str] = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y=None):  # noqa: D401 - conforms to estimator API
        """No-op fit to satisfy sklearn estimator interface."""
        return self

    def get_params(self, deep: bool = False) -> Dict[str, object]:
        return {}

    def set_params(self, **params):
        return self

    def set_dataset(self, key: str) -> None:
        if key not in self._probas:
            raise KeyError(f"Dataset '{key}' has no stored probabilities.")
        self._current_key = key

    def predict_proba(self, X) -> np.ndarray:
        if self._current_key is None:
            raise ValueError("Dataset context not set before calling predict_proba.")
        proba = self._probas[self._current_key]
        n_samples = len(X)
        if len(proba) != n_samples:
            raise ValueError(
                f"Stored probabilities (n={len(proba)}) do not match input size (n={n_samples})."
            )
        proba_clipped = np.clip(proba, 0.0, 1.0)
        return np.column_stack([1.0 - proba_clipped, proba_clipped])


def _ensure_dataframe(data) -> pd.DataFrame:
    if isinstance(data, pd.DataFrame):
        return data.reset_index(drop=True).copy()
    return pd.DataFrame(data)


def _ensure_series(values) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.reset_index(drop=True)
    return pd.Series(values).reset_index(drop=True)


def _collapse_low_support_groups(
    X: pd.DataFrame,
    y: pd.Series,
    attr: str,
    min_pos: int = 5,
    min_neg: int = 0
) -> Tuple[pd.Series, Dict[str, Dict[str, int]]]:
    attr_series = X[attr].astype(str).copy()
    support_info: Dict[str, Dict[str, int]] = {}
    for group_value in attr_series.unique():
        mask = attr_series == group_value
        y_slice = y.loc[mask]
        n_pos = int((y_slice == 1).sum())
        n_neg = int((y_slice == 0).sum())
        support_info[str(group_value)] = {"n_pos": n_pos, "n_neg": n_neg, "n_total": int(mask.sum())}
        if n_pos < min_pos or n_neg < min_neg:
            attr_series.loc[mask] = "__OTHER__"
    return attr_series.reset_index(drop=True), support_info


def _calculate_group_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series
) -> Tuple[pd.DataFrame, float, float]:
    if not FAIRLEARN_AVAILABLE:
        raise ImportError("Fairlearn is required for this function")

    def positive_predictive_value(y_true_arr, y_pred_arr):
        cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
        if cm.size != 4:
            return 0.0
        tn, fp, fn, tp = cm.ravel()
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    mf = MetricFrame(
        metrics={
            'TPR': true_positive_rate,
            'FPR': false_positive_rate,
            'PPV': positive_predictive_value,
            'selection_rate': selection_rate,
        },
        y_true=y_true,
        y_pred=y_pred,
        sensitive_features=sensitive_features
    )

    metrics_df = mf.by_group.reset_index().rename(columns={'index': 'group'})
    if not metrics_df.empty:
        cols = list(metrics_df.columns)
        cols[0] = 'group'
        metrics_df.columns = cols
    if metrics_df.empty:
        return metrics_df, float('nan'), float('nan')
    tpr_gap = float(metrics_df['TPR'].max() - metrics_df['TPR'].min())
    fpr_gap = float(metrics_df['FPR'].max() - metrics_df['FPR'].min())
    return metrics_df, tpr_gap, fpr_gap


def _bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sensitive_features: pd.Series,
    n_boot: int = 1000,
    ci_quantiles: Tuple[float, float] = (0.05, 0.95),
    random_state: int = 42
) -> Dict[str, object]:
    if not FAIRLEARN_AVAILABLE:
        raise ImportError("Fairlearn is required for this function")

    rng = np.random.default_rng(random_state)
    n = len(y_true)
    sensitive = sensitive_features.reset_index(drop=True)
    unique_groups = sensitive.unique()
    group_scores = {group: [] for group in unique_groups}
    gap_scores: List[float] = []

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            mf = MetricFrame(
                metrics={'TPR': true_positive_rate},
                y_true=y_true[idx],
                y_pred=y_pred[idx],
                sensitive_features=sensitive.iloc[idx].reset_index(drop=True)
            )
        except Exception:
            continue
        tpr_by_group = mf.by_group['TPR']
        for group, value in tpr_by_group.items():
            group_scores[group].append(float(value))
        if len(tpr_by_group) >= 2:
            gap_scores.append(float(tpr_by_group.max() - tpr_by_group.min()))

    group_rows = []
    for group, scores in group_scores.items():
        if scores:
            scores_arr = np.asarray(scores)
            group_rows.append({
                'group': group,
                'TPR_mean': float(scores_arr.mean()),
                'TPR_ci_lower': float(np.quantile(scores_arr, ci_quantiles[0])),
                'TPR_ci_upper': float(np.quantile(scores_arr, ci_quantiles[1])),
            })
    group_df = pd.DataFrame(group_rows)
    if group_df.empty:
        group_df = pd.DataFrame(columns=['group', 'TPR_mean', 'TPR_ci_lower', 'TPR_ci_upper'])

    gap_dict: Dict[str, float] = {}
    if gap_scores:
        gaps = np.asarray(gap_scores)
        gap_dict = {
            'gap_mean': float(gaps.mean()),
            'gap_ci_lower': float(np.quantile(gaps, ci_quantiles[0])),
            'gap_ci_upper': float(np.quantile(gaps, ci_quantiles[1])),
        }

    return {'group_cis': group_df, 'gap_cis': gap_dict}


def audit_fairness_baseline(
    X: pd.DataFrame,
    y: pd.Series,
    y_proba: np.ndarray,
    threshold: float,
    sensitive_attrs: List[str],
    dataset_name: str,
    results_path: Path = Path("results"),
    n_boot: int = 1000,
    ci_quantiles: Tuple[float, float] = (0.05, 0.95)
) -> Dict[str, object]:
    if not FAIRLEARN_AVAILABLE:
        raise ImportError("Fairlearn is required. Install with: pip install fairlearn")

    X_df = _ensure_dataframe(X)
    y_series = _ensure_series(y).astype(int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= threshold).astype(int)

    audit_results: Dict[str, object] = {'attributes': {}}
    all_rows: List[pd.DataFrame] = []

    for attr in sensitive_attrs:
        if attr not in X_df.columns:
            logger.warning("Attribute '%s' not found in dataset %s; skipping baseline audit", attr, dataset_name)
            continue

        collapsed, support = _collapse_low_support_groups(X_df, y_series, attr, min_pos=5, min_neg=0)
        metrics_df, tpr_gap, fpr_gap = _calculate_group_metrics(y_series.values, y_pred, collapsed)
        ci = _bootstrap_confidence_intervals(y_series.values, y_pred, collapsed, n_boot=n_boot, ci_quantiles=ci_quantiles)

        metrics_with_meta = metrics_df.copy()
        metrics_with_meta['attribute'] = attr
        metrics_with_meta['dataset'] = dataset_name
        all_rows.append(metrics_with_meta)

        audit_results['attributes'][attr] = {
            'support_info': support,
            'metrics': metrics_with_meta.to_dict('records'),
            'confidence_intervals': ci['group_cis'].to_dict('records'),
            'TPR_gap': float(tpr_gap),
            'FPR_gap': float(fpr_gap),
            'TPR_gap_ci': ci['gap_cis'],
        }

    if all_rows:
        fairness_df = pd.concat(all_rows, ignore_index=True)
        output_path = results_path / f"fairness_pre_{dataset_name}.csv"
        fairness_df.to_csv(output_path, index=False)
        logger.info("Saved baseline fairness metrics to %s", output_path)

    return audit_results


def mitigate_fairness_staged(
    X_val: pd.DataFrame,
    y_val: pd.Series,
    y_proba_val: np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    y_proba_test: np.ndarray,
    sensitive_attrs: List[str],
    threshold_base: float,
    n_boot: int = 1000,
    ci_quantiles: Tuple[float, float] = (0.05, 0.95),
    results_path: Path = Path("results")
) -> Dict[str, object]:
    if not FAIRLEARN_AVAILABLE:
        raise ImportError("Fairlearn is required. Install with: pip install fairlearn")

    X_val_df = _ensure_dataframe(X_val)
    X_test_df = _ensure_dataframe(X_test)
    y_val_series = _ensure_series(y_val).astype(int)
    y_test_series = _ensure_series(y_test).astype(int)
    y_proba_val = np.asarray(y_proba_val, dtype=float)
    y_proba_test = np.asarray(y_proba_test, dtype=float)

    mitigation_results: Dict[str, object] = {
        'stages_applied': [],
        'stages_skipped': [],
        'post_mitigation_val': {},
        'post_mitigation_test': {},
        'alerts': []
    }

    all_post_val_rows: List[pd.DataFrame] = []
    all_post_test_rows: List[pd.DataFrame] = []

    for attr in sensitive_attrs:
        if attr not in X_val_df.columns or attr not in X_test_df.columns:
            logger.warning("Attribute '%s' not found in validation/test features; skipping", attr)
            continue

        collapsed_val, support_val = _collapse_low_support_groups(X_val_df, y_val_series, attr, min_pos=5, min_neg=0)
        collapsed_test, support_test = _collapse_low_support_groups(X_test_df, y_test_series, attr, min_pos=5, min_neg=0)

        eligible_eo = all(info['n_pos'] >= 5 for info in support_val.values())
        eligible_eod = all(info['n_pos'] >= 10 and info['n_neg'] >= 10 for info in support_val.values())

        final_stage_data: Optional[Dict[str, object]] = None
        last_failure_reason: Optional[str] = None

        if eligible_eo:
            try:
                estimator = PrecomputedProbabilityEstimator({'validation': y_proba_val, 'test': y_proba_test})
                estimator.set_dataset('validation')
                optimizer_eo = ThresholdOptimizer(
                    estimator=estimator,
                    constraints="true_positive_rate_parity",
                    objective="balanced_accuracy_score",
                    prefit=True,
                    predict_method='predict_proba'
                )
                optimizer_eo.fit(
                    X_val_df,
                    y_val_series.values,
                    sensitive_features=collapsed_val
                )

                estimator.set_dataset('validation')
                y_pred_val_eo = optimizer_eo.predict(
                    X_val_df,
                    sensitive_features=collapsed_val,
                    random_state=42
                )
                estimator.set_dataset('test')
                y_pred_test_eo = optimizer_eo.predict(
                    X_test_df,
                    sensitive_features=collapsed_test,
                    random_state=42
                )

                metrics_val_eo, tpr_gap_val_eo, fpr_gap_val_eo = _calculate_group_metrics(
                    y_true=y_val_series.values,
                    y_pred=y_pred_val_eo,
                    sensitive_features=collapsed_val
                )
                metrics_test_eo, tpr_gap_test_eo, fpr_gap_test_eo = _calculate_group_metrics(
                    y_true=y_test_series.values,
                    y_pred=y_pred_test_eo,
                    sensitive_features=collapsed_test
                )
                ci_test_eo = _bootstrap_confidence_intervals(
                    y_true=y_test_series.values,
                    y_pred=y_pred_test_eo,
                    sensitive_features=collapsed_test,
                    n_boot=n_boot,
                    ci_quantiles=ci_quantiles
                )

                val_with_meta = metrics_val_eo.copy()
                val_with_meta['attribute'] = attr
                val_with_meta['dataset'] = 'validation'
                val_with_meta['stage'] = 'post_equal_opportunity'
                all_post_val_rows.append(val_with_meta)

                test_with_meta = metrics_test_eo.merge(ci_test_eo['group_cis'], on='group', how='left')
                test_with_meta['attribute'] = attr
                test_with_meta['dataset'] = 'test'
                test_with_meta['stage'] = 'post_equal_opportunity'
                all_post_test_rows.append(test_with_meta)

                gap_ci = ci_test_eo.get('gap_cis', {})
                mitigation_results['stages_applied'].append({
                    'attribute': attr,
                    'stage': 'equal_opportunity',
                    'tpr_gap_val_post': float(tpr_gap_val_eo),
                    'tpr_gap_test_post': float(tpr_gap_test_eo),
                    'tpr_gap_test_ci': gap_ci
                })

                if tpr_gap_test_eo > 0.10 and gap_ci.get('gap_ci_lower', 0.0) > 0:
                    alert_msg = (
                        f"ALERT: {attr} retains TPR disparity after Equal Opportunity "
                        f"(gap={tpr_gap_test_eo:.4f}, CI lower={gap_ci['gap_ci_lower']:.4f})"
                    )
                    logger.warning(alert_msg)
                    mitigation_results['alerts'].append({
                        'attribute': attr,
                        'stage': 'post_equal_opportunity',
                        'message': alert_msg,
                        'tpr_gap': float(tpr_gap_test_eo),
                        'ci_lower': float(gap_ci['gap_ci_lower'])
                    })

                final_stage_data = {
                    'stage': 'equal_opportunity',
                    'val_df': val_with_meta.copy(),
                    'test_df': test_with_meta.copy(),
                    'tpr_gap_val': float(tpr_gap_val_eo),
                    'fpr_gap_val': float(fpr_gap_val_eo),
                    'tpr_gap_test': float(tpr_gap_test_eo),
                    'fpr_gap_test': float(fpr_gap_test_eo),
                    'gap_ci_test': gap_ci
                }
                last_failure_reason = None
            except Exception as exc:
                last_failure_reason = f'error: {exc}'
                logger.error("Stage 1 failed for %s: %s", attr, exc)
                mitigation_results['stages_skipped'].append({
                    'attribute': attr,
                    'stage': 'equal_opportunity',
                    'reason': last_failure_reason
                })
        else:
            last_failure_reason = 'insufficient_support'
            logger.info("Stage 1 (Equal Opportunity) skipped for %s due to low support", attr)
            mitigation_results['stages_skipped'].append({
                'attribute': attr,
                'stage': 'equal_opportunity',
                'reason': 'insufficient_support',
                'support_info': support_val
            })

        if eligible_eod and final_stage_data is not None:
            try:
                estimator_eod = PrecomputedProbabilityEstimator({'validation': y_proba_val, 'test': y_proba_test})
                estimator_eod.set_dataset('validation')
                optimizer_eod = ThresholdOptimizer(
                    estimator=estimator_eod,
                    constraints="equalized_odds",
                    objective="balanced_accuracy_score",
                    prefit=True,
                    predict_method='predict_proba'
                )
                optimizer_eod.fit(
                    X_val_df,
                    y_val_series.values,
                    sensitive_features=collapsed_val
                )

                estimator_eod.set_dataset('validation')
                y_pred_val_eod = optimizer_eod.predict(
                    X_val_df,
                    sensitive_features=collapsed_val,
                    random_state=42
                )
                estimator_eod.set_dataset('test')
                y_pred_test_eod = optimizer_eod.predict(
                    X_test_df,
                    sensitive_features=collapsed_test,
                    random_state=42
                )

                metrics_val_eod, tpr_gap_val_eod, fpr_gap_val_eod = _calculate_group_metrics(
                    y_true=y_val_series.values,
                    y_pred=y_pred_val_eod,
                    sensitive_features=collapsed_val
                )
                metrics_test_eod, tpr_gap_test_eod, fpr_gap_test_eod = _calculate_group_metrics(
                    y_true=y_test_series.values,
                    y_pred=y_pred_test_eod,
                    sensitive_features=collapsed_test
                )
                ci_test_eod = _bootstrap_confidence_intervals(
                    y_true=y_test_series.values,
                    y_pred=y_pred_test_eod,
                    sensitive_features=collapsed_test,
                    n_boot=n_boot,
                    ci_quantiles=ci_quantiles
                )

                val_eod_meta = metrics_val_eod.copy()
                val_eod_meta['attribute'] = attr
                val_eod_meta['dataset'] = 'validation'
                val_eod_meta['stage'] = 'post_equalized_odds'
                all_post_val_rows.append(val_eod_meta)

                test_eod_meta = metrics_test_eod.merge(ci_test_eod['group_cis'], on='group', how='left')
                test_eod_meta['attribute'] = attr
                test_eod_meta['dataset'] = 'test'
                test_eod_meta['stage'] = 'post_equalized_odds'
                all_post_test_rows.append(test_eod_meta)

                gap_ci_eod = ci_test_eod.get('gap_cis', {})
                mitigation_results['stages_applied'].append({
                    'attribute': attr,
                    'stage': 'equalized_odds',
                    'tpr_gap_test_post': float(tpr_gap_test_eod),
                    'fpr_gap_test_post': float(fpr_gap_test_eod),
                    'tpr_gap_test_ci': gap_ci_eod,
                    'note': 'Equalized odds may conflict with calibration; prioritize Equal Opportunity'
                })

                if tpr_gap_test_eod > 0.10 and gap_ci_eod.get('gap_ci_lower', 0.0) > 0:
                    alert_msg = (
                        f"ALERT: {attr} retains TPR disparity after Equalized Odds "
                        f"(gap={tpr_gap_test_eod:.4f}, CI lower={gap_ci_eod['gap_ci_lower']:.4f})"
                    )
                    logger.warning(alert_msg)
                    mitigation_results['alerts'].append({
                        'attribute': attr,
                        'stage': 'post_equalized_odds',
                        'message': alert_msg,
                        'tpr_gap': float(tpr_gap_test_eod),
                        'ci_lower': float(gap_ci_eod['gap_ci_lower'])
                    })

                final_stage_data = {
                    'stage': 'equalized_odds',
                    'val_df': val_eod_meta.copy(),
                    'test_df': test_eod_meta.copy(),
                    'tpr_gap_val': float(tpr_gap_val_eod),
                    'fpr_gap_val': float(fpr_gap_val_eod),
                    'tpr_gap_test': float(tpr_gap_test_eod),
                    'fpr_gap_test': float(fpr_gap_test_eod),
                    'gap_ci_test': gap_ci_eod
                }
            except Exception as exc:
                logger.error("Stage 2 failed for %s: %s", attr, exc)
                mitigation_results['stages_skipped'].append({
                    'attribute': attr,
                    'stage': 'equalized_odds',
                    'reason': f'error: {exc}'
                })
        elif not eligible_eod:
            mitigation_results['stages_skipped'].append({
                'attribute': attr,
                'stage': 'equalized_odds',
                'reason': 'insufficient_data',
                'support_info': support_val
            })
        elif final_stage_data is None:
            mitigation_results['stages_skipped'].append({
                'attribute': attr,
                'stage': 'equalized_odds',
                'reason': 'stage1_not_applied'
            })

        if final_stage_data:
            mitigation_results['post_mitigation_val'][attr] = {
                'stage': final_stage_data['stage'],
                'tpr_gap': final_stage_data['tpr_gap_val'],
                'fpr_gap': final_stage_data['fpr_gap_val'],
                'metrics': final_stage_data['val_df'].to_dict('records')
            }
            mitigation_results['post_mitigation_test'][attr] = {
                'stage': final_stage_data['stage'],
                'tpr_gap': final_stage_data['tpr_gap_test'],
                'fpr_gap': final_stage_data['fpr_gap_test'],
                'tpr_gap_ci': final_stage_data['gap_ci_test'],
                'metrics': final_stage_data['test_df'].to_dict('records')
            }
        else:
            mitigation_results['post_mitigation_val'][attr] = {
                'stage': 'not_applied',
                'reason': last_failure_reason or 'not_applicable'
            }
            mitigation_results['post_mitigation_test'][attr] = {
                'stage': 'not_applied',
                'reason': last_failure_reason or 'not_applicable'
            }

    if all_post_val_rows:
        post_val_df = pd.concat(all_post_val_rows, ignore_index=True)
        output_path = results_path / "fairness_post_val.csv"
        post_val_df.to_csv(output_path, index=False)
        logger.info("Saved post-mitigation validation metrics to %s", output_path)

    if all_post_test_rows:
        post_test_df = pd.concat(all_post_test_rows, ignore_index=True)
        output_path = results_path / "fairness_post_test.csv"
        post_test_df.to_csv(output_path, index=False)
        logger.info("Saved post-mitigation test metrics to %s", output_path)

    return mitigation_results


def generate_fairness_report(
    baseline_val: Dict,
    baseline_test: Dict,
    mitigation: Dict,
    results_path: Path = Path("results")
) -> Dict:
    report = {
        'baseline': {
            'validation': baseline_val,
            'test': baseline_test
        },
        'mitigation': mitigation,
        'summary': {
            'total_alerts': len(mitigation.get('alerts', [])),
            'stages_applied': len(mitigation.get('stages_applied', [])),
            'stages_skipped': len(mitigation.get('stages_skipped', []))
        }
    }

    output_path = results_path / "fairness_audit.json"
    output_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding='utf-8')
    logger.info("Saved fairness audit report to %s", output_path)
    return report

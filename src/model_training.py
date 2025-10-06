"""
Enhanced model training module with comprehensive outputs
"""

import numpy as np
import pandas as pd
from typing import Protocol, runtime_checkable, Any, Dict, List, Tuple, cast
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, RandomizedSearchCV, cross_validate, cross_val_predict
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from .config import SEED, ADVANCED_LIBS
from .evaluation import optimize_thresholds_multiobjective, summarize_threshold_performance
import time


@runtime_checkable
class ProbabilityEstimator(Protocol):
    """Minimal protocol for estimators that expose fit/predict_proba."""

    def fit(self, X: Any, y: Any) -> "ProbabilityEstimator": ...

    def predict_proba(self, X: Any) -> np.ndarray: ...

# Import advanced libraries with proper handling
if ADVANCED_LIBS:
    try:
        import xgboost as xgb
    except ImportError:
        xgb = None
    
    try:
        import lightgbm as lgb
    except ImportError:
        lgb = None
else:
    xgb = None
    lgb = None

def train_model_suite(
    X_train,
    y_train,
    X_val,
    y_val,
    preprocessor,
    cv_folds: int = 5,
    cv_repeats: int = 2,
    search_iterations: int = 15,
    include_alternative_models: bool = False,
    models_subset: List[str] | None = None,
):
    """
    Trains multiple models with comprehensive cross-validation and detailed outputs
    
    Returns:
    --------
    results : dict
        Complete results for each model
    ranking : list
        Models ranked by validation PR-AUC
    summary_df : pd.DataFrame
        Summary table of all metrics
    """
    
    print("=" * 80)
    print("🤖 INICIANDO TREINAMENTO DE MODELOS")
    print("=" * 80)
    
    results = {}
    cv = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=SEED)
    y_train_array = np.asarray(y_train)

    def _slice(data, indices):
        if hasattr(data, "iloc"):
            return data.iloc[indices]
        try:
            return data[indices]
        except (TypeError, KeyError):
            return [data[i] for i in indices]
    
    # Define all metrics to track
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision',
        'balanced_acc': 'balanced_accuracy',
        'recall': 'recall',
        'precision': 'precision',
        'f1': 'f1'
    }
    
    # ========== MODEL DEFINITIONS ==========
    model_registry: Dict[str, ProbabilityEstimator] = {}
    
    # Logistic Regression (baseline – always available)
    model_registry['logistic_l2'] = cast(ProbabilityEstimator, ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=SEED, k_neighbors=3)),
        ('clf', LogisticRegression(
            penalty='elasticnet',
            C=1.0,
            l1_ratio=0.0,
            solver='saga',
            max_iter=2000,
            class_weight='balanced',
            random_state=SEED
        ))
    ]))
    
    # Optional alternatives
    model_registry['gradient_boosting'] = cast(ProbabilityEstimator, ImbPipeline([
        ('prep', preprocessor),
        ('smote', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
        ('clf', GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            random_state=SEED
        ))
    ]))
    
    model_registry['random_forest'] = cast(ProbabilityEstimator, ImbPipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=5,
            class_weight='balanced_subsample',
            random_state=SEED,
            n_jobs=-1
        ))
    ]))
    
    if ADVANCED_LIBS:
        if xgb is not None:
            try:
                model_registry['xgboost'] = cast(ProbabilityEstimator, ImbPipeline([
                    ('prep', preprocessor),
                    ('smote', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
                    ('clf', xgb.XGBClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=6,
                        subsample=0.8,
                        scale_pos_weight=19,
                        random_state=SEED,
                        n_jobs=-1,
                        eval_metric='logloss'
                    ))
                ]))
            except Exception as e:
                print(f"⚠️ Erro ao configurar XGBoost: {e}")
        
        if lgb is not None:
            try:
                model_registry['lightgbm'] = cast(ProbabilityEstimator, ImbPipeline([
                    ('prep', preprocessor),
                    ('smote', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
                    ('clf', lgb.LGBMClassifier(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=8,
                        num_leaves=31,
                        class_weight='balanced',
                        random_state=SEED,
                        n_jobs=-1,
                        verbose=-1
                    ))
                ]))
            except Exception as e:
                print(f"⚠️ Erro ao configurar LightGBM: {e}")
    
    if models_subset is not None:
        selected = [name for name in models_subset if name in model_registry]
    elif include_alternative_models:
        selected = list(model_registry.keys())
    else:
        selected = ['logistic_l2']
    
    models: Dict[str, ProbabilityEstimator] = {name: model_registry[name] for name in selected}
    
    if not models:
        raise ValueError("Nenhum modelo válido foi selecionado para o treinamento.")
    
    print(f"\n📊 Treinando {len(models)} modelos com {cv_folds}-fold CV (repeats={cv_repeats})...\n")

    # Hyperparameter search spaces
    search_spaces = {
        'logistic_l2': {
            'clf__C': np.logspace(-3, 2, 30),
            'clf__l1_ratio': np.linspace(0.0, 0.6, 7)
        },
        'gradient_boosting': {
            'clf__n_estimators': [150, 250, 350, 450],
            'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
            'clf__max_depth': [2, 3, 4],
            'clf__subsample': [0.6, 0.8, 1.0]
        },
        'random_forest': {
            'clf__n_estimators': [300, 500, 700, 900],
            'clf__max_depth': [8, 12, 16, None],
            'clf__min_samples_split': [2, 4, 6, 8],
            'clf__min_samples_leaf': [1, 2, 4, 8]
        }
    }

    if ADVANCED_LIBS:
        if xgb is not None and 'xgboost' in models:
            search_spaces['xgboost'] = {
                'clf__n_estimators': [200, 300, 400],
                'clf__max_depth': [3, 4, 5, 6],
                'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
                'clf__subsample': [0.6, 0.8, 1.0],
                'clf__colsample_bytree': [0.6, 0.8, 1.0],
                'clf__scale_pos_weight': [10, 15, 20, 25]
            }
        if lgb is not None and 'lightgbm' in models:
            search_spaces['lightgbm'] = {
                'clf__n_estimators': [200, 300, 400],
                'clf__learning_rate': [0.01, 0.03, 0.05, 0.1],
                'clf__num_leaves': [15, 31, 63],
                'clf__max_depth': [-1, 8, 12],
                'clf__subsample': [0.6, 0.8, 1.0],
                'clf__colsample_bytree': [0.6, 0.8, 1.0],
                'clf__scale_pos_weight': [10, 15, 20, 25]
            }

    search_spaces = {k: v for k, v in search_spaces.items() if k in models}

    per_model_iterations = {
        'lightgbm': max(5, min(search_iterations, 8)),
        'xgboost': max(5, min(search_iterations, 10)),
        'random_forest': max(8, min(search_iterations, 12)),
        'gradient_boosting': max(8, min(search_iterations, 12))
    }
    per_model_iterations = {k: v for k, v in per_model_iterations.items() if k in models}
    
    # ========== TRAIN EACH MODEL ==========
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"🔧 Modelo: {name.upper()}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            tuned_model: ProbabilityEstimator = model
            search_summary: Dict[str, Any] | None = None
            
            if name in search_spaces:
                effective_iters = per_model_iterations.get(name, search_iterations)
                print(f"\n🔍 Busca de hiperparâmetros (RandomizedSearchCV, n_iter={effective_iters})")
                search_cv = StratifiedKFold(
                    n_splits=min(cv_folds, 5),
                    shuffle=True,
                    random_state=SEED
                )
                search_start = time.time()
                search = RandomizedSearchCV(
                    estimator=cast(Any, model),
                    param_distributions=search_spaces[name],
                    n_iter=effective_iters,
                    scoring='average_precision',
                    cv=search_cv,
                    n_jobs=-1,
                    random_state=SEED,
                    refit=True,
                    return_train_score=True,
                    verbose=0
                )
                search.fit(X_train, y_train)
                tuned_model = cast(ProbabilityEstimator, search.best_estimator_)
                search_summary = {
                    'best_score': search.best_score_,
                    'best_params': search.best_params_,
                    'duration': time.time() - search_start
                }
                print(f"   ✅ Melhor PR-AUC médio: {search_summary['best_score']:.4f}")
                print(f"   🧮 Hiperparâmetros: {search_summary['best_params']}")
                print(f"   ⏱️ Tempo de busca: {search_summary['duration']:.2f}s")
            
            print(f"\n⏳ Executando {cv_folds}-fold cross-validation (repeats={cv_repeats})...")
            cv_eval = RepeatedStratifiedKFold(n_splits=cv_folds, n_repeats=cv_repeats, random_state=SEED)
            cv_results = cross_validate(
                cast(Any, tuned_model), X_train, y_train,
                cv=cv_eval,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1
            )
            
            cv_stats = {}
            for metric in scoring.keys():
                train_scores = cv_results[f'train_{metric}']
                test_scores = cv_results[f'test_{metric}']
                
                cv_stats[metric] = {
                    'train_mean': train_scores.mean(),
                    'train_std': train_scores.std(),
                    'test_mean': test_scores.mean(),
                    'test_std': test_scores.std(),
                    'train_scores': train_scores.tolist(),
                    'test_scores': test_scores.tolist()
                }
            
            # Print CV results
            print(f"\n📊 Resultados do Cross-Validation:")
            print(f"{'Métrica':<15} {'Train Mean':<12} {'Train Std':<12} {'Test Mean':<12} {'Test Std':<12}")
            print("-" * 80)
            for metric_name in scoring.keys():
                stats = cv_stats[metric_name]
                print(f"{metric_name:<15} {stats['train_mean']:>11.4f} {stats['train_std']:>11.4f} "
                      f"{stats['test_mean']:>11.4f} {stats['test_std']:>11.4f}")
            
            # ===== FIT ON FULL TRAINING SET =====
            print(f"\n⏳ Treinando no dataset completo...")
            tuned_model.fit(X_train, y_train)
            
            # ===== VALIDATION SET EVALUATION =====
            print(f"\n⏳ Avaliando no validation set...")
            
            y_proba_val = tuned_model.predict_proba(X_val)[:, 1]
            y_pred_val = (y_proba_val >= 0.5).astype(int)
            
            val_metrics = {
                'roc_auc': roc_auc_score(y_val, y_proba_val),
                'pr_auc': average_precision_score(y_val, y_proba_val),
                'balanced_acc': balanced_accuracy_score(y_val, y_pred_val)
            }
            
            # ===== THRESHOLD SEARCH (VALIDAÇÃO) =====
            threshold_summary = optimize_thresholds_multiobjective(
                y_val,
                y_proba_val,
                betas=(1.0, 2.0),
                min_precision=0.12,
                min_recall=0.40,
                max_threshold=0.30
            )
            
            validation_threshold = None
            validation_threshold_metrics: Dict[str, Any] | None = None
            if threshold_summary and threshold_summary['best_by_beta']:
                if 2.0 in threshold_summary['best_by_beta']:
                    validation_threshold_metrics = threshold_summary['best_by_beta'][2.0]
                elif 1.0 in threshold_summary['best_by_beta']:
                    validation_threshold_metrics = threshold_summary['best_by_beta'][1.0]
                else:
                    validation_threshold_metrics = next(iter(threshold_summary['best_by_beta'].values()))
            
            if validation_threshold_metrics is not None:
                validation_threshold = float(validation_threshold_metrics['threshold'])
                print(
                    f"\n✅ Limiar sugerido (validação): {validation_threshold:.3f} "
                    f"(precision={validation_threshold_metrics['precision']:.3f}, "
                    f"recall={validation_threshold_metrics['recall']:.3f})"
                )
            else:
                validation_threshold = 0.5
                print("\n⚠️ Nenhum limiar válido encontrado na validação - usando 0.500 por padrão.")
            
            # ===== THRESHOLD SEARCH (OUT-OF-FOLD) =====
            oof_threshold_summary = None
            oof_threshold = None
            oof_threshold_metrics: Dict[str, Any] | None = None
            oof_proba = None
            try:
                oof_cv = StratifiedKFold(
                    n_splits=min(cv_folds, 5),
                    shuffle=True,
                    random_state=SEED
                )
                oof_proba = np.zeros(len(y_train_array), dtype=float)
                
                for fold_idx, (train_idx, val_idx) in enumerate(oof_cv.split(X_train, y_train)):
                    model_fold = clone(tuned_model)
                    X_tr = _slice(X_train, train_idx)
                    y_tr = _slice(y_train, train_idx)
                    X_val_fold = _slice(X_train, val_idx)
                    
                    model_fold.fit(X_tr, y_tr)
                    oof_proba[val_idx] = model_fold.predict_proba(X_val_fold)[:, 1]
                
                oof_threshold_summary = optimize_thresholds_multiobjective(
                    y_train,
                    oof_proba,
                    betas=(1.0, 2.0),
                    min_precision=0.12,
                    min_recall=0.40,
                    max_threshold=0.30
                )
                
                if oof_threshold_summary['best_by_beta']:
                    if 2.0 in oof_threshold_summary['best_by_beta']:
                        oof_threshold_metrics = oof_threshold_summary['best_by_beta'][2.0]
                    elif 1.0 in oof_threshold_summary['best_by_beta']:
                        oof_threshold_metrics = oof_threshold_summary['best_by_beta'][1.0]
                    else:
                        oof_threshold_metrics = next(iter(oof_threshold_summary['best_by_beta'].values()))
                    
                    if oof_threshold_metrics is not None:
                        oof_threshold = float(oof_threshold_metrics['threshold'])
                        print(
                            f"🔁 Limiar sugerido (OOF): {oof_threshold:.3f} "
                            f"(precision={oof_threshold_metrics['precision']:.3f}, "
                            f"recall={oof_threshold_metrics['recall']:.3f})"
                        )
            except Exception as exc:
                print(f"⚠️ Não foi possível gerar previsões OOF para limiarização: {exc}")
                oof_proba = None
            
            # ===== CONSOLIDAR LIMIAR DE PRODUÇÃO =====
            candidate_thresholds: List[Tuple[str, float, Dict[str, Any] | None]] = []
            if validation_threshold is not None:
                candidate_thresholds.append(('validation_fbeta', validation_threshold, validation_threshold_metrics))
            if oof_threshold is not None:
                candidate_thresholds.append(('oof_fbeta', oof_threshold, oof_threshold_metrics))
            
            if candidate_thresholds:
                production_source, production_threshold, ref_metrics = min(candidate_thresholds, key=lambda x: x[1])
            else:
                production_source, production_threshold, ref_metrics = ('default', 0.5, None)
            
            production_metrics_val = summarize_threshold_performance(
                y_val, y_proba_val, production_threshold
            )
            production_metrics_oof = None
            if oof_proba is not None:
                production_metrics_oof = summarize_threshold_performance(
                    y_train, oof_proba, production_threshold
                )
            
            print(
                f"\n✅ Limiar de produção definido: {production_threshold:.3f} "
                f"(fonte: {production_source})"
            )
            print(
                f"   → Validação: precision={production_metrics_val['precision']:.3f} | "
                f"recall={production_metrics_val['recall']:.3f} | "
                f"bal_acc={production_metrics_val['balanced_accuracy']:.3f}"
            )
            if production_metrics_oof is not None:
                print(
                    f"   → OOF: precision={production_metrics_oof['precision']:.3f} | "
                    f"recall={production_metrics_oof['recall']:.3f} | "
                    f"bal_acc={production_metrics_oof['balanced_accuracy']:.3f}"
                )
            
            def _serialize_threshold_metrics(metrics_dict: Dict[str, Any] | None):
                if metrics_dict is None:
                    return None
                sanitized = dict(metrics_dict)
                cm = sanitized.get('confusion_matrix')
                if cm is not None:
                    sanitized['confusion_matrix'] = np.asarray(cm).tolist()
                return sanitized
            
            production_metrics_val_serial = _serialize_threshold_metrics(production_metrics_val)
            production_metrics_oof_serial = _serialize_threshold_metrics(production_metrics_oof)
            
            # Calculate overfitting metrics
            overfitting_gap = {
                metric: abs(cv_stats[metric]['train_mean'] - val_metrics.get(metric, cv_stats[metric]['test_mean']))
                for metric in ['roc_auc', 'pr_auc']
            }
            
            # Store results
            elapsed = time.time() - start_time
            
            results[name] = {
                'model': tuned_model,
                'cv_stats': cv_stats,
                'val_metrics': val_metrics,
                'y_proba': y_proba_val,
                'overfitting_gap': overfitting_gap,
                'training_time_seconds': elapsed,
                'search': search_summary,
                'thresholds': {
                    'validation': threshold_summary,
                    'oof': oof_threshold_summary
                },
                'validation_threshold': validation_threshold,
                'validation_threshold_metrics': validation_threshold_metrics,
                'oof_threshold': oof_threshold,
                'oof_threshold_metrics': oof_threshold_metrics,
                'production_threshold': production_threshold,
                'production_threshold_source': production_source,
                'production_threshold_metrics_val': production_metrics_val_serial,
                'production_threshold_metrics_oof': production_metrics_oof_serial
            }
            
            # Print validation results
            print(f"\n📊 Métricas no Validation Set:")
            print(f"{'Métrica':<15} {'Valor':<12}")
            print("-" * 30)
            for metric_name, metric_value in val_metrics.items():
                print(f"{metric_name:<15} {metric_value:>11.4f}")
            
            if threshold_summary and threshold_summary['best_by_beta']:
                print(f"\n🎯 Sugestões de limiar (validação):")
                for beta, info in threshold_summary['best_by_beta'].items():
                    print(
                        f"   F{beta:.1f} -> thr={info['threshold']:.3f} | "
                        f"prec={info['precision']:.3f} | rec={info['recall']:.3f} | "
                        f"bal_acc={info['balanced_accuracy']:.3f}"
                    )
            
            # Overfitting check
            print(f"\n⚠️ Análise de Overfitting:")
            for metric, gap in overfitting_gap.items():
                status = "✅ OK" if gap < 0.05 else "⚠️ ATENÇÃO" if gap < 0.10 else "❌ CRÍTICO"
                print(f"   {metric.upper()} gap: {gap:.4f} {status}")
            
            print(f"\n⏱️ Tempo de treinamento: {elapsed:.2f}s")
            print(f"{'='*80}\n")
            
        except Exception as e:
            print(f"❌ Erro ao treinar {name}: {e}")
            continue
    
    # ========== RANKING ==========
    valid_results = {k: v for k, v in results.items() if 'val_metrics' in v}
    ranking = sorted(valid_results.items(), 
                    key=lambda x: x[1]['val_metrics']['pr_auc'], 
                    reverse=True)
    
    print(f"\n{'='*80}")
    print(f"🏆 RANKING FINAL (por PR-AUC no Validation Set)")
    print(f"{'='*80}\n")
    
    print(f"{'Rank':<6} {'Modelo':<20} {'PR-AUC':<10} {'ROC-AUC':<10} {'Bal. Acc':<12} {'Overfit Gap':<12}")
    print("-" * 80)
    
    for i, (name, result) in enumerate(ranking, 1):
        pr_auc = result['val_metrics']['pr_auc']
        roc_auc = result['val_metrics']['roc_auc']
        bal_acc = result['val_metrics']['balanced_acc']
        gap = result['overfitting_gap']['pr_auc']
        
        print(f"{i:<6} {name:<20} {pr_auc:>9.4f} {roc_auc:>9.4f} {bal_acc:>11.4f} {gap:>11.4f}")
    
    # ========== SUMMARY DATAFRAME ==========
    summary_data = []
    
    for name, result in results.items():
        if 'val_metrics' not in result:
            continue
        
        cv_stats = result['cv_stats']
        val_metrics = result['val_metrics']
        
        row = {
            'Model': name,
            'CV_PR-AUC_Mean': cv_stats['pr_auc']['test_mean'],
            'CV_PR-AUC_Std': cv_stats['pr_auc']['test_std'],
            'CV_ROC-AUC_Mean': cv_stats['roc_auc']['test_mean'],
            'CV_ROC-AUC_Std': cv_stats['roc_auc']['test_std'],
            'Val_PR-AUC': val_metrics['pr_auc'],
            'Val_ROC-AUC': val_metrics['roc_auc'],
            'Val_Balanced_Acc': val_metrics['balanced_acc'],
            'PR-AUC_Overfit_Gap': result['overfitting_gap']['pr_auc'],
            'Training_Time_s': result['training_time_seconds']
        }

        if result.get('production_threshold') is not None:
            row['Production_Threshold'] = result['production_threshold']
            row['Production_Threshold_Source'] = result.get('production_threshold_source')
            prod_metrics = result.get('production_threshold_metrics_val') or {}
            if prod_metrics:
                row['Production_Precision_Val'] = prod_metrics.get('precision')
                row['Production_Recall_Val'] = prod_metrics.get('recall')
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Val_PR-AUC', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*80}")
    print(f"📋 TABELA RESUMO DE TODOS OS MODELOS")
    print(f"{'='*80}\n")
    
    print(summary_df.to_string(index=False))
    
    return results, ranking, summary_df

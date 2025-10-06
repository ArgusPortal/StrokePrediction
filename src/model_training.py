"""
Enhanced model training module with comprehensive outputs
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import make_scorer, roc_auc_score, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from .config import SEED, ADVANCED_LIBS, MODEL_REGISTRY
import time

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

def train_model_suite(X_train, y_train, X_val, y_val, preprocessor, cv_folds=10):
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
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
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
    models = {}
    
    # Logistic Regression
    models['logistic_l2'] = ImbPipeline([
        ('prep', preprocessor),
        ('smote', SMOTE(random_state=SEED, k_neighbors=3)),
        ('clf', LogisticRegression(penalty='l2', C=1.0, max_iter=1000,
                                   class_weight='balanced', random_state=SEED))
    ])
    
    # Gradient Boosting
    models['gradient_boosting'] = ImbPipeline([
        ('prep', preprocessor),
        ('smote', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
        ('clf', GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=6, subsample=0.8, random_state=SEED))
    ])
    
    # Random Forest
    models['random_forest'] = ImbPipeline([
        ('prep', preprocessor),
        ('clf', RandomForestClassifier(n_estimators=500, max_depth=15, 
                                       min_samples_split=5, class_weight='balanced_subsample',
                                       random_state=SEED, n_jobs=-1))
    ])
    
    # Advanced models if available
    if ADVANCED_LIBS:
        if xgb is not None:
            try:
                models['xgboost'] = ImbPipeline([
                    ('prep', preprocessor),
                    ('smote', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
                    ('clf', xgb.XGBClassifier(n_estimators=300, learning_rate=0.05, max_depth=6,
                                             subsample=0.8, scale_pos_weight=19,
                                             random_state=SEED, n_jobs=-1, eval_metric='logloss'))
                ])
            except Exception as e:
                print(f"⚠️ Erro ao configurar XGBoost: {e}")
        
        if lgb is not None:
            try:
                models['lightgbm'] = ImbPipeline([
                    ('prep', preprocessor),
                    ('smote', BorderlineSMOTE(random_state=SEED, k_neighbors=3)),
                    ('clf', lgb.LGBMClassifier(n_estimators=300, learning_rate=0.05, max_depth=8,
                                              num_leaves=31, class_weight='balanced',
                                              random_state=SEED, n_jobs=-1, verbose=-1))
                ])
            except Exception as e:
                print(f"⚠️ Erro ao configurar LightGBM: {e}")
    
    print(f"\n📊 Treinando {len(models)} modelos com {cv_folds}-fold CV...\n")
    
    # ========== TRAIN EACH MODEL ==========
    for name, model in models.items():
        print(f"\n{'='*80}")
        print(f"🔧 Modelo: {name.upper()}")
        print(f"{'='*80}")
        
        start_time = time.time()
        
        try:
            # ===== CROSS-VALIDATION =====
            print(f"\n⏳ Executando {cv_folds}-fold cross-validation...")
            
            cv_results = cross_validate(
                model, X_train, y_train,
                cv=cv,
                scoring=scoring,
                return_train_score=True,
                n_jobs=-1,
                verbose=0
            )
            
            # Calculate CV statistics
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
            for metric in scoring.keys():
                stats = cv_stats[metric]
                print(f"{metric:<15} {stats['train_mean']:>11.4f} {stats['train_std']:>11.4f} "
                      f"{stats['test_mean']:>11.4f} {stats['test_std']:>11.4f}")
            
            # ===== FIT ON FULL TRAINING SET =====
            print(f"\n⏳ Treinando no dataset completo...")
            model.fit(X_train, y_train)
            
            # ===== VALIDATION SET EVALUATION =====
            print(f"\n⏳ Avaliando no validation set...")
            
            y_proba_val = model.predict_proba(X_val)[:, 1]
            y_pred_val = (y_proba_val >= 0.5).astype(int)
            
            val_metrics = {
                'roc_auc': roc_auc_score(y_val, y_proba_val),
                'pr_auc': average_precision_score(y_val, y_proba_val),
                'balanced_acc': (y_pred_val == y_val).mean()  # Simplified
            }
            
            # Calculate overfitting metrics
            overfitting_gap = {
                metric: abs(cv_stats[metric]['train_mean'] - val_metrics.get(metric, cv_stats[metric]['test_mean']))
                for metric in ['roc_auc', 'pr_auc']
            }
            
            # Store results
            elapsed = time.time() - start_time
            
            results[name] = {
                'model': model,
                'cv_stats': cv_stats,
                'val_metrics': val_metrics,
                'y_proba': y_proba_val,
                'overfitting_gap': overfitting_gap,
                'training_time_seconds': elapsed
            }
            
            # Print validation results
            print(f"\n📊 Métricas no Validation Set:")
            print(f"{'Métrica':<15} {'Valor':<12}")
            print("-" * 30)
            for metric, value in val_metrics.items():
                print(f"{metric:<15} {value:>11.4f}")
            
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
        
        summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df = summary_df.sort_values('Val_PR-AUC', ascending=False).reset_index(drop=True)
    
    print(f"\n{'='*80}")
    print(f"📋 TABELA RESUMO DE TODOS OS MODELOS")
    print(f"{'='*80}\n")
    
    print(summary_df.to_string(index=False))
    
    return results, ranking, summary_df

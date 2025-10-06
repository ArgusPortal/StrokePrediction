"""
Advanced calibration module for stroke prediction
Implements multiple calibration methods with rigorous validation
"""

import logging
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Literal
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss
from sklearn.model_selection import StratifiedKFold

from .config import RESULTS_PATH, SEED

logger = logging.getLogger(__name__)


def expected_calibration_error(y_true, y_proba, n_bins=10):
    """
    Calculate Expected Calibration Error (ECE)
    
    Target: ECE < 0.05 (excellent calibration)
    """
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    # Calculate bin counts for weighted ECE
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_proba, bins[1:-1])
    bin_counts = np.bincount(bin_indices, minlength=n_bins)
    
    # Weighted ECE
    ece = np.sum(bin_counts * np.abs(fraction_of_positives - mean_predicted_value)) / len(y_true)
    
    return ece


def brier_skill_score(y_true, y_proba):
    """
    Calculate Brier Skill Score (BSS)
    
    BSS > 0 means better than baseline (always predict prevalence)
    Target: BSS > 0.10
    """
    brier_model = brier_score_loss(y_true, y_proba)
    brier_baseline = brier_score_loss(y_true, np.full_like(y_proba, y_true.mean()))
    
    bss = 1 - (brier_model / brier_baseline)
    
    return bss, brier_model, brier_baseline


def calibrate_model_comprehensive(model, X_train, y_train, X_val, y_val, cv_folds=10):
    """
    Comprehensive calibration with multiple methods + validation
    
    Returns best calibrated model based on ECE + BSS
    """
    
    print("\n" + "="*80)
    print("🔬 RECALIBRAÇÃO AVANÇADA - CORRIGINDO ECE CRÍTICO")
    print("="*80)
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    
    calibration_methods: dict[str, Literal['sigmoid', 'isotonic']] = {
        'sigmoid': 'sigmoid',  # Platt scaling
        'isotonic': 'isotonic'  # Isotonic regression
    }
    
    results = {}
    results = {}
    
    # 1. Baseline (uncalibrated)
    print("\n📊 Baseline (Sem Calibração):")
    y_proba_uncal = model.predict_proba(X_val)[:, 1]
    ece_uncal = expected_calibration_error(y_val, y_proba_uncal)
    bss_uncal, brier_uncal, brier_base = brier_skill_score(y_val, y_proba_uncal)
    
    print(f"   ECE: {ece_uncal:.4f} {'❌ CRÍTICO' if ece_uncal > 0.05 else '✅ OK'}")
    print(f"   Brier Score: {brier_uncal:.4f}")
    print(f"   Brier Skill Score: {bss_uncal:.4f} {'❌ PIOR QUE BASELINE' if bss_uncal < 0 else '✅ OK'}")
    
    results['uncalibrated'] = {
        'model': model,
        'ece': ece_uncal,
        'brier': brier_uncal,
        'bss': bss_uncal
    }
    
    # 2. Test calibration methods
    for method_name, method in calibration_methods.items():
        print(f"\n🔧 Testando: {method_name.upper()}")
        
        try:
            # Calibrate with cross-validation
            calibrated_model = CalibratedClassifierCV(
                model, 
                method=method, 
                cv=cv,
                n_jobs=-1
            )
            
            calibrated_model.fit(X_train, y_train)
            
            # Evaluate on validation
            y_proba_cal = calibrated_model.predict_proba(X_val)[:, 1]
            
            ece_cal = expected_calibration_error(y_val, y_proba_cal)
            bss_cal, brier_cal, _ = brier_skill_score(y_val, y_proba_cal)
            
            print(f"   ECE: {ece_cal:.4f} {'✅ EXCELENTE' if ece_cal < 0.05 else '⚠️ ATENÇÃO' if ece_cal < 0.10 else '❌ RUIM'}")
            print(f"   Brier Score: {brier_cal:.4f}")
            print(f"   Brier Skill Score: {bss_cal:.4f} {'✅ OK' if bss_cal > 0.10 else '⚠️ FRACO' if bss_cal > 0 else '❌ CRÍTICO'}")
            
            # Calculate improvement
            ece_improvement = ((ece_uncal - ece_cal) / ece_uncal) * 100
            bss_improvement = bss_cal - bss_uncal
            
            print(f"   Melhoria ECE: {ece_improvement:+.1f}%")
            print(f"   Melhoria BSS: {bss_improvement:+.4f}")
            
            results[method_name] = {
                'model': calibrated_model,
                'ece': ece_cal,
                'brier': brier_cal,
                'bss': bss_cal,
                'ece_improvement': ece_improvement,
                'bss_improvement': bss_improvement
            }
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
    
    # 3. Select best method
    print("\n" + "="*80)
    print("🏆 SELEÇÃO DO MELHOR MÉTODO")
    print("="*80)
    
    # Ranking por ECE (objetivo primário)
    valid_methods = {k: v for k, v in results.items() if k != 'uncalibrated'}
    
    if not valid_methods:
        print("❌ Nenhum método de calibração funcionou - usando modelo original")
        return model, results['uncalibrated']
    
    best_method = min(valid_methods.items(), key=lambda x: x[1]['ece'])
    best_name, best_result = best_method
    
    print(f"\n🥇 VENCEDOR: {best_name.upper()}")
    print(f"   ECE: {best_result['ece']:.4f} (target: <0.05)")
    print(f"   BSS: {best_result['bss']:.4f} (target: >0.10)")
    print(f"   Brier: {best_result['brier']:.4f}")
    
    # Check if meets requirements
    meets_ece = best_result['ece'] < 0.05
    meets_bss = best_result['bss'] > 0
    
    print(f"\n✅ STATUS DE COMPLIANCE:")
    print(f"   ECE < 0.05: {'✅ CONFORME' if meets_ece else '⚠️ NÃO CONFORME'}")
    print(f"   BSS > 0: {'✅ CONFORME' if meets_bss else '❌ NÃO CONFORME'}")
    
    if meets_ece and meets_bss:
        print(f"\n🎉 MODELO CALIBRADO APROVADO PARA PRODUÇÃO")
    else:
        print(f"\n⚠️ ATENÇÃO: Calibração não atingiu todos os targets")
    
    return best_result['model'], best_result


def analyze_calibration(model, X_val, y_val, n_bins=10):
    """
    Comprehensive calibration analysis
    
    Returns:
        dict: Calibration metrics and plots
    """
    
    logger.info("Analyzing model calibration...")
    
    # Get probabilities
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Calibration curve
    fraction_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=n_bins)
    
    # Expected Calibration Error (ECE)
    ece = np.mean(np.abs(fraction_pos - mean_pred))
    
    # Brier score
    brier = brier_score_loss(y_val, y_proba)
    
    # Brier skill score (vs always predicting mean)
    brier_null = brier_score_loss(y_val, np.full_like(y_proba, y_val.mean()))
    brier_skill = 1 - (brier / brier_null)
    
    results = {
        'ece': ece,
        'brier_score': brier,
        'brier_skill': brier_skill,
        'calibration_curve': {
            'fraction_positives': fraction_pos,
            'mean_predicted': mean_pred
        }
    }
    
    logger.info(f"ECE: {ece:.4f}, Brier: {brier:.4f}, Skill: {brier_skill:.4f}")
    
    return results


def plot_calibration_curve(model, X_val, y_val, model_name='Model'):
    """
    Plot comprehensive calibration analysis
    """
    
    y_proba = model.predict_proba(X_val)[:, 1]
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. Calibration curve
    ax1 = axes[0]
    fraction_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10)
    
    ax1.plot(mean_pred, fraction_pos, 's-', linewidth=2, markersize=8, 
            label=model_name, color='#3498db')
    ax1.plot([0, 1], [0, 1], 'k--', label='Perfeitamente Calibrado', alpha=0.7)
    
    ax1.set_xlabel('Probabilidade Predita Média', fontweight='bold')
    ax1.set_ylabel('Fração de Positivos', fontweight='bold')
    ax1.set_title('Curva de Calibração', fontweight='bold', fontsize=14)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Add ECE annotation
    ece = expected_calibration_error(y_val, y_proba)
    ax1.text(0.02, 0.98, f'ECE = {ece:.4f}', 
            transform=ax1.transAxes, fontweight='bold', fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='yellow' if ece < 0.05 else 'red', alpha=0.5))
    
    # 2. Reliability diagram (histogram)
    ax2 = axes[1]
    ax2.hist(y_proba, bins=50, alpha=0.7, color='#9b59b6', edgecolor='black')
    ax2.set_xlabel('Probabilidade Predita', fontweight='bold')
    ax2.set_ylabel('Frequência', fontweight='bold')
    ax2.set_title('Distribuição de Probabilidades', fontweight='bold', fontsize=14)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Brier score decomposition
    ax3 = axes[2]
    
    bss, brier_model, brier_baseline = brier_skill_score(y_val, y_proba)
    
    categories = ['Baseline\n(Prevalence)', 'Model']
    brier_values = [float(brier_baseline), float(brier_model)]
    colors_bars = ['#e74c3c', '#2ecc71' if brier_model < brier_baseline else '#e74c3c']
    
    bars = ax3.bar(categories, brier_values, color=colors_bars, alpha=0.8, edgecolor='black', linewidth=2)
    
    ax3.set_ylabel('Brier Score (Menor = Melhor)', fontweight='bold')
    ax3.set_title('Brier Score Comparison', fontweight='bold', fontsize=14)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Add values and BSS
    for bar, val in zip(bars, brier_values):
        ax3.text(bar.get_x() + bar.get_width()/2, val, 
                f'{val:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=11)
    
    ax3.text(0.5, max(brier_values) * 0.9, f'BSS = {bss:.4f}', 
            ha='center', fontweight='bold', fontsize=12,
            bbox=dict(boxstyle='round', facecolor='green' if bss > 0.10 else 'orange', alpha=0.5))
    
    plt.suptitle(f'ANÁLISE DE CALIBRAÇÃO - {model_name.upper()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f'calibration_analysis_{model_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ Curva de calibração salva: calibration_analysis_{model_name}.png")

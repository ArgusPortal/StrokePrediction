"""
Clinical utility analysis using Decision Curve Analysis (DCA)
Validates model usefulness vs. treat-all/treat-none strategies
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from .config import RESULTS_PATH

def calculate_net_benefit(y_true, y_proba, threshold):
    """
    Calculate net benefit at a given threshold
    
    Net Benefit = (TP / n) - (FP / n) * (threshold / (1 - threshold))
    
    Interpretation:
    - >0: Model adds value vs. treat none
    - Higher than treat-all: Model adds value vs. treating everyone
    """
    
    y_pred = (y_proba >= threshold).astype(int)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    n = len(y_true)
    
    # Net benefit formula
    net_benefit = (tp / n) - (fp / n) * (threshold / (1 - threshold))
    
    return net_benefit

def decision_curve_analysis(y_true, y_proba, threshold_range=None, model_name='Model'):
    """
    Comprehensive Decision Curve Analysis
    
    Compares model against:
    - Treat None (no intervention)
    - Treat All (intervene everyone)
    
    Returns:
        DataFrame: DCA results with net benefit curves
        float: Optimal threshold with max net benefit
    """
    
    print("\n" + "="*80)
    print("📈 DECISION CURVE ANALYSIS - VALIDAÇÃO DE UTILIDADE CLÍNICA")
    print("="*80)
    
    # FIX: Threshold range mais restrito (evitar extremos)
    if threshold_range is None:
        threshold_range = np.arange(0.05, 0.35, 0.01)  # ← 5% a 35% (não 1% a 50%)
    
    results = []
    
    # Calculate net benefit for each threshold
    for threshold in threshold_range:
        # Model net benefit
        nb_model = calculate_net_benefit(y_true, y_proba, threshold)
        
        # Treat All: assume everyone has condition
        prevalence = y_true.mean()
        nb_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        
        # Treat None: no intervention
        nb_none = 0
        
        # Calculate advantage
        advantage_vs_all = nb_model - nb_all
        advantage_vs_none = nb_model - nb_none
        
        results.append({
            'threshold': threshold,
            'model': nb_model,
            'treat_all': nb_all,
            'treat_none': nb_none,
            'advantage_vs_treat_all': advantage_vs_all,
            'advantage_vs_treat_none': advantage_vs_none
        })
    
    dca_df = pd.DataFrame(results)
    
    # Find optimal threshold (max net benefit)
    optimal_idx = dca_df['model'].idxmax()
    
    # FIX: Convert index to int explicitly and use safe indexing
    try:
        # Ensure optimal_idx is converted to int
        idx = int(optimal_idx) if pd.notna(optimal_idx) else 0
        
        # Use .iloc with guaranteed int index
        optimal_row = dca_df.iloc[idx]
        
        # Extract values directly from Series (no need for np.asarray)
        optimal_threshold = float(optimal_row['threshold'])
        optimal_nb = float(optimal_row['model'])
        
    except (ValueError, TypeError, IndexError, KeyError) as e:
        print(f"⚠️ Warning: Error extracting optimal values: {e}")
        # Fallback: use first valid threshold
        optimal_threshold = float(threshold_range[0])
        optimal_nb = float(dca_df['model'].iloc[0])
    
    # FIX: VALIDAR SE THRESHOLD É RAZOÁVEL
    if optimal_threshold < 0.05:
        print(f"\n⚠️ ATENÇÃO: Threshold muito baixo ({optimal_threshold:.3f})")
        print(f"   Isso pode gerar excesso de falsos positivos")
        print(f"   Considerar threshold mínimo de 0.08-0.10")
        
        # Forçar threshold mínimo
        optimal_threshold = max(optimal_threshold, 0.08)
        print(f"   → Ajustado para: {optimal_threshold:.3f}")
    
    # Find useful range (where model > treat all)
    useful_range = dca_df[dca_df['advantage_vs_treat_all'] > 0]
    
    print(f"\n🎯 THRESHOLD ÓTIMO:")
    print(f"   Threshold: {optimal_threshold:.3f}")
    print(f"   Net Benefit: {optimal_nb:.4f}")
    
    if len(useful_range) > 0:
        min_threshold = useful_range['threshold'].min()
        max_threshold = useful_range['threshold'].max()
        print(f"\n✅ RANGE DE UTILIDADE CLÍNICA:")
        print(f"   Modelo supera 'treat all' entre {min_threshold:.3f} e {max_threshold:.3f}")
        print(f"   Amplitude: {max_threshold - min_threshold:.3f}")
    else:
        print(f"\n❌ CRÍTICO: Modelo não adiciona valor clínico")
        print(f"   Revisar feature engineering ou dados")
    
    # Analyze specific clinical scenarios
    print(f"\n📋 ANÁLISE POR CENÁRIO CLÍNICO:")
    print("-" * 80)
    
    scenarios = [
        {'name': 'Conservador (25%)', 'threshold': 0.25},
        {'name': 'Agressivo (8%)', 'threshold': 0.08}
    ]
    
    for scenario in scenarios:
        t = scenario['threshold']
        closest_idx = (dca_df['threshold'] - t).abs().idxmin()
        row = dca_df.loc[closest_idx]
        
        print(f"\n{scenario['name']}:")
        print(f"   Threshold: {row['threshold']:.3f}")
        print(f"   Net Benefit (Modelo): {row['model']:.4f}")
        print(f"   Net Benefit (Treat All): {row['treat_all']:.4f}")
        print(f"   Vantagem: {row['advantage_vs_treat_all']:+.4f} {'✅ ÚTIL' if row['advantage_vs_treat_all'] > 0 else '❌ SEM VALOR'}")
    
    return dca_df, optimal_threshold


def plot_decision_curve(dca_df, model_name='Model'):
    """
    Plot Decision Curve Analysis
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # 1. Net Benefit Curves
    ax1 = axes[0]
    ax1.plot(dca_df['threshold'], dca_df['model'], 
            linewidth=3, label=model_name, color='#2ecc71')
    ax1.plot(dca_df['threshold'], dca_df['treat_all'], 
            '--', linewidth=2, label='Treat All', color='#e74c3c', alpha=0.7)
    ax1.axhline(y=0, color='black', linestyle='-', linewidth=1, 
                label='Treat None', alpha=0.5)
    
    ax1.set_xlabel('Threshold', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Net Benefit', fontweight='bold', fontsize=12)
    ax1.set_title('Decision Curve Analysis', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim(dca_df['threshold'].min(), dca_df['threshold'].max())
    
    # Highlight optimal threshold - FIX: Same robust extraction
    optimal_idx = dca_df['model'].idxmax()
    
    try:
        # Convert index to int explicitly
        idx = int(optimal_idx) if pd.notna(optimal_idx) else 0
        optimal_row = dca_df.iloc[idx]
        
        optimal_t = float(optimal_row['threshold'])
        optimal_nb = float(optimal_row['model'])
    except (ValueError, TypeError, IndexError, KeyError):
        # Fallback values
        optimal_t = float(dca_df['threshold'].iloc[0])
        optimal_nb = float(dca_df['model'].iloc[0])
    
    ax1.scatter([optimal_t], [optimal_nb], s=200, color='gold', 
                edgecolor='black', linewidth=2, zorder=5, 
                label=f'Ótimo: {optimal_t:.3f}')
    ax1.legend(loc='upper right', fontsize=11)
    
    # 2. Advantage over Treat All
    ax2 = axes[1]
    ax2.fill_between(dca_df['threshold'], 0, dca_df['advantage_vs_treat_all'],
                     where=(dca_df['advantage_vs_treat_all'] > 0),
                     color='#2ecc71', alpha=0.3, label='Útil')
    ax2.fill_between(dca_df['threshold'], 0, dca_df['advantage_vs_treat_all'],
                     where=(dca_df['advantage_vs_treat_all'] <= 0),
                     color='#e74c3c', alpha=0.3, label='Sem Valor')
    
    ax2.plot(dca_df['threshold'], dca_df['advantage_vs_treat_all'],
            linewidth=2, color='#3498db')
    ax2.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
    
    ax2.set_xlabel('Threshold', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Net Benefit (Model - Treat All)', fontweight='bold', fontsize=12)
    ax2.set_title('Clinical Utility vs. Treat All', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim(dca_df['threshold'].min(), dca_df['threshold'].max())
    
    plt.suptitle(f'DECISION CURVE ANALYSIS - {model_name.upper()}',
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f'decision_curve_{model_name}.png',
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Decision curve salva: decision_curve_{model_name}.png")

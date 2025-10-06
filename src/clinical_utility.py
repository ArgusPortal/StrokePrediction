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
    - Treat all (assume everyone positive)
    - Treat none (assume everyone negative)
    
    Returns threshold range where model adds clinical value
    """
    
    if threshold_range is None:
        threshold_range = np.arange(0.01, 0.51, 0.01)
    
    print("\n" + "="*80)
    print("📈 DECISION CURVE ANALYSIS - VALIDAÇÃO DE UTILIDADE CLÍNICA")
    print("="*80)
    
    results = []
    
    prevalence = y_true.mean()
    
    for threshold in threshold_range:
        # Model net benefit
        nb_model = calculate_net_benefit(y_true, y_proba, threshold)
        
        # Treat all net benefit
        nb_treat_all = prevalence - (1 - prevalence) * (threshold / (1 - threshold))
        
        # Treat none net benefit
        nb_treat_none = 0
        
        results.append({
            'threshold': threshold,
            'model': nb_model,
            'treat_all': nb_treat_all,
            'treat_none': nb_treat_none,
            'advantage_vs_treat_all': nb_model - nb_treat_all,
            'advantage_vs_treat_none': nb_model - nb_treat_none
        })
    
    dca_df = pd.DataFrame(results)
    
    # Find optimal threshold (max net benefit)
    optimal_idx = dca_df['model'].idxmax()
    optimal_threshold = dca_df.loc[optimal_idx, 'threshold']
    optimal_nb = dca_df.loc[optimal_idx, 'model']
    
    # Find range where model > treat all
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
        
        # Clinical scenarios
        scenarios = {
            'Conservador (25%)': 0.25,
            'Balanceado (15%)': 0.15,
            'Agressivo (8%)': 0.08
        }
        
        print(f"\n📋 ANÁLISE POR CENÁRIO CLÍNICO:")
        print("-" * 80)
        
        for scenario_name, scenario_thresh in scenarios.items():
            if scenario_thresh in dca_df['threshold'].values:
                row = dca_df[dca_df['threshold'] == scenario_thresh].iloc[0]
                
                nb_model = row['model']
                nb_treat_all = row['treat_all']
                advantage = row['advantage_vs_treat_all']
                
                status = '✅ ÚTIL' if advantage > 0 else '❌ INÚTIL'
                
                print(f"\n{scenario_name}:")
                print(f"   Threshold: {scenario_thresh:.3f}")
                print(f"   Net Benefit (Modelo): {nb_model:.4f}")
                print(f"   Net Benefit (Treat All): {nb_treat_all:.4f}")
                print(f"   Vantagem: {advantage:+.4f} {status}")
        
    else:
        print(f"\n❌ MODELO NÃO ADICIONA VALOR CLÍNICO")
        print(f"   Net benefit sempre inferior a 'treat all'")
    
    return dca_df, optimal_threshold

def plot_decision_curve(dca_df, model_name='Model'):
    """
    Plot Decision Curve Analysis
    """
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # === PLOT 1: Net Benefit Curves ===
    ax1 = axes[0]
    
    ax1.plot(dca_df['threshold'], dca_df['model'], 
            linewidth=3, label=f'{model_name}', color='#3498db', marker='o', markersize=4, markevery=5)
    ax1.plot(dca_df['threshold'], dca_df['treat_all'], 
            linewidth=2, label='Treat All', color='#e74c3c', linestyle='--', alpha=0.7)
    ax1.axhline(0, linewidth=2, label='Treat None', color='black', linestyle=':', alpha=0.7)
    
    # Highlight optimal threshold
    optimal_idx = dca_df['model'].idxmax()
    optimal_thresh = dca_df.loc[optimal_idx, 'threshold']
    optimal_nb = dca_df.loc[optimal_idx, 'model']
    
    ax1.scatter(optimal_thresh, optimal_nb, s=200, color='yellow', 
               zorder=5, edgecolor='black', linewidth=2)
    ax1.annotate(f'Optimal\n({optimal_thresh:.2f})', 
                xy=(optimal_thresh, optimal_nb), 
                xytext=(10, 10), textcoords='offset points',
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.7),
                arrowprops=dict(arrowstyle='->', color='black', lw=2))
    
    ax1.set_xlabel('Threshold Probability', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Net Benefit', fontweight='bold', fontsize=12)
    ax1.set_title('Decision Curve Analysis', fontweight='bold', fontsize=14)
    ax1.legend(loc='upper right', fontsize=11)
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 0.5])
    
    # === PLOT 2: Advantage Over Treat All ===
    ax2 = axes[1]
    
    ax2.fill_between(dca_df['threshold'], 0, dca_df['advantage_vs_treat_all'], 
                     where=dca_df['advantage_vs_treat_all'] > 0, 
                     color='green', alpha=0.3, label='Modelo Útil')
    ax2.fill_between(dca_df['threshold'], 0, dca_df['advantage_vs_treat_all'], 
                     where=dca_df['advantage_vs_treat_all'] <= 0, 
                     color='red', alpha=0.3, label='Modelo Inútil')
    
    ax2.plot(dca_df['threshold'], dca_df['advantage_vs_treat_all'], 
            linewidth=2, color='black', marker='s', markersize=4, markevery=5)
    
    ax2.axhline(0, linewidth=2, color='black', linestyle='-', alpha=0.5)
    
    ax2.set_xlabel('Threshold Probability', fontweight='bold', fontsize=12)
    ax2.set_ylabel('Net Benefit Advantage vs. Treat All', fontweight='bold', fontsize=12)
    ax2.set_title('Clinical Utility Range', fontweight='bold', fontsize=14)
    ax2.legend(loc='upper right', fontsize=11)
    ax2.grid(True, alpha=0.3)
    ax2.set_xlim([0, 0.5])
    
    plt.suptitle(f'DECISION CURVE ANALYSIS - {model_name.upper()}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / f'decision_curve_analysis_{model_name}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ DCA salvo: decision_curve_analysis_{model_name}.png")

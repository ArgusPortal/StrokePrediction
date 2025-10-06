"""
Comprehensive visualization module for Stroke Prediction v3
All publication-ready plots and dashboards
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from sklearn.metrics import (
    precision_recall_curve, roc_curve, confusion_matrix,
    average_precision_score, roc_auc_score,
    brier_score_loss
)
from sklearn.calibration import calibration_curve
from .config import RESULTS_PATH, VIZ_CONFIG

# Configurar estilo global
plt.style.use(VIZ_CONFIG['style'])
sns.set_palette(VIZ_CONFIG['color_palette'])

def plot_model_comparison_comprehensive(results, ranking, y_val):
    """Visualização completa de comparação de modelos (como v2)"""
    
    fig = plt.figure(figsize=(24, 16))
    gs = fig.add_gridspec(4, 4, hspace=0.35, wspace=0.3)
    
    colors_models = VIZ_CONFIG['color_palette']
    
    # 1. COMPARAÇÃO DE MÉTRICAS
    ax1 = fig.add_subplot(gs[0, :2])
    
    model_names = []
    roc_aucs, pr_aucs, balanced_accs = [], [], []
    
    for name, result in results.items():
        if 'val_metrics' in result and 'dummy' not in name.lower():
            model_names.append(name.replace('_', ' ').title())
            roc_aucs.append(result['val_metrics']['roc_auc'])
            pr_aucs.append(result['val_metrics']['pr_auc'])
            balanced_accs.append(result['val_metrics'].get('balanced_acc', 0))
    
    x = np.arange(len(model_names))
    width = 0.25
    
    bars1 = ax1.bar(x - width, roc_aucs, width, label='ROC-AUC', 
                    color=colors_models[0], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x, pr_aucs, width, label='PR-AUC', 
                    color=colors_models[1], alpha=0.8, edgecolor='black', linewidth=1.5)
    bars3 = ax1.bar(x + width, balanced_accs, width, label='Balanced Acc', 
                    color=colors_models[2], alpha=0.8, edgecolor='black', linewidth=1.5)
    
    ax1.set_xlabel('Modelos', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title('COMPARACAO DE PERFORMANCE', fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=10)
    ax1.legend(loc='upper left', fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Adicionar valores
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax1.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3), textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # 2. PRECISION-RECALL CURVES
    ax3 = fig.add_subplot(gs[1, :2])
    
    for i, (name, result) in enumerate(list(results.items())[:7]):
        if 'y_proba' in result and 'dummy' not in name.lower():
            precision, recall, _ = precision_recall_curve(y_val, result['y_proba'])
            pr_auc = result['val_metrics']['pr_auc']
            
            ax3.plot(recall, precision, 
                    color=colors_models[i % len(colors_models)], 
                    label=f"{name.replace('_', ' ').title()} (AUC={pr_auc:.3f})", 
                    linewidth=2.5, alpha=0.8)
    
    baseline = (y_val == 1).sum() / len(y_val)
    ax3.axhline(baseline, color='red', linestyle='--', linewidth=2, 
                label=f'Baseline ({baseline:.3f})', alpha=0.7)
    
    ax3.set_xlabel('Recall', fontweight='bold', fontsize=12)
    ax3.set_ylabel('Precision', fontweight='bold', fontsize=12)
    ax3.set_title('CURVAS PRECISION-RECALL', fontweight='bold', fontsize=14)
    ax3.legend(loc='lower left', fontsize=9)
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim(0, 1.0)
    ax3.set_ylim(0, 1.0)
    
    # 3. ROC CURVES
    ax4 = fig.add_subplot(gs[1, 2:])
    
    for i, (name, result) in enumerate(list(results.items())[:7]):
        if 'y_proba' in result and 'dummy' not in name.lower():
            fpr, tpr, _ = roc_curve(y_val, result['y_proba'])
            roc_auc = result['val_metrics']['roc_auc']
            
            ax4.plot(fpr, tpr, 
                    color=colors_models[i % len(colors_models)], 
                    label=f"{name.replace('_', ' ').title()} (AUC={roc_auc:.3f})", 
                    linewidth=2.5, alpha=0.8)
    
    ax4.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label='Random')
    ax4.set_xlabel('False Positive Rate', fontweight='bold', fontsize=12)
    ax4.set_ylabel('True Positive Rate', fontweight='bold', fontsize=12)
    ax4.set_title('CURVAS ROC', fontweight='bold', fontsize=14)
    ax4.legend(loc='lower right', fontsize=9)
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle('ANALISE COMPLETA DE PERFORMANCE', 
                fontsize=18, fontweight='bold', y=0.995)
    
    plt.savefig(RESULTS_PATH / 'comprehensive_model_evaluation.png', 
                dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.show()
    
    # ========== SUMÁRIO TEXTUAL ==========
    print("\n" + "="*80)
    print("📊 SUMÁRIO DA COMPARAÇÃO DE MODELOS")
    print("="*80)
    
    # Tabela de métricas
    summary_data = []
    for name, result in results.items():
        if 'val_metrics' in result and 'dummy' not in name.lower():
            summary_data.append({
                'Modelo': name.replace('_', ' ').title(),
                'ROC-AUC': f"{result['val_metrics']['roc_auc']:.4f}",
                'PR-AUC': f"{result['val_metrics']['pr_auc']:.4f}",
                'Balanced Acc': f"{result['val_metrics'].get('balanced_acc', 0):.4f}"
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\n📋 TABELA DE MÉTRICAS (Validation Set):")
    print(summary_df.to_string(index=False))
    
    # Ranking
    print("\n🏆 RANKING (por PR-AUC):")
    for i, (name, result) in enumerate(ranking[:5], 1):
        pr_auc = result['val_metrics']['pr_auc']
        roc_auc = result['val_metrics']['roc_auc']
        print(f"   {i}. {name.replace('_', ' ').title():25s} → PR-AUC: {pr_auc:.4f} | ROC-AUC: {roc_auc:.4f}")
    
    # Melhor modelo
    if ranking:
        best_name, best_result = ranking[0]
        print(f"\n🥇 MELHOR MODELO: {best_name.upper()}")
        print(f"   PR-AUC:      {best_result['val_metrics']['pr_auc']:.4f}")
        print(f"   ROC-AUC:     {best_result['val_metrics']['roc_auc']:.4f}")
        print(f"   Balanced Acc: {best_result['val_metrics'].get('balanced_acc', 0):.4f}")
    
    return summary_df


def plot_calibration_analysis(results, ranking, y_val):
    """Análise de calibração completa com todos os 6 gráficos"""
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('ANALISE DE CALIBRACAO', fontsize=16, fontweight='bold')
    
    # ========== 1. CURVAS DE CALIBRAÇÃO ==========
    ax1 = axes[0, 0]
    
    for i, (name, result) in enumerate(list(ranking[:5])):
        if 'y_proba' in result:
            y_proba = result['y_proba']
            
            fraction_of_positives, mean_predicted_value = calibration_curve(
                y_val, y_proba, n_bins=10, strategy='uniform'
            )
            
            ax1.plot(mean_predicted_value, fraction_of_positives, 's-',
                    label=name.replace('_', ' ').title(), 
                    linewidth=2, markersize=8, alpha=0.8)
    
    ax1.plot([0, 1], [0, 1], 'k:', label="Perfeitamente Calibrado", linewidth=2)
    ax1.set_xlabel('Probabilidade Predita Media', fontweight='bold')
    ax1.set_ylabel('Fracao de Positivos', fontweight='bold')
    ax1.set_title('Curvas de Calibracao', fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # ========== 2. ERRO DE CALIBRAÇÃO ==========
    ax2 = axes[0, 1]
    
    model_names_cal = []
    calibration_errors = []
    
    for name, result in list(ranking[:6]):
        if 'y_proba' in result:
            y_proba = result['y_proba']
            fraction_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10)
            cal_error = np.mean(np.abs(fraction_pos - mean_pred))
            
            model_names_cal.append(name.replace('_', ' ').title())
            calibration_errors.append(cal_error)
    
    colors_cal = ['green' if e < 0.05 else 'orange' if e < 0.1 else 'red' 
                  for e in calibration_errors]
    
    ax2.barh(model_names_cal, calibration_errors, color=colors_cal, alpha=0.8, edgecolor='black')
    ax2.axvline(0.05, color='green', linestyle='--', linewidth=2, label='Excelente (<0.05)')
    ax2.axvline(0.10, color='orange', linestyle='--', linewidth=2, label='Aceitavel (<0.10)')
    
    ax2.set_xlabel('Erro de Calibracao', fontweight='bold')
    ax2.set_title('Erro de Calibracao por Modelo', fontweight='bold')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ========== 3. BRIER SCORE ==========
    ax3 = axes[0, 2]
    
    model_names_brier = []
    brier_scores = []
    
    for name, result in list(ranking[:6]):
        if 'y_proba' in result:
            brier = brier_score_loss(y_val, result['y_proba'])
            model_names_brier.append(name.replace('_', ' ').title())
            brier_scores.append(brier)
    
    ax3.bar(range(len(model_names_brier)), brier_scores, 
           color=plt.cm.get_cmap('RdYlGn_r')(np.linspace(0.3, 0.7, len(brier_scores))), 
           alpha=0.8, edgecolor='black')
    
    ax3.set_xticks(range(len(model_names_brier)))
    ax3.set_xticklabels(model_names_brier, rotation=45, ha='right', fontsize=9)
    ax3.set_ylabel('Brier Score', fontweight='bold')
    ax3.set_title('Brier Score (Menor = Melhor)', fontweight='bold')
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== 4. RELIABILITY DIAGRAM (MELHOR MODELO) ==========
    ax4 = axes[1, 0]
    
    if ranking:
        best_result = ranking[0][1]
        y_proba_best = best_result['y_proba']
        
        fraction_of_positives, mean_predicted_value = calibration_curve(
            y_val, y_proba_best, n_bins=10
        )
        
        bin_edges = np.linspace(0, 1, 11)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        ax4.bar(bin_centers, fraction_of_positives, width=0.08, 
               alpha=0.7, color='#3498db', edgecolor='black')
        ax4.plot([0, 1], [0, 1], 'k--', label='Ideal', linewidth=2)
        
        ax4.set_xlabel('Probabilidade Predita', fontweight='bold')
        ax4.set_ylabel('Fracao Real de Positivos', fontweight='bold')
        ax4.set_title(f'Reliability Diagram - {ranking[0][0].upper()}', fontweight='bold')
        ax4.legend(fontsize=9)
        ax4.grid(True, alpha=0.3)
    
    # ========== 5. DISTRIBUIÇÃO DE PROBABILIDADES ==========
    ax5 = axes[1, 1]
    
    if ranking:
        best_result = ranking[0][1]
        y_proba_best = best_result['y_proba']
        
        ax5.hist(y_proba_best, bins=50, alpha=0.7, color='#9b59b6', edgecolor='black', density=True)
        
        mean_prob = np.mean(y_proba_best)
        median_prob = np.median(y_proba_best)
        
        ax5.axvline(mean_prob, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_prob:.3f}')
        ax5.axvline(median_prob, color='blue', linestyle='--', linewidth=2, label=f'Mediana: {median_prob:.3f}')
        
        ax5.set_xlabel('Probabilidade Predita', fontweight='bold')
        ax5.set_ylabel('Densidade', fontweight='bold')
        ax5.set_title('Distribuicao de Probabilidades', fontweight='bold')
        ax5.legend(fontsize=9)
        ax5.grid(True, alpha=0.3, axis='y')
    
    # ========== 6. CALIBRATION BINS ANALYSIS ==========
    ax6 = axes[1, 2]
    
    if ranking:
        best_result = ranking[0][1]
        y_proba_best = best_result['y_proba']
        
        n_bins = 10
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_proba_best, bins) - 1
        
        bin_counts = []
        bin_accuracies = []
        
        for i in range(n_bins):
            mask = bin_indices == i
            if mask.sum() > 0:
                bin_counts.append(mask.sum())
                bin_accuracies.append(y_val[mask].mean())
            else:
                bin_counts.append(0)
                bin_accuracies.append(0)
        
        x_bins = range(n_bins)
        
        ax6_twin = ax6.twinx()
        
        ax6.bar(x_bins, bin_counts, alpha=0.6, color='#3498db', edgecolor='black')
        ax6_twin.plot(x_bins, bin_accuracies, 'ro-', linewidth=2.5, markersize=8)
        
        ax6.set_xlabel('Bin de Probabilidade', fontweight='bold')
        ax6.set_ylabel('Contagem', color='#3498db', fontweight='bold')
        ax6_twin.set_ylabel('Taxa Real de Stroke', color='red', fontweight='bold')
        ax6.set_title('Analise por Bins de Probabilidade', fontweight='bold')
        ax6.tick_params(axis='y', labelcolor='#3498db')
        ax6_twin.tick_params(axis='y', labelcolor='red')
        ax6.grid(True, alpha=0.3)
        
        bin_labels = [f'{bins[i]:.1f}-{bins[i+1]:.1f}' for i in range(n_bins)]
        ax6.set_xticks(x_bins)
        ax6.set_xticklabels(bin_labels, rotation=45, ha='right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'calibration_analysis.png', 
                dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.show()
    
    # ========== SUMÁRIO TEXTUAL ==========
    print("\n" + "="*80)
    print("📊 SUMÁRIO DA ANÁLISE DE CALIBRAÇÃO")
    print("="*80)
    
    # Tabela de calibração
    calibration_summary = []
    
    for name, result in list(ranking[:6]):
        if 'y_proba' in result:
            y_proba = result['y_proba']
            
            # Calcular métricas de calibração
            fraction_pos, mean_pred = calibration_curve(y_val, y_proba, n_bins=10)
            cal_error = np.mean(np.abs(fraction_pos - mean_pred))
            brier = brier_score_loss(y_val, y_proba)
            
            # Brier Skill Score (vs baseline)
            baseline_brier = brier_score_loss(y_val, np.full_like(y_proba, y_val.mean()))
            brier_skill = 1 - (brier / baseline_brier)
            
            # Status
            if cal_error < 0.05:
                status = '✅ Excelente'
            elif cal_error < 0.10:
                status = '⚠️ Aceitável'
            else:
                status = '❌ Ruim'
            
            calibration_summary.append({
                'Modelo': name.replace('_', ' ').title(),
                'Cal. Error': f"{cal_error:.4f}",
                'Brier Score': f"{brier:.4f}",
                'Brier Skill': f"{brier_skill:.4f}",
                'Status': status
            })
    
    cal_df = pd.DataFrame(calibration_summary)
    print("\n📋 MÉTRICAS DE CALIBRAÇÃO:")
    print(cal_df.to_string(index=False))
    
    # Interpretação
    print("\n📖 INTERPRETAÇÃO:")
    print("   Cal. Error < 0.05:  Excelente calibração")
    print("   Cal. Error < 0.10:  Calibração aceitável")
    print("   Brier Score:        Menor = melhor (erro quadrático)")
    print("   Brier Skill > 0:    Melhor que baseline")
    
    # Melhor calibrado
    best_cal_idx = cal_df['Cal. Error'].str.replace('[^0-9.]', '', regex=True).astype(float).idxmin()
    best_cal_model = cal_df.loc[best_cal_idx, 'Modelo']
    print(f"\n🏆 MELHOR CALIBRAÇÃO: {best_cal_model}")
    
    return cal_df


def plot_confusion_matrices(results, ranking, y_val, threshold=0.15):
    """Confusion matrices para top modelos"""
    
    n_models = min(4, len(ranking))
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    axes = axes.ravel()
    
    fig.suptitle('CONFUSION MATRICES - TOP 4 MODELOS', 
                fontsize=16, fontweight='bold')
    
    for i, (name, result) in enumerate(ranking[:n_models]):
        ax = axes[i]
        
        y_proba = result['y_proba']
        y_pred = (y_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_val, y_pred)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=['No Stroke', 'Stroke'],
                   yticklabels=['No Stroke', 'Stroke'],
                   linewidths=2, linecolor='black')
        
        tn, fp, fn, tp = cm.ravel()
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        
        title = f"{name.replace('_', ' ').title()}\n"
        title += f"Recall: {recall:.3f} | Precision: {precision:.3f}"
        
        ax.set_title(title, fontweight='bold', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(RESULTS_PATH / 'confusion_matrices.png', 
                dpi=VIZ_CONFIG['figure_dpi'], bbox_inches='tight')
    plt.show()
    
    # ========== SUMÁRIO TEXTUAL ==========
    print("\n" + "="*80)
    print("📊 SUMÁRIO DAS CONFUSION MATRICES")
    print("="*80)
    
    n_models = min(4, len(ranking))
    
    cm_summary = []
    
    for i, (name, result) in enumerate(ranking[:n_models]):
        y_proba = result['y_proba']
        y_pred = (y_proba >= threshold).astype(int)
        
        cm = confusion_matrix(y_val, y_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Calcular métricas
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        cm_summary.append({
            'Modelo': name.replace('_', ' ').title(),
            'TP': int(tp),
            'FP': int(fp),
            'FN': int(fn),
            'TN': int(tn),
            'Recall': f"{recall:.3f}",
            'Precision': f"{precision:.3f}",
            'F1-Score': f"{f1:.3f}",
            'Specificity': f"{specificity:.3f}"
        })
    
    cm_df = pd.DataFrame(cm_summary)
    print(f"\n📋 MÉTRICAS (Threshold = {threshold}):")
    print(cm_df.to_string(index=False))
    
    # Análise de tradeoff
    print("\n⚖️ TRADEOFF RECALL vs PRECISION:")
    for _, row in cm_df.iterrows():
        recall = float(row['Recall'])
        precision = float(row['Precision'])
        
        if recall >= 0.70 and precision >= 0.15:
            verdict = "✅ Atende requisitos"
        elif recall >= 0.70:
            verdict = "⚠️ Baixa precisão"
        else:
            verdict = "❌ Baixo recall"
        
        print(f"   {row['Modelo']:25s}: Recall={recall:.3f}, Precision={precision:.3f} → {verdict}")
    
    return cm_df


def plot_training_summary(results, summary_df):
    """Visualização completa dos resultados de treinamento"""
    
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    # ========== 1. CV SCORES COMPARISON ==========
    ax1 = fig.add_subplot(gs[0, :])
    
    models = summary_df['Model'].values
    x = np.arange(len(models))
    width = 0.2
    
    bars1 = ax1.bar(x - width, summary_df['CV_PR-AUC_Mean'], width, 
                   yerr=summary_df['CV_PR-AUC_Std'], label='CV PR-AUC',
                   color='#3498db', alpha=0.8, capsize=5)
    bars2 = ax1.bar(x, summary_df['Val_PR-AUC'], width, label='Val PR-AUC',
                   color='#e74c3c', alpha=0.8)
    bars3 = ax1.bar(x + width, summary_df['Val_ROC-AUC'], width, label='Val ROC-AUC',
                   color='#2ecc71', alpha=0.8)
    
    ax1.set_xlabel('Modelos', fontweight='bold', fontsize=12)
    ax1.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax1.set_title('COMPARAÇÃO DE MÉTRICAS - Cross-Validation vs Validation', 
                 fontweight='bold', fontsize=14)
    ax1.set_xticks(x)
    ax1.set_xticklabels(models, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.set_ylim(0, 1.0)
    
    # Adicionar valores
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # ========== 2. OVERFITTING ANALYSIS ==========
    ax2 = fig.add_subplot(gs[1, 0])
    
    colors_overfit = ['green' if g < 0.05 else 'orange' if g < 0.10 else 'red'
                      for g in summary_df['PR-AUC_Overfit_Gap']]
    
    bars = ax2.barh(models, summary_df['PR-AUC_Overfit_Gap'], 
                    color=colors_overfit, alpha=0.8, edgecolor='black')
    
    ax2.axvline(0.05, color='green', linestyle='--', linewidth=2, label='OK (<0.05)')
    ax2.axvline(0.10, color='orange', linestyle='--', linewidth=2, label='Atenção (<0.10)')
    
    ax2.set_xlabel('Overfitting Gap', fontweight='bold')
    ax2.set_title('ANÁLISE DE OVERFITTING', fontweight='bold', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='x')
    
    # ========== 3. TRAINING TIME ==========
    ax3 = fig.add_subplot(gs[1, 1])
    
    ax3.bar(range(len(models)), summary_df['Training_Time_s'],
           color='#9b59b6', alpha=0.8, edgecolor='black')
    
    ax3.set_xticks(range(len(models)))
    ax3.set_xticklabels(models, rotation=45, ha='right')
    ax3.set_ylabel('Tempo (segundos)', fontweight='bold')
    ax3.set_title('TEMPO DE TREINAMENTO', fontweight='bold', fontsize=12)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # ========== 4. CV STABILITY (boxplot) ==========
    ax4 = fig.add_subplot(gs[1, 2])
    
    cv_pr_aucs = [results[model]['cv_stats']['pr_auc']['test_scores'] 
                  for model in models]
    
    bp = ax4.boxplot(cv_pr_aucs, patch_artist=True)
    
    for patch in bp['boxes']:
        patch.set_facecolor('#3498db')
        patch.set_alpha(0.7)
    
    ax4.set_xticklabels(models, rotation=45, ha='right')
    ax4.set_ylabel('PR-AUC', fontweight='bold')
    ax4.set_title('ESTABILIDADE DO CV (PR-AUC)', fontweight='bold', fontsize=12)
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle('RESULTADOS COMPLETOS DO TREINAMENTO', fontsize=16, fontweight='bold')
    
    plt.savefig(RESULTS_PATH / 'training_summary_complete.png', dpi=300, bbox_inches='tight')
    plt.show()

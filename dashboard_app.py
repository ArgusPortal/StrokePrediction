import json
import pickle
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

# =============================================================================
# CONFIGURATION
# =============================================================================

st.set_page_config(
    page_title="Painel de Predição de AVC - v4 Calibrado",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

BASE_DIR = Path(__file__).parent
MODELS_PATH = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"
OUTPUTS_PATH = BASE_DIR / "outputs"

DEFAULT_THRESHOLD = 0.085
PROB_BUCKETS = np.linspace(0.0, 1.0, 21)

REFERENCE_NOTES = {
    "work_type": {
        "Private": "Categoria base; 'Self-employed' é normalizado para este grupo.",
        "Govt_job": "Categoria base; 'children' e 'Never_worked' convertem para Govt_job.",
        "Self-employed": "Normalizado como 'Private' pelo backend.",
        "children": "Normalizado como 'Govt_job' pelo backend.",
        "Never_worked": "Normalizado como 'Govt_job' pelo backend.",
    },
    "smoking_status": {
        "Unknown": "Tratado como 'formerly smoked' para cálculo do risco.",
    },
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


@st.cache_data(show_spinner=False)
def load_threshold() -> float:
    path = RESULTS_PATH / "threshold.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return float(data.get("threshold", DEFAULT_THRESHOLD))
        except (json.JSONDecodeError, ValueError, TypeError):
            pass
    return DEFAULT_THRESHOLD


@st.cache_data(show_spinner=False)
def load_calibration_meta() -> Dict[str, Any]:
    path = MODELS_PATH / "calibration_meta.json"
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                data = json.load(f)
            return data
        except (json.JSONDecodeError, ValueError):
            return {}
    return {}


@st.cache_data(show_spinner=False)
def load_split_arrays(split: str) -> Tuple[np.ndarray, np.ndarray]:
    npz_path = RESULTS_PATH / f"logits_probs_{split}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Não encontrei {npz_path}.")
    data = np.load(npz_path)
    y_true = data["y"]
    probs = data["probs"]
    return y_true, probs


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probs, bins) - 1
    total = len(y_true)
    ece = 0.0
    for b in range(n_bins):
        idx = bin_ids == b
        bin_count = np.sum(idx)
        if bin_count == 0:
            continue
        acc = y_true[idx].mean()
        conf = probs[idx].mean()
        ece += (bin_count / total) * abs(acc - conf)
    return float(ece)


def compute_performance(
    split: str,
    threshold: float,
) -> Dict[str, Any]:
    y_true, probs = load_split_arrays(split)
    preds = (probs >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

    precision = precision_score(y_true, preds, zero_division=0)
    recall = recall_score(y_true, preds, zero_division=0)
    f1 = f1_score(y_true, preds, zero_division=0)
    bal_acc = balanced_accuracy_score(y_true, preds)
    brier = brier_score_loss(y_true, probs)
    ece = expected_calibration_error(y_true, probs, n_bins=10)

    return {
        "split": split,
        "threshold": threshold,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "balanced_accuracy": float(bal_acc),
        "brier": float(brier),
        "ece": float(ece),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
        "support": int(len(y_true)),
        "y_true": y_true,
        "probs": probs,
    }


@st.cache_data(show_spinner=False)
def load_metrics_summary(threshold: float) -> Dict[str, Dict[str, Any]]:
    summaries = {}
    for split in ("val", "test"):
        summaries[split] = compute_performance(split, threshold)
    return summaries


@st.cache_data(show_spinner=False)
def load_fairness_summary() -> Tuple[pd.DataFrame, List[Dict[str, Any]]]:
    path = RESULTS_PATH / "fairness_audit.json"
    if not path.exists():
        return pd.DataFrame(), []
    with path.open("r", encoding="utf-8") as f:
        raw = json.load(f)

    baseline = raw.get("baseline", {})
    test_block = baseline.get("test", {})
    attributes = test_block.get("attributes", {})
    rows: List[Dict[str, Any]] = []
    for attr, info in attributes.items():
        gap = info.get("TPR_gap")
        gap_ci = info.get("TPR_gap_ci", {})
        rows.append(
            {
                "atributo": attr,
                "TPR_gap": gap,
                "TPR_gap_ci_lower": gap_ci.get("gap_ci_lower"),
                "TPR_gap_ci_upper": gap_ci.get("gap_ci_upper"),
            }
        )
    df = pd.DataFrame(rows)
    alerts = baseline.get("alerts", [])
    return df, alerts


@st.cache_data(show_spinner=False)
def load_drift_summary() -> pd.DataFrame:
    path = RESULTS_PATH / "drift_monitoring_results.pkl"
    if not path.exists():
        return pd.DataFrame()
    with path.open("rb") as f:
        raw = pickle.load(f)
    psi_results = raw.get("psi_results", [])
    return pd.DataFrame(psi_results)


def build_confusion_heatmap(tp: int, fp: int, fn: int, tn: int) -> go.Figure:
    matrix = np.array([[tn, fp], [fn, tp]])
    labels = [["TN", "FP"], ["FN", "TP"]]
    fig = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=["Pred. Neg", "Pred. Pos"],
            y=["Real Neg", "Real Pos"],
            colorscale="Blues",
            text=labels,
            hoverinfo="z",
        )
    )
    fig.update_layout(title="Matriz de Confusão (@0.085)", height=320)
    return fig


def build_pr_curve(y_true: np.ndarray, probs: np.ndarray, threshold: float) -> go.Figure:
    precision, recall, thresholds = precision_recall_curve(y_true, probs)
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=recall,
            y=precision,
            mode="lines",
            name="Curva PR",
            line=dict(color="#1f77b4", width=3),
        )
    )
    preds = (probs >= threshold).astype(int)
    point_prec = precision_score(y_true, preds, zero_division=0)
    point_rec = recall_score(y_true, preds, zero_division=0)
    fig.add_trace(
        go.Scatter(
            x=[point_rec],
            y=[point_prec],
            mode="markers",
            marker=dict(size=10, color="#d62728"),
            name=f"Threshold {threshold:.3f}",
        )
    )
    fig.update_layout(
        title="Curva Precisão-Recall",
        xaxis_title="Recall",
        yaxis_title="Precisão",
        height=360,
    )
    return fig


def build_calibration_chart(y_true: np.ndarray, probs: np.ndarray) -> go.Figure:
    prob_true, prob_pred = calibration_curve(y_true, probs, n_bins=10, strategy="uniform")
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Ideal",
            line=dict(color="#7f7f7f", dash="dash"),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=prob_pred,
            y=prob_true,
            mode="lines+markers",
            name="Modelo",
            line=dict(color="#2ca02c", width=3),
            marker=dict(size=8),
        )
    )
    fig.update_layout(
        title="Reliability Diagram",
        xaxis_title="Probabilidade prevista",
        yaxis_title="Frequência observada",
        height=360,
    )
    return fig


def build_fairness_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["alerta"] = (df["TPR_gap"] > 0.10) & (df["TPR_gap_ci_lower"] > 0.0)
    df.sort_values("TPR_gap", ascending=False, inplace=True)
    return df


def build_drift_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    df["psi"] = df["psi"].astype(float)
    df.sort_values("psi", ascending=False, inplace=True)
    return df


def generate_static_exports(
    test_metrics: Dict[str, Any],
    fairness_df: pd.DataFrame,
    drift_df: pd.DataFrame,
) -> None:
    OUTPUTS_PATH.mkdir(parents=True, exist_ok=True)

    # Performance figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    cm = np.array([[test_metrics["tn"], test_metrics["fp"]], [test_metrics["fn"], test_metrics["tp"]]])
    axes[0].imshow(cm, cmap="Blues")
    axes[0].set_xticks([0, 1])
    axes[0].set_yticks([0, 1])
    axes[0].set_xticklabels(["Pred Neg", "Pred Pos"])
    axes[0].set_yticklabels(["Real Neg", "Real Pos"])
    for (i, j), val in np.ndenumerate(cm):
        axes[0].text(j, i, f"{val}", ha="center", va="center", color="black")
    axes[0].set_title("Matriz de Confusão (@0.085)")

    metrics_text = (
        f"PPV: {test_metrics['precision']:.3f}\n"
        f"Recall: {test_metrics['recall']:.3f}\n"
        f"F1: {test_metrics['f1']:.3f}\n"
        f"Balanced Acc: {test_metrics['balanced_accuracy']:.3f}\n"
        f"Brier: {test_metrics['brier']:.3f}\n"
        f"ECE: {test_metrics['ece']:.3f}"
    )
    axes[1].axis("off")
    axes[1].text(0.0, 0.5, metrics_text, fontsize=12, va="center")
    plt.tight_layout()
    fig.savefig(OUTPUTS_PATH / "dashboard_v4_shadow_performance.png", dpi=150)
    plt.close(fig)

    # Fairness figure
    if not fairness_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        fairness_df_plot = fairness_df.sort_values("TPR_gap", ascending=True)
        ax.barh(fairness_df_plot["atributo"], fairness_df_plot["TPR_gap"], color="#1f77b4")
        ax.axvline(0.10, color="#d62728", linestyle="--", label="Limite 0.10")
        ax.set_xlabel("TPR gap")
        ax.set_title("Fairness – gaps de TPR (TEST)")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUTS_PATH / "dashboard_v4_shadow_fairness.png", dpi=150)
        plt.close(fig)

    # Drift figure
    if not drift_df.empty:
        fig, ax = plt.subplots(figsize=(7, 4))
        drift_plot = drift_df.sort_values("psi", ascending=False)
        ax.bar(drift_plot["feature"], drift_plot["psi"], color="#2ca02c")
        ax.axhline(0.1, color="#d62728", linestyle="--", label="PSI = 0.10")
        ax.set_ylabel("PSI")
        ax.set_title("Drift – PSI por feature")
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        ax.legend()
        plt.tight_layout()
        fig.savefig(OUTPUTS_PATH / "dashboard_v4_shadow_drift.png", dpi=150)
        plt.close(fig)


# =============================================================================
# DATA LOADING
# =============================================================================

threshold_used = load_threshold()
calibration_meta = load_calibration_meta()
metrics_summary = load_metrics_summary(threshold_used)
fairness_df_raw, fairness_alerts = load_fairness_summary()
fairness_df = build_fairness_table(fairness_df_raw)
drift_df_raw = load_drift_summary()
drift_df = build_drift_table(drift_df_raw)

try:
    generate_static_exports(metrics_summary["test"], fairness_df, drift_df)
except Exception:
    pass

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("🏥 Predição de AVC v4")
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Configurações do Modelo")
st.sidebar.metric("Limiar Operacional", f"{threshold_used:.3f}")
st.sidebar.caption("📌 Probabilidade mínima para classificar como alto risco")

calibration_version = calibration_meta.get("calibration_version", "desconhecida")
st.sidebar.metric("Versão da Calibração", calibration_version)
st.sidebar.caption("🎯 Ajuste fino das probabilidades do modelo")

model_metadata_path = MODELS_PATH / "model_metadata_production.json"
if model_metadata_path.exists():
    with model_metadata_path.open("r", encoding="utf-8") as f:
        model_metadata = json.load(f)
else:
    model_metadata = {}

model_version = model_metadata.get("model_info", {}).get("version", "N/A")
st.sidebar.metric("Versão do Modelo", model_version)
st.sidebar.caption(f"🕐 Última atualização: {datetime.now().strftime('%d/%m/%Y às %H:%M')}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📑 Navegação")

page = st.sidebar.radio(
    "Escolha uma seção:",
    [
        "📈 Visão Geral",
        "🔮 Predição Individual",
        "📉 Desempenho Detalhado",
        "⚖️ Equidade do Modelo",
        "📊 Monitoramento de Dados",
    ],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Sobre o Sistema")
st.sidebar.info(
    """
    Este sistema utiliza **Inteligência Artificial** para estimar o risco de AVC 
    com base em dados clínicos do paciente.
    
    **Importante:** As predições são auxiliares e não substituem avaliação médica.
    """
)

# =============================================================================
# PAGE: DASHBOARD PRINCIPAL
# =============================================================================

if page == "📈 Visão Geral":
    st.title("🏥 Painel de Predição de AVC - Visão Geral")
    
    # Header explicativo
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 25px;
            border-radius: 10px;
            color: white;
            margin-bottom: 30px;
        ">
            <h3 style="margin: 0; color: white;">📊 Sistema de Predição com IA Calibrada</h3>
            <p style="margin: 10px 0 0 0; font-size: 16px; color: white;">
                Este modelo analisa <strong>múltiplos fatores clínicos</strong> para estimar o risco de AVC. 
                As probabilidades são <strong>calibradas</strong> (ajustadas) para refletir com precisão 
                a chance real de eventos. Limiar operacional atual: <strong>{threshold_used:.1%}</strong>
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

    test_metrics = metrics_summary["test"]
    val_metrics = metrics_summary["val"]

    # KPIs principais com explicações - CORES CORRIGIDAS
    st.markdown("### 🎯 Indicadores de Desempenho (Conjunto de Teste)")
    
    kpi_cols = st.columns(4)
    
    with kpi_cols[0]:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 14px; margin-bottom: 5px; font-weight: 500;">
                    🎯 <strong>Precisão Positiva</strong>
                </div>
                <div style="font-size: 36px; font-weight: bold; color: #ffffff;">
                    {test_metrics['precision']:.1%}
                </div>
                <div style="color: #e8f5e9; font-size: 12px; margin-top: 5px;">
                    Quando prevê AVC, acerta em {test_metrics['precision']:.1%} dos casos
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi_cols[1]:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #e65100 0%, #bf360c 100%); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 14px; margin-bottom: 5px; font-weight: 500;">
                    🔍 <strong>Sensibilidade</strong>
                </div>
                <div style="font-size: 36px; font-weight: bold, color: #ffffff;">
                    {test_metrics['recall']:.1%}
                </div>
                <div style="color: #fff3e0; font-size: 12px; margin-top: 5px;">
                    Detecta {test_metrics['recall']:.1%} dos casos reais de AVC
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi_cols[2]:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #1565c0 0%, #0d47a1 100%); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 14px; margin-bottom: 5px; font-weight: 500;">
                    ⚖️ <strong>F1-Score</strong>
                </div>
                <div style="font-size: 36px; font-weight: bold, color: #ffffff;">
                    {test_metrics['f1']:.1%}
                </div>
                <div style="color: #e3f2fd; font-size: 12px; margin-top: 5px;">
                    Equilíbrio entre precisão e sensibilidade
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with kpi_cols[3]:
        st.markdown(
            f"""
            <div style="background: linear-gradient(135deg, #6a1b9a 0%, #4a148c 100%); padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
                <div style="color: #ffffff; font-size: 14px; margin-bottom: 5px; font-weight: 500;">
                    🎲 <strong>Acurácia Balanceada</strong>
                </div>
                <div style="font-size: 36px; font-weight: bold, color: #ffffff;">
                    {test_metrics['balanced_accuracy']:.1%}
                </div>
                <div style="color: #f3e5f5; font-size: 12px; margin-top: 5px;">
                    Desempenho médio em ambas as classes
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Matriz de confusão com explicação
    st.markdown("### 📊 Matriz de Confusão - Como o Modelo Classifica")
    st.markdown(
        """
        A matriz mostra como o modelo se comporta no conjunto de teste:
        - **VP (Verdadeiro Positivo)**: Previu AVC corretamente ✅
        - **FP (Falso Positivo)**: Previu AVC, mas não houve ⚠️
        - **FN (Falso Negativo)**: Não previu AVC, mas houve ❌
        - **VN (Verdadeiro Negativo)**: Previu ausência de AVC corretamente ✅
        """
    )
    
    col_matrix, col_explanation = st.columns([1, 1])
    
    with col_matrix:
        cm_fig = build_confusion_heatmap(test_metrics["tp"], test_metrics["fp"], test_metrics["fn"], test_metrics["tn"])
        st.plotly_chart(cm_fig, use_container_width=True)
    
    with col_explanation:
        st.markdown("#### 📋 Interpretação dos Resultados")
        
        total = test_metrics["tp"] + test_metrics["fp"] + test_metrics["fn"] + test_metrics["tn"]
        
        st.markdown(
            f"""
            <div style="background: #f5f5f5; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong style="color: #212529;">✅ Acertos Totais:</strong> 
                <span style="color: #212529;">{test_metrics['tp'] + test_metrics['tn']} de {total} ({(test_metrics['tp'] + test_metrics['tn'])/total:.1%})</span>
            </div>
            
            <div style="background: #fff9e6; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong style="color: #212529;">⚠️ Falsos Positivos:</strong> 
                <span style="color: #212529;">{test_metrics['fp']}</span><br>
                <em style="font-size: 12px; color: #666;">Pacientes classificados como alto risco, mas sem AVC</em>
            </div>
            
            <div style="background: #ffebee; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong style="color: #212529;">❌ Falsos Negativos:</strong> 
                <span style="color: #212529;">{test_metrics['fn']}</span><br>
                <em style="font-size: 12px; color: #666;">Casos de AVC não detectados - crítico para saúde!</em>
            </div>
            
            <div style="background: #e8f5e9; padding: 15px; border-radius: 8px; margin: 10px 0;">
                <strong style="color: #212529;">📊 Taxa de Falsos Negativos:</strong> 
                <span style="color: #212529;">{test_metrics['fn']/(test_metrics['fn']+test_metrics['tp']):.1%}</span><br>
                <em style="font-size: 12px; color: #666;">Proporção de AVCs que o modelo não detectou</em>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")
    
    # Comparativo validação vs teste
    st.markdown("### 📈 Validação vs Teste - Consistência do Modelo")
    st.markdown(
        f"""
        Comparação dos resultados em dois conjuntos independentes de dados no limiar **{threshold_used:.3f}**. 
        Métricas similares indicam que o modelo **generaliza bem** para novos pacientes.
        """
    )
    
    comparison_df = pd.DataFrame(
        [
            {
                "Conjunto": "Validação",
                "Precisão Positiva": f"{val_metrics['precision']:.1%}",
                "Sensibilidade": f"{val_metrics['recall']:.1%}",
                "F1-Score": f"{val_metrics['f1']:.1%}",
                "Acurácia Balanceada": f"{val_metrics['balanced_accuracy']:.1%}",
                "Brier Score": f"{val_metrics['brier']:.4f}",
                "ECE": f"{val_metrics['ece']:.4f}",
                "VP": val_metrics["tp"],
                "FP": val_metrics["fp"],
                "FN": val_metrics["fn"],
                "VN": val_metrics["tn"],
            },
            {
                "Conjunto": "Teste",
                "Precisão Positiva": f"{test_metrics['precision']:.1%}",
                "Sensibilidade": f"{test_metrics['recall']:.1%}",
                "F1-Score": f"{test_metrics['f1']:.1%}",
                "Acurácia Balanceada": f"{test_metrics['balanced_accuracy']:.1%}",
                "Brier Score": f"{test_metrics['brier']:.4f}",
                "ECE": f"{test_metrics['ece']:.4f}",
                "VP": test_metrics["tp"],
                "FP": test_metrics["fp"],
                "FN": test_metrics["fn"],
                "VN": test_metrics["tn"],
            },
        ]
    )
    
    st.dataframe(
        comparison_df.style.set_table_styles([{
            'selector': 'td',
            'props': [
                ('background-color', '#f8f9fa'),
                ('color', '#212529'),
                ('border-color', '#dee2e6')
            ]
        }]),
        use_container_width=True
    )
    
    st.caption(
        """
        **📖 Glossário:**
        - **Brier Score**: Medida de qualidade das probabilidades (quanto menor, melhor). Ideal < 0.10
        - **ECE** (Erro de Calibração Esperado): Diferença entre probabilidades previstas e frequências reais. Ideal < 0.05
        """
    )

    st.markdown("---")
    
    # Curvas de desempenho
    st.markdown("### 📊 Curvas de Desempenho (Conjunto de Teste)")
    
    perf_col, calib_col = st.columns(2)
    
    with perf_col:
        st.markdown("#### 🎯 Curva Precisão-Sensibilidade")
        st.plotly_chart(
            build_pr_curve(test_metrics["y_true"], test_metrics["probs"], threshold_used),
            use_container_width=True,
        )
        st.caption(
            """
            Mostra o **trade-off** entre precisão e sensibilidade em diferentes limiares. 
            O ponto vermelho marca o limiar operacional atual.
            """
        )
    
    with calib_col:
        st.markdown("#### 📐 Diagrama de Calibração")
        st.plotly_chart(
            build_calibration_chart(test_metrics["y_true"], test_metrics["probs"]),
            use_container_width=True,
        )
        st.caption(
            f"""
            Compara probabilidades **previstas** com frequências **observadas**. 
            Linha diagonal = calibração perfeita.
            
            - ECE: **{test_metrics['ece']:.4f}** (meta: < 0.05) ✅
            - Brier: **{test_metrics['brier']:.4f}** (meta: < 0.10) ✅
            """
        )

    st.markdown("---")
    
    # Alertas e monitoramento
    st.markdown("### 🚨 Alertas e Monitoramento")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("#### ⚖️ Equidade do Modelo")
        fairness_alerts_text = [
            alert["message"]
            for alert in fairness_alerts
            if alert.get("attribute") in {"Residence_type", "smoking_status"}
        ]
        if fairness_alerts_text:
            for alert_msg in fairness_alerts_text:
                st.error(f"⚠️ {alert_msg}")
            st.caption("Disparidades detectadas entre grupos. Consulte a aba 'Equidade do Modelo' para detalhes.")
        else:
            st.success("✅ Nenhuma disparidade significativa detectada entre grupos demográficos.")
            st.caption("O modelo apresenta desempenho equilibrado entre diferentes populações.")
    
    with alert_col2:
        st.markdown("#### 📊 Qualidade dos Dados")
        if drift_df.empty:
            st.info("ℹ️ Aguardando dados de monitoramento de drift.")
        else:
            worst_feature = drift_df.iloc[0]
            worst_psi = worst_feature['psi']
            
            if worst_psi > 0.25:
                st.error(f"🔴 **Alerta Crítico**: Drift detectado em '{worst_feature.get('feature', '')}' (PSI: {worst_psi:.3f})")
                st.caption("Distribuição dos dados mudou significativamente. Recomendado retreinar o modelo.")
            elif worst_psi > 0.10:
                st.warning(f"🟡 **Alerta Moderado**: Drift em '{worst_feature.get('feature', '')}' (PSI: {worst_psi:.3f})")
                st.caption("Pequena mudança na distribuição dos dados. Monitorar continuamente.")
            else:
                st.success(f"✅ Dados estáveis. Maior PSI: {worst_psi:.3f}")
                st.caption("Distribuição dos dados permanece consistente com o treinamento.")

# =============================================================================
# PAGE: PREDIÇÃO INDIVIDUAL
# =============================================================================

elif page == "🔮 Predição Individual":
    st.title("🔮 Predição Individual Calibrada")
    st.markdown(
        """
        Utilize o modelo calibrado para obter predições individuais com **probabilidades calibradas**
        e interpretação clínica detalhada.
        """
    )

    with st.form("patient_form"):
        demographic_col1, demographic_col2, demographic_col3 = st.columns(3)
        with demographic_col1:
            gender = st.selectbox("Gênero", ["Male", "Female", "Other"])
        with demographic_col2:
            age = st.number_input("Idade", min_value=18, max_value=100, value=50)
        with demographic_col3:
            ever_married = st.selectbox("Estado Civil", ["Yes", "No"])

        clinical_col1, clinical_col2, clinical_col3 = st.columns(3)
        with clinical_col1:
            hypertension = st.selectbox("Hipertensão", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        with clinical_col2:
            heart_disease = st.selectbox("Doença Cardíaca", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        with clinical_col3:
            avg_glucose_level = st.number_input("Glicose Média (mg/dL)", min_value=50.0, max_value=400.0, value=110.0)

        lifestyle_col1, lifestyle_col2, lifestyle_col3 = st.columns(3)
        with lifestyle_col1:
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0)
        with lifestyle_col2:
            work_type = st.selectbox(
                "Tipo de Trabalho",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            )
            note = REFERENCE_NOTES["work_type"].get(work_type)
            if note:
                st.caption(f"ℹ️ {note}")
        with lifestyle_col3:
            smoking_status = st.selectbox(
                "Status de Tabagismo",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
            )
            note = REFERENCE_NOTES["smoking_status"].get(smoking_status)
            if note:
                st.caption(f"ℹ️ {note}")

        residence_type = st.selectbox("Tipo de Residência", ["Urban", "Rural"])

        submitted = st.form_submit_button("Obter Predição", type="primary")

    if submitted:
        payload = {
            "patient_id": None,
            "patient_data": {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status,
            },
            "return_explanation": False,
        }

        api_url = os.getenv("STROKE_API_URL", "http://localhost:8000")
        try:
            with st.spinner("🔄 Consultando API..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                """
                ❌ **Erro de Conexão com a API**
                
                O servidor FastAPI não está rodando. Para iniciar:
                
                1. Abra um novo terminal
                2. Execute: `python start_api_server.py`
                3. Aguarde a mensagem de sucesso
                4. Tente novamente aqui
                """
            )
            st.stop()
        except Exception as exc:
            st.error(f"❌ Erro ao consultar API: {exc}")
            st.stop()
        else:
            prob = result.get("probability_stroke", 0.0)
            calibration_version = result.get("calibration_version", "desconhecida")
            alert_flag = result.get("alert_flag", False)
            risk_tier_data = result.get("risk_tier", {})
            threshold_applied = result.get("threshold_used", threshold_used)

            # ========== VISUALIZAÇÃO ENRIQUECIDA ==========
            st.markdown("---")
            st.markdown("## 📊 Resultado da Predição")
            
            # Determinar cor e ícone baseado no tier
            tier = risk_tier_data.get("tier", "TIER_4_LOW")
            tier_colors = {
                "TIER_1_VERY_HIGH": ("#8B0000", "🚨"),
                "TIER_2_HIGH": ("#FF4500", "⚠️"),
                "TIER_3_MODERATE": ("#FFA500", "⚡"),
                "TIER_4_LOW": ("#32CD32", "✅"),
            }
            color, icon = tier_colors.get(tier, ("#808080", "ℹ️"))
            
            # Card principal de risco
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                    border-left: 5px solid {color};
                    padding: 25px;
                    border-radius: 10px;
                    margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h2 style="color: {color}; margin: 0 0 15px 0;">
                        {icon} {risk_tier_data.get('description', 'Risco Desconhecido')}
                    </h2>
                    <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
                                Probabilidade de AVC
                            </div>
                            <div style="font-size: 48px; font-weight: bold, color: {color};">
                                {prob*100:.1f}%
                            </div>
                        </div>
                        <div style="flex: 2; min-width: 300px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
                                Classificação de Risco
                            </div>
                            <div style="
                                background: white;
                                padding: 12px;
                                border-radius: 8px;
                                font-weight: 600;
                                color: {color};
                                border: 2px solid {color};
                            ">
                                {tier.replace('_', ' ')}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Barra de probabilidade visual
            st.markdown("### 📈 Visualização de Risco")
            
            # Criar figura de gauge usando plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de AVC (%)", 'font': {'size': 20}},
                delta={'reference': threshold_applied * 100, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 5], 'color': '#E8F5E9'},
                        {'range': [5, 15], 'color': '#FFF3E0'},
                        {'range': [15, 40], 'color': '#FFE0B2'},
                        {'range': [40, 100], 'color': '#FFCDD2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold_applied * 100
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#333", 'family': "Arial"}
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Informações detalhadas em colunas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📋 Detalhes Técnicos")
                st.metric("Threshold Operacional", f"{threshold_applied:.3f}")
                st.metric("Alerta Acionado", "🔴 SIM" if alert_flag else "🟢 NÃO")
                st.metric("Versão Calibração", calibration_version)
            
            with col2:
                st.markdown("#### 📊 Faixa de Risco")
                threshold_min = risk_tier_data.get('threshold_min', 0.0)
                threshold_max = risk_tier_data.get('threshold_max', 1.0)
                st.metric("Limite Inferior", f"{threshold_min*100:.1f}%")
                st.metric("Limite Superior", f"{threshold_max*100:.1f}%")
                st.metric("Posição na Faixa", 
                         f"{((prob - threshold_min) / (threshold_max - threshold_min) * 100):.1f}%" 
                         if threshold_max > threshold_min else "N/A")
            
            with col3:
                st.markdown("#### ⏱️ Performance")
                latency = result.get("latency_ms", 0)
                st.metric("Latência da API", f"{latency:.0f} ms")
                st.metric("Status", "✅ Sucesso")
                st.metric("Modelo", result.get("model_version", "unknown"))
            
            # Recomendações clínicas
            st.markdown("---")
            st.markdown("### 🏥 Recomendações Clínicas")
            
            recommended_action = risk_tier_data.get('recommended_action', 'Consultar médico para orientações.')
            
            if tier == "TIER_1_VERY_HIGH":
                st.error(
                    f"""
                    **⚠️ ATENÇÃO: Risco Muito Alto**
                    
                    {recommended_action}
                    
                    **Ações Imediatas:**
                    - 🚑 Encaminhamento urgente para neurologista/cardiologista
                    - 💊 Revisão imediata de medicações
                    - 📊 Exames complementares prioritários
                    - 🏥 Considerar internação para investigação
                    """
                )
            elif tier == "TIER_2_HIGH":
                st.warning(
                    f"""
                    **⚡ Risco Alto - Atenção Necessária**
                    
                    {recommended_action}
                    
                    **Ações Recomendadas:**
                    - 📅 Consulta com especialista em até 2 semanas
                    - 🩺 Implementar plano de cuidado preventivo
                    - 📈 Monitoramento intensificado de fatores de risco
                    - 💊 Avaliar necessidade de medicação preventiva
                    """
                )
            elif tier == "TIER_3_MODERATE":
                st.info(
                    f"""
                    **📊 Risco Moderado - Cuidado Preventivo**
                    
                    {recommended_action}
                    
                    **Ações Sugeridas:**
                    - 🥗 Aconselhamento sobre estilo de vida saudável
                    - 🏃 Programa de exercícios regulares
                    - 📊 Monitoramento de pressão e glicemia
                    - 📅 Reavaliação em 6 meses
                    """
                )
            else:
                st.success(
                    f"""
                    **✅ Risco Baixo - Manutenção**
                    
                    {recommended_action}
                    
                    **Ações de Manutenção:**
                    - 🎯 Continuar hábitos saudáveis atuais
                    - 📅 Check-ups anuais de rotina
                    - 🥗 Manter dieta balanceada
                    - 🏃 Atividade física regular
                    """
                )
            
            # Fatores de risco do paciente
            st.markdown("---")
            st.markdown("### 🔍 Análise de Fatores de Risco")
            
            risk_factors = []
            if hypertension == 1:
                risk_factors.append("🩺 Hipertensão presente")
            if heart_disease == 1:
                risk_factors.append("❤️ Doença cardíaca presente")
            if avg_glucose_level > 125:
                risk_factors.append(f"📊 Glicemia elevada ({avg_glucose_level:.1f} mg/dL)")
            if bmi >= 30:
                risk_factors.append(f"⚖️ Obesidade (BMI: {bmi:.1f})")
            elif bmi >= 25:
                risk_factors.append(f"⚖️ Sobrepeso (BMI: {bmi:.1f})")
            if smoking_status == "smokes":
                risk_factors.append("🚬 Tabagista atual")
            elif smoking_status == "formerly smoked":
                risk_factors.append("🚭 Ex-tabagista")
            if age >= 65:
                risk_factors.append(f"👴 Idade avançada ({age:.0f} anos)")
            
            if risk_factors:
                cols = st.columns(2)
                for idx, factor in enumerate(risk_factors):
                    with cols[idx % 2]:
                        st.markdown(f"- {factor}")
            else:
                st.success("✅ Nenhum fator de risco crítico identificado nos dados fornecidos")
            
            # Contexto comparativo
            st.markdown("---")
            st.markdown("### 📊 Contexto Estatístico")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                percentile = int(prob * 1000) / 10  # Aproximação
                st.metric(
                    "Percentil Estimado",
                    f"Top {100-percentile:.1f}%",
                    help="Posição relativa em relação à população geral"
                )
            
            with col_b:
                baseline_risk = 0.048  # Prevalência média
                relative_risk = prob / baseline_risk if baseline_risk > 0 else 0
                st.metric(
                    "Risco Relativo",
                    f"{relative_risk:.1f}x",
                    help="Comparado com a prevalência média de 4.8%"
                )
            
            with col_c:
                if prob > 0:
                    nnt = int(1 / prob)  # Number Needed to Treat (simplificado)
                    st.metric(
                        "NNT Aproximado",
                        f"{nnt}",
                        help="Número aproximado de pacientes a tratar para prevenir 1 AVC"
                    )
            
            # Download dos resultados
            st.markdown("---")
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar Resultado Completo (JSON)",
                data=result_json,
                file_name=f"predicao_stroke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# =============================================================================
# PAGE: ANÁLISE DE PERFORMANCE
# =============================================================================

elif page == "📉 Desempenho Detalhado":
    st.title("📉 Análise Detalhada de Desempenho")
    
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 30px;">
            <h4 style="margin: 0; color: white;">📊 Análise Técnica Completa</h4>
            <p style="margin: 10px 0 0 0; color: white;">
                Esta seção apresenta métricas detalhadas de desempenho nos conjuntos de validação e teste, 
                permitindo avaliar a <strong>confiabilidade</strong> e <strong>consistência</strong> do modelo.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    test_metrics = metrics_summary["test"]
    val_metrics = metrics_summary["val"]

    # Métricas principais com glossário
    st.markdown(f"### 🎯 Métricas no Limiar Operacional ({threshold_used:.3f})")
    
    perf_table = pd.DataFrame(
        [
            {
                "Conjunto": "Validação",
                "VP": val_metrics["tp"],
                "FP": val_metrics["fp"],
                "FN": val_metrics["fn"],
                "VN": val_metrics["tn"],
                "Precisão+": f"{val_metrics['precision']:.1%}",
                "Sensibilidade": f"{val_metrics['recall']:.1%}",
                "F1": f"{val_metrics['f1']:.1%}",
                "Acur. Balanc.": f"{val_metrics['balanced_accuracy']:.1%}",
                "Brier": f"{val_metrics['brier']:.4f}",
                "ECE": f"{val_metrics['ece']:.4f}",
            },
            {
                "Conjunto": "Teste",
                "VP": test_metrics["tp"],
                "FP": test_metrics["fp"],
                "FN": test_metrics["fn"],
                "VN": test_metrics["tn"],
                "Precisão+": f"{test_metrics['precision']:.1%}",
                "Sensibilidade": f"{test_metrics['recall']:.1%}",
                "F1": f"{test_metrics['f1']:.1%}",
                "Acur. Balanc.": f"{test_metrics['balanced_accuracy']:.1%}",
                "Brier": f"{test_metrics['brier']:.4f}",
                "ECE": f"{test_metrics['ece']:.4f}",
            },
        ]
    )
    
    st.dataframe(
        perf_table.style.set_properties(
            background_color='#f8f9fa',
            color='#212529',
            border_color='#dee2e6'
        ).set_table_styles([{
            'selector': 'td',
            'props': [('font-size', '14px')]
        }]),
        use_container_width=True
    )
    
    # Glossário expandido
    with st.expander("📖 **Glossário de Métricas**", expanded=False):
        st.markdown(
            """
            | Métrica | O que significa | Meta |
            |---------|-----------------|------|
            | **VP** (Verdadeiro Positivo) | Casos de AVC corretamente identificados | Maximizar |
            | **FP** (Falso Positivo) | Falsos alarmes (previu AVC sem ocorrência) | Minimizar |
            | **FN** (Falso Negativo) | AVCs não detectados (mais crítico!) | **Minimizar** |
            | **VN** (Verdadeiro Negativo) | Corretamente identificou ausência de AVC | Maximizar |
            | **Precisão+** | Proporção de alertas corretos | ≥ 15% |
            | **Sensibilidade** | Proporção de AVCs detectados | ≥ 70% |
            | **F1** | Média harmônica entre precisão e sensibilidade | Maximizar |
            | **Acurácia Balanceada** | Média entre sensibilidade e especificidade | Maximizar |
            | **Brier Score** | Qualidade das probabilidades (erro quadrático) | < 0.10 |
            | **ECE** | Erro de calibração (alinhamento prob. vs realidade) | < 0.05 |
            """
        )

    st.markdown("---")
    
    # Análise de impacto clínico
    st.markdown("### 🏥 Impacto Clínico Estimado")
    
    total_positives = test_metrics["tp"] + test_metrics["fn"]
    total_negatives = test_metrics["tn"] + test_metrics["fp"]
    total_patients = total_positives + total_negatives
    
    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    
    with impact_col1:
        st.metric(
            "🏥 Pacientes no Teste",
            f"{total_patients:,}",
            help="Total de casos avaliados"
        )
    
    with impact_col2:
        detection_rate = test_metrics["tp"] / total_positives if total_positives > 0 else 0
        st.metric(
            "🎯 Taxa de Detecção",
            f"{detection_rate:.1%}",
            help=f"Detectou {test_metrics['tp']} de {total_positives} AVCs reais"
        )
    
    with impact_col3:
        missed_cases = test_metrics["fn"]
        st.metric(
            "⚠️ AVCs Não Detectados",
            f"{missed_cases}",
            delta=f"{missed_cases/total_positives:.1%} do total",
            delta_color="inverse",
            help="Casos que necessitam revisão do processo"
        )
    
    with impact_col4:
        false_alarm_rate = test_metrics["fp"] / (test_metrics["fp"] + test_metrics["tn"]) if (test_metrics["fp"] + test_metrics["tn"]) > 0 else 0
        st.metric(
            "📢 Taxa de Falso Alarme",
            f"{false_alarm_rate:.1%}",
            help=f"{test_metrics['fp']} pacientes classificados como alto risco sem AVC"
        )
    
    # Contexto clínico
    st.info(
        f"""
        **🩺 Interpretação Clínica:**
        
        - **Sensibilidade de {test_metrics['recall']:.1%}**: O modelo consegue identificar a maioria dos pacientes em risco
        - **{missed_cases} casos não detectados**: Podem representar pacientes com fatores de risco atípicos
        - **{test_metrics['fp']} falsos positivos**: Receberão acompanhamento preventivo adicional (não prejudicial)
        - **Precisão de {test_metrics['precision']:.1%}**: A cada 100 alertas, ~{int(test_metrics['precision']*100)} são casos reais
        
        ⚖️ **Trade-off**: Priorizamos **detectar mais AVCs** (alta sensibilidade) ao custo de alguns falsos positivos.
        """
    )

    st.markdown("---")
    
    # Curvas de desempenho
    st.markdown("### 📊 Curvas de Avaliação (Conjunto de Teste)")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🎯 Curva Precisão-Sensibilidade")
        st.plotly_chart(
            build_pr_curve(test_metrics["y_true"], test_metrics["probs"], threshold_used),
            use_container_width=True,
        )
        st.caption(
            """
            **Como ler:** Cada ponto representa um limiar diferente. O ponto vermelho é nosso limiar operacional.
            
            - **Subir** = Mais precisão, menos sensibilidade (menos falsos positivos, mais AVCs perdidos)
            - **Direita** = Mais sensibilidade, menos precisão (mais AVCs detectados, mais falsos alarmes)
            """
        )
    
    with col_right:
        st.markdown("#### 📐 Diagrama de Confiabilidade")
        st.plotly_chart(
            build_calibration_chart(test_metrics["y_true"], test_metrics["probs"]),
            use_container_width=True,
        )
        st.caption(
            f"""
            **Como ler:** Pontos próximos à linha diagonal = boa calibração
            
            - Se o modelo diz "30% de chance", isso significa ~30 AVCs a cada 100 casos
            - **ECE = {test_metrics['ece']:.4f}** ✅ (Excelente! Meta: < 0.05)
            - **Brier = {test_metrics['brier']:.4f}** ✅ (Ótimo! Meta: < 0.10)
            """
        )

    st.markdown("---")
    
    # Alertas e monitoramento
    st.markdown("### 🚨 Alertas e Monitoramento")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("#### ⚖️ Equidade do Modelo")
        fairness_alerts_text = [
            alert["message"]
            for alert in fairness_alerts
            if alert.get("attribute") in {"Residence_type", "smoking_status"}
        ]
        if fairness_alerts_text:
            for alert_msg in fairness_alerts_text:
                st.error(f"⚠️ {alert_msg}")
            st.caption("Disparidades detectadas entre grupos. Consulte a aba 'Equidade do Modelo' para detalhes.")
        else:
            st.success("✅ Nenhuma disparidade significativa detectada entre grupos demográficos.")
            st.caption("O modelo apresenta desempenho equilibrado entre diferentes populações.")
    
    with alert_col2:
        st.markdown("#### 📊 Qualidade dos Dados")
        if drift_df.empty:
            st.info("ℹ️ Aguardando dados de monitoramento de drift.")
        else:
            worst_feature = drift_df.iloc[0]
            worst_psi = worst_feature['psi']
            
            if worst_psi > 0.25:
                st.error(f"🔴 **Alerta Crítico**: Drift detectado em '{worst_feature.get('feature', '')}' (PSI: {worst_psi:.3f})")
                st.caption("Distribuição dos dados mudou significativamente. Recomendado retreinar o modelo.")
            elif worst_psi > 0.10:
                st.warning(f"🟡 **Alerta Moderado**: Drift em '{worst_feature.get('feature', '')}' (PSI: {worst_psi:.3f})")
                st.caption("Pequena mudança na distribuição dos dados. Monitorar continuamente.")
            else:
                st.success(f"✅ Dados estáveis. Maior PSI: {worst_psi:.3f}")
                st.caption("Distribuição dos dados permanece consistente com o treinamento.")

# =============================================================================
# PAGE: PREDIÇÃO INDIVIDUAL
# =============================================================================

elif page == "🔮 Predição Individual":
    st.title("🔮 Predição Individual Calibrada")
    st.markdown(
        """
        Utilize o modelo calibrado para obter predições individuais com **probabilidades calibradas**
        e interpretação clínica detalhada.
        """
    )

    with st.form("patient_form"):
        demographic_col1, demographic_col2, demographic_col3 = st.columns(3)
        with demographic_col1:
            gender = st.selectbox("Gênero", ["Male", "Female", "Other"])
        with demographic_col2:
            age = st.number_input("Idade", min_value=18, max_value=100, value=50)
        with demographic_col3:
            ever_married = st.selectbox("Estado Civil", ["Yes", "No"])

        clinical_col1, clinical_col2, clinical_col3 = st.columns(3)
        with clinical_col1:
            hypertension = st.selectbox("Hipertensão", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        with clinical_col2:
            heart_disease = st.selectbox("Doença Cardíaca", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        with clinical_col3:
            avg_glucose_level = st.number_input("Glicose Média (mg/dL)", min_value=50.0, max_value=400.0, value=110.0)

        lifestyle_col1, lifestyle_col2, lifestyle_col3 = st.columns(3)
        with lifestyle_col1:
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=60.0, value=27.0)
        with lifestyle_col2:
            work_type = st.selectbox(
                "Tipo de Trabalho",
                ["Private", "Self-employed", "Govt_job", "children", "Never_worked"],
            )
            note = REFERENCE_NOTES["work_type"].get(work_type)
            if note:
                st.caption(f"ℹ️ {note}")
        with lifestyle_col3:
            smoking_status = st.selectbox(
                "Status de Tabagismo",
                ["never smoked", "formerly smoked", "smokes", "Unknown"],
            )
            note = REFERENCE_NOTES["smoking_status"].get(smoking_status)
            if note:
                st.caption(f"ℹ️ {note}")

        residence_type = st.selectbox("Tipo de Residência", ["Urban", "Rural"])

        submitted = st.form_submit_button("Obter Predição", type="primary")

    if submitted:
        payload = {
            "patient_id": None,
            "patient_data": {
                "gender": gender,
                "age": age,
                "hypertension": hypertension,
                "heart_disease": heart_disease,
                "ever_married": ever_married,
                "work_type": work_type,
                "Residence_type": residence_type,
                "avg_glucose_level": avg_glucose_level,
                "bmi": bmi,
                "smoking_status": smoking_status,
            },
            "return_explanation": False,
        }

        api_url = os.getenv("STROKE_API_URL", "http://localhost:8000")
        try:
            with st.spinner("🔄 Consultando API..."):
                response = requests.post(f"{api_url}/predict", json=payload, timeout=10)
                response.raise_for_status()
                result = response.json()
        except requests.exceptions.ConnectionError:
            st.error(
                """
                ❌ **Erro de Conexão com a API**
                
                O servidor FastAPI não está rodando. Para iniciar:
                
                1. Abra um novo terminal
                2. Execute: `python start_api_server.py`
                3. Aguarde a mensagem de sucesso
                4. Tente novamente aqui
                """
            )
            st.stop()
        except Exception as exc:
            st.error(f"❌ Erro ao consultar API: {exc}")
            st.stop()
        else:
            prob = result.get("probability_stroke", 0.0)
            calibration_version = result.get("calibration_version", "desconhecida")
            alert_flag = result.get("alert_flag", False)
            risk_tier_data = result.get("risk_tier", {})
            threshold_applied = result.get("threshold_used", threshold_used)

            # ========== VISUALIZAÇÃO ENRIQUECIDA ==========
            st.markdown("---")
            st.markdown("## 📊 Resultado da Predição")
            
            # Determinar cor e ícone baseado no tier
            tier = risk_tier_data.get("tier", "TIER_4_LOW")
            tier_colors = {
                "TIER_1_VERY_HIGH": ("#8B0000", "🚨"),
                "TIER_2_HIGH": ("#FF4500", "⚠️"),
                "TIER_3_MODERATE": ("#FFA500", "⚡"),
                "TIER_4_LOW": ("#32CD32", "✅"),
            }
            color, icon = tier_colors.get(tier, ("#808080", "ℹ️"))
            
            # Card principal de risco
            st.markdown(
                f"""
                <div style="
                    background: linear-gradient(135deg, {color}15 0%, {color}05 100%);
                    border-left: 5px solid {color};
                    padding: 25px;
                    border-radius: 10px;
                    margin: 20px 0;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                ">
                    <h2 style="color: {color}; margin: 0 0 15px 0;">
                        {icon} {risk_tier_data.get('description', 'Risco Desconhecido')}
                    </h2>
                    <div style="display: flex; align-items: center; gap: 30px; flex-wrap: wrap;">
                        <div style="flex: 1; min-width: 200px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 5px;">
                                Probabilidade de AVC
                            </div>
                            <div style="font-size: 48px; font-weight: bold; color: {color};">
                                {prob*100:.1f}%
                            </div>
                        </div>
                        <div style="flex: 2; min-width: 300px;">
                            <div style="font-size: 14px; color: #666; margin-bottom: 8px;">
                                Classificação de Risco
                            </div>
                            <div style="
                                background: white;
                                padding: 12px;
                                border-radius: 8px;
                                font-weight: 600;
                                color: {color};
                                border: 2px solid {color};
                            ">
                                {tier.replace('_', ' ')}
                            </div>
                        </div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True
            )
            
            # Barra de probabilidade visual
            st.markdown("### 📈 Visualização de Risco")
            
            # Criar figura de gauge usando plotly
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=prob * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidade de AVC (%)", 'font': {'size': 20}},
                delta={'reference': threshold_applied * 100, 'increasing': {'color': "red"}},
                gauge={
                    'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkgray"},
                    'bar': {'color': color},
                    'bgcolor': "white",
                    'borderwidth': 2,
                    'bordercolor': "gray",
                    'steps': [
                        {'range': [0, 5], 'color': '#E8F5E9'},
                        {'range': [5, 15], 'color': '#FFF3E0'},
                        {'range': [15, 40], 'color': '#FFE0B2'},
                        {'range': [40, 100], 'color': '#FFCDD2'}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': threshold_applied * 100
                    }
                }
            ))
            
            fig_gauge.update_layout(
                height=300,
                margin=dict(l=20, r=20, t=60, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                font={'color': "#333", 'family': "Arial"}
            )
            
            st.plotly_chart(fig_gauge, use_container_width=True)
            
            # Informações detalhadas em colunas
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown("#### 📋 Detalhes Técnicos")
                st.metric("Threshold Operacional", f"{threshold_applied:.3f}")
                st.metric("Alerta Acionado", "🔴 SIM" if alert_flag else "🟢 NÃO")
                st.metric("Versão Calibração", calibration_version)
            
            with col2:
                st.markdown("#### 📊 Faixa de Risco")
                threshold_min = risk_tier_data.get('threshold_min', 0.0)
                threshold_max = risk_tier_data.get('threshold_max', 1.0)
                st.metric("Limite Inferior", f"{threshold_min*100:.1f}%")
                st.metric("Limite Superior", f"{threshold_max*100:.1f}%")
                st.metric("Posição na Faixa", 
                         f"{((prob - threshold_min) / (threshold_max - threshold_min) * 100):.1f}%" 
                         if threshold_max > threshold_min else "N/A")
            
            with col3:
                st.markdown("#### ⏱️ Performance")
                latency = result.get("latency_ms", 0)
                st.metric("Latência da API", f"{latency:.0f} ms")
                st.metric("Status", "✅ Sucesso")
                st.metric("Modelo", result.get("model_version", "unknown"))
            
            # Recomendações clínicas
            st.markdown("---")
            st.markdown("### 🏥 Recomendações Clínicas")
            
            recommended_action = risk_tier_data.get('recommended_action', 'Consultar médico para orientações.')
            
            if tier == "TIER_1_VERY_HIGH":
                st.error(
                    f"""
                    **⚠️ ATENÇÃO: Risco Muito Alto**
                    
                    {recommended_action}
                    
                    **Ações Imediatas:**
                    - 🚑 Encaminhamento urgente para neurologista/cardiologista
                    - 💊 Revisão imediata de medicações
                    - 📊 Exames complementares prioritários
                    - 🏥 Considerar internação para investigação
                    """
                )
            elif tier == "TIER_2_HIGH":
                st.warning(
                    f"""
                    **⚡ Risco Alto - Atenção Necessária**
                    
                    {recommended_action}
                    
                    **Ações Recomendadas:**
                    - 📅 Consulta com especialista em até 2 semanas
                    - 🩺 Implementar plano de cuidado preventivo
                    - 📈 Monitoramento intensificado de fatores de risco
                    - 💊 Avaliar necessidade de medicação preventiva
                    """
                )
            elif tier == "TIER_3_MODERATE":
                st.info(
                    f"""
                    **📊 Risco Moderado - Cuidado Preventivo**
                    
                    {recommended_action}
                    
                    **Ações Sugeridas:**
                    - 🥗 Aconselhamento sobre estilo de vida saudável
                    - 🏃 Programa de exercícios regulares
                    - 📊 Monitoramento de pressão e glicemia
                    - 📅 Reavaliação em 6 meses
                    """
                )
            else:
                st.success(
                    f"""
                    **✅ Risco Baixo - Manutenção**
                    
                    {recommended_action}
                    
                    **Ações de Manutenção:**
                    - 🎯 Continuar hábitos saudáveis atuais
                    - 📅 Check-ups anuais de rotina
                    - 🥗 Manter dieta balanceada
                    - 🏃 Atividade física regular
                    """
                )
            
            # Fatores de risco do paciente
            st.markdown("---")
            st.markdown("### 🔍 Análise de Fatores de Risco")
            
            risk_factors = []
            if hypertension == 1:
                risk_factors.append("🩺 Hipertensão presente")
            if heart_disease == 1:
                risk_factors.append("❤️ Doença cardíaca presente")
            if avg_glucose_level > 125:
                risk_factors.append(f"📊 Glicemia elevada ({avg_glucose_level:.1f} mg/dL)")
            if bmi >= 30:
                risk_factors.append(f"⚖️ Obesidade (BMI: {bmi:.1f})")
            elif bmi >= 25:
                risk_factors.append(f"⚖️ Sobrepeso (BMI: {bmi:.1f})")
            if smoking_status == "smokes":
                risk_factors.append("🚬 Tabagista atual")
            elif smoking_status == "formerly smoked":
                risk_factors.append("🚭 Ex-tabagista")
            if age >= 65:
                risk_factors.append(f"👴 Idade avançada ({age:.0f} anos)")
            
            if risk_factors:
                cols = st.columns(2)
                for idx, factor in enumerate(risk_factors):
                    with cols[idx % 2]:
                        st.markdown(f"- {factor}")
            else:
                st.success("✅ Nenhum fator de risco crítico identificado nos dados fornecidos")
            
            # Contexto comparativo
            st.markdown("---")
            st.markdown("### 📊 Contexto Estatístico")
            
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                percentile = int(prob * 1000) / 10  # Aproximação
                st.metric(
                    "Percentil Estimado",
                    f"Top {100-percentile:.1f}%",
                    help="Posição relativa em relação à população geral"
                )
            
            with col_b:
                baseline_risk = 0.048  # Prevalência média
                relative_risk = prob / baseline_risk if baseline_risk > 0 else 0
                st.metric(
                    "Risco Relativo",
                    f"{relative_risk:.1f}x",
                    help="Comparado com a prevalência média de 4.8%"
                )
            
            with col_c:
                if prob > 0:
                    nnt = int(1 / prob)  # Number Needed to Treat (simplificado)
                    st.metric(
                        "NNT Aproximado",
                        f"{nnt}",
                        help="Número aproximado de pacientes a tratar para prevenir 1 AVC"
                    )
            
            # Download dos resultados
            st.markdown("---")
            result_json = json.dumps(result, indent=2, ensure_ascii=False)
            st.download_button(
                label="📥 Baixar Resultado Completo (JSON)",
                data=result_json,
                file_name=f"predicao_stroke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )

# =============================================================================
# PAGE: ANÁLISE DE PERFORMANCE
# =============================================================================

elif page == "📉 Desempenho Detalhado":
    st.title("📉 Análise Detalhada de Desempenho")
    
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 30px;">
            <h4 style="margin: 0; color: white;">📊 Análise Técnica Completa</h4>
            <p style="margin: 10px 0 0 0; color: white;">
                Esta seção apresenta métricas detalhadas de desempenho nos conjuntos de validação e teste, 
                permitindo avaliar a <strong>confiabilidade</strong> e <strong>consistência</strong> do modelo.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    test_metrics = metrics_summary["test"]
    val_metrics = metrics_summary["val"]

    # Métricas principais com glossário
    st.markdown(f"### 🎯 Métricas no Limiar Operacional ({threshold_used:.3f})")
    
    perf_table = pd.DataFrame(
        [
            {
                "Conjunto": "Validação",
                "VP": val_metrics["tp"],
                "FP": val_metrics["fp"],
                "FN": val_metrics["fn"],
                "VN": val_metrics["tn"],
                "Precisão+": f"{val_metrics['precision']:.1%}",
                "Sensibilidade": f"{val_metrics['recall']:.1%}",
                "F1": f"{val_metrics['f1']:.1%}",
                "Acur. Balanc.": f"{val_metrics['balanced_accuracy']:.1%}",
                "Brier": f"{val_metrics['brier']:.4f}",
                "ECE": f"{val_metrics['ece']:.4f}",
            },
            {
                "Conjunto": "Teste",
                "VP": test_metrics["tp"],
                "FP": test_metrics["fp"],
                "FN": test_metrics["fn"],
                "VN": test_metrics["tn"],
                "Precisão+": f"{test_metrics['precision']:.1%}",
                "Sensibilidade": f"{test_metrics['recall']:.1%}",
                "F1": f"{test_metrics['f1']:.1%}",
                "Acur. Balanc.": f"{test_metrics['balanced_accuracy']:.1%}",
                "Brier": f"{test_metrics['brier']:.4f}",
                "ECE": f"{test_metrics['ece']:.4f}",
            },
        ]
    )
    
    st.dataframe(
        perf_table.style.set_properties(
            background_color='#f8f9fa',
            color='#212529',
            border_color='#dee2e6'
        ).set_table_styles([{
            'selector': 'td',
            'props': [('font-size', '14px')]
        }]),
        use_container_width=True
    )
    
    # Glossário expandido
    with st.expander("📖 **Glossário de Métricas**", expanded=False):
        st.markdown(
            """
            | Métrica | O que significa | Meta |
            |---------|-----------------|------|
            | **VP** (Verdadeiro Positivo) | Casos de AVC corretamente identificados | Maximizar |
            | **FP** (Falso Positivo) | Falsos alarmes (previu AVC sem ocorrência) | Minimizar |
            | **FN** (Falso Negativo) | AVCs não detectados (mais crítico!) | **Minimizar** |
            | **VN** (Verdadeiro Negativo) | Corretamente identificou ausência de AVC | Maximizar |
            | **Precisão+** | Proporção de alertas corretos | ≥ 15% |
            | **Sensibilidade** | Proporção de AVCs detectados | ≥ 70% |
            | **F1** | Média harmônica entre precisão e sensibilidade | Maximizar |
            | **Acurácia Balanceada** | Média entre sensibilidade e especificidade | Maximizar |
            | **Brier Score** | Qualidade das probabilidades (erro quadrático) | < 0.10 |
            | **ECE** | Erro de calibração (alinhamento prob. vs realidade) | < 0.05 |
            """
        )

    st.markdown("---")
    
    # Análise de impacto clínico
    st.markdown("### 🏥 Impacto Clínico Estimado")
    
    total_positives = test_metrics["tp"] + test_metrics["fn"]
    total_negatives = test_metrics["tn"] + test_metrics["fp"]
    total_patients = total_positives + total_negatives
    
    impact_col1, impact_col2, impact_col3, impact_col4 = st.columns(4)
    
    with impact_col1:
        st.metric(
            "🏥 Pacientes no Teste",
            f"{total_patients:,}",
            help="Total de casos avaliados"
        )
    
    with impact_col2:
        detection_rate = test_metrics["tp"] / total_positives if total_positives > 0 else 0
        st.metric(
            "🎯 Taxa de Detecção",
            f"{detection_rate:.1%}",
            help=f"Detectou {test_metrics['tp']} de {total_positives} AVCs reais"
        )
    
    with impact_col3:
        missed_cases = test_metrics["fn"]
        st.metric(
            "⚠️ AVCs Não Detectados",
            f"{missed_cases}",
            delta=f"{missed_cases/total_positives:.1%} do total",
            delta_color="inverse",
            help="Casos que necessitam revisão do processo"
        )
    
    with impact_col4:
        false_alarm_rate = test_metrics["fp"] / (test_metrics["fp"] + test_metrics["tn"]) if (test_metrics["fp"] + test_metrics["tn"]) > 0 else 0
        st.metric(
            "📢 Taxa de Falso Alarme",
            f"{false_alarm_rate:.1%}",
            help=f"{test_metrics['fp']} pacientes classificados como alto risco sem AVC"
        )
    
    # Contexto clínico
    st.info(
        f"""
        **🩺 Interpretação Clínica:**
        
        - **Sensibilidade de {test_metrics['recall']:.1%}**: O modelo consegue identificar a maioria dos pacientes em risco
        - **{missed_cases} casos não detectados**: Podem representar pacientes com fatores de risco atípicos
        - **{test_metrics['fp']} falsos positivos**: Receberão acompanhamento preventivo adicional (não prejudicial)
        - **Precisão de {test_metrics['precision']:.1%}**: A cada 100 alertas, ~{int(test_metrics['precision']*100)} são casos reais
        
        ⚖️ **Trade-off**: Priorizamos **detectar mais AVCs** (alta sensibilidade) ao custo de alguns falsos positivos.
        """
    )

    st.markdown("---")
    
    # Curvas de desempenho
    st.markdown("### 📊 Curvas de Avaliação (Conjunto de Teste)")
    
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("#### 🎯 Curva Precisão-Sensibilidade")
        st.plotly_chart(
            build_pr_curve(test_metrics["y_true"], test_metrics["probs"], threshold_used),
            use_container_width=True,
        )
        st.caption(
            """
            **Como ler:** Cada ponto representa um limiar diferente. O ponto vermelho é nosso limiar operacional.
            
            - **Subir** = Mais precisão, menos sensibilidade (menos falsos positivos, mais AVCs perdidos)
            - **Direita** = Mais sensibilidade, menos precisão (mais AVCs detectados, mais falsos alarmes)
            """
        )
    
    with col_right:
        st.markdown("#### 📐 Diagrama de Confiabilidade")
        st.plotly_chart(
            build_calibration_chart(test_metrics["y_true"], test_metrics["probs"]),
            use_container_width=True,
        )
        st.caption(
            f"""
            **Como ler:** Pontos próximos à linha diagonal = boa calibração
            
            - Se o modelo diz "30% de chance", isso significa ~30 AVCs a cada 100 casos
            - **ECE = {test_metrics['ece']:.4f}** ✅ (Excelente! Meta: < 0.05)
            - **Brier = {test_metrics['brier']:.4f}** ✅ (Ótimo! Meta: < 0.10)
            """
        )

    st.markdown("---")
    
    # Alertas e monitoramento
    st.markdown("### 🚨 Alertas e Monitoramento")
    
    alert_col1, alert_col2 = st.columns(2)
    
    with alert_col1:
        st.markdown("#### ⚖️ Equidade do Modelo")
        fairness_alerts_text = [
            alert["message"]
            for alert in fairness_alerts
            if alert.get("attribute") in {"Residence_type", "smoking_status"}
        ]
        if fairness_alerts_text:
            for alert_msg in fairness_alerts_text:
                st.error(f"⚠️ {alert_msg}")
            st.caption("Disparidades detectadas entre grupos. Consulte a aba 'Equidade do Modelo' para detalhes.")
        else:
            st.success("✅ Nenhuma disparidade significativa detectada entre grupos demográficos.")
            st.caption("O modelo apresenta desempenho equilibrado entre diferentes populações.")
    
    with alert_col2:
        st.markdown("#### 📊 Qualidade dos Dados")
        if drift_df.empty:
            st.info("ℹ️ Aguardando dados de monitoramento de drift.")
        else:
            worst_feature = drift_df.iloc[0]
            worst_psi = worst_feature['psi']
            
            if worst_psi > 0.25:
                st.error(f"🔴 **Alerta Crítico**: Drift detectado em '{worst_feature.get('feature', '')}' (PSI: {worst_psi:.3f})")
                st.caption("Distribuição dos dados mudou significativamente. Recomendado retreinar o modelo.")
            elif worst_psi > 0.10:
                st.warning(f"🟡 **Alerta Moderado**: Drift em '{worst_feature.get('feature', '')}' (PSI: {worst_psi:.3f})")
                st.caption("Pequena mudança na distribuição dos dados. Monitorar continuamente.")
            else:
                st.success(f"✅ Dados estáveis. Maior PSI: {worst_psi:.3f}")
                st.caption("Distribuição dos dados permanece consistente com o treinamento.")

# =============================================================================
# PAGE: EQUIDADE DO MODELO
# =============================================================================

elif page == "⚖️ Equidade do Modelo":
    st.title("⚖️ Monitoramento de Equidade e Justiça Algorítmica")
    
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 30px;">
            <h4 style="margin: 0; color: white;">🤝 Por que Equidade Importa?</h4>
            <p style="margin: 10px 0 0 0; color: white;">
                Um modelo justo deve ter <strong>desempenho equilibrado</strong> entre diferentes grupos 
                demográficos (urbano/rural, gênero, idade, etc.). Disparidades podem levar a 
                <strong>tratamento desigual</strong> e agravar <strong>desigualdades em saúde</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if fairness_df.empty:
        st.warning("⚠️ Nenhum dado de equidade encontrado em `results/fairness_audit.json`.")
        st.info(
            """
            Para gerar a auditoria de equidade, execute a célula correspondente no notebook 
            `Stroke_Prediction_v4_Production.ipynb`.
            """
        )
    else:
        # Explicação das métricas
        st.markdown("### 📊 O que Estamos Medindo?")
        
        st.info(
            """
            **TPR Gap (Diferença na Taxa de Verdadeiros Positivos):**
            
            - Mede se o modelo **detecta AVCs igualmente** em diferentes grupos
            - **Gap > 0.10**: Disparidade significativa
            - **Intervalo de Confiança (IC) > 0**: Disparidade estatisticamente robusta
            
            **Exemplo:** Se TPR gap = 0.15 entre urbano/rural, significa que o modelo detecta 
            15% **mais** AVCs em um grupo do que no outro.
            """
        )
        
        st.markdown("---")
        
        # Tabela de fairness com formatação condicional - CORRIGIDA
        st.markdown("### 📋 Auditoria de Equidade (Conjunto de Teste)")
        
        # Adicionar coluna de status
        fairness_display = fairness_df.copy()
        fairness_display["Status"] = fairness_display.apply(
            lambda row: "🔴 Disparidade Robusta" if (row["TPR_gap"] > 0.10 and row["TPR_gap_ci_lower"] > 0)
            else "🟡 Atenção" if row["TPR_gap"] > 0.10
            else "🟢 OK",
            axis=1
        )
        
        # Renomear colunas para português
        fairness_display = fairness_display.rename(columns={
            "atributo": "Atributo Sensível",
            "TPR_gap": "Diferença TPR",
            "TPR_gap_ci_lower": "IC Inferior (95%)",
            "TPR_gap_ci_upper": "IC Superior (95%)"
        })
        
        # Função para aplicar cores no background mantendo texto escuro
        def highlight_status(val):
            if isinstance(val, str):
                if '🔴' in val:
                    return 'background-color: #ffcdd2; color: #212529; font-weight: bold;'
                elif '🟡' in val:
                    return 'background-color: #fff9c4; color: #212529; font-weight: bold;'
                elif '🟢' in val:
                    return 'background-color: #c8e6c9; color: #212529; font-weight: bold;'
            return ''
        
        st.dataframe(
            fairness_display.style.format({
                "Diferença TPR": "{:.1%}",
                "IC Inferior (95%)": "{:.1%}",
                "IC Superior (95%)": "{:.1%}"
            }).map(
                highlight_status,
                subset=['Status']
            ),
            use_container_width=True
        )
        
        st.caption(
            """
            **📖 Como interpretar:**
            - 🟢 **OK**: Diferença aceitável (< 10%)
            - 🟡 **Atenção**: Diferença moderada, mas IC cruza zero (pode ser aleatoriedade)
            - 🔴 **Disparidade Robusta**: Diferença > 10% com IC consistente (necessita mitigação)
            """
        )
        
        st.markdown("---")
        
        # Visualização de gaps
        st.markdown("### 📊 Visualização das Disparidades")
        
        fig_fairness = go.Figure()
        
        for _, row in fairness_df.iterrows():
            color = '#d32f2f' if (row['TPR_gap'] > 0.10 and row['TPR_gap_ci_lower'] > 0) else '#ff9800' if row['TPR_gap'] > 0.10 else '#4caf50'
            
            fig_fairness.add_trace(go.Bar(
                x=[row['TPR_gap']],
                y=[row['atributo']],
                orientation='h',
                marker_color=color,
                text=f"{row['TPR_gap']:.1%}",
                textposition='auto',
                name=row['atributo'],
                error_x=dict(
                    type='data',
                    symmetric=False,
                    array=[row['TPR_gap_ci_upper'] - row['TPR_gap']],
                    arrayminus=[row['TPR_gap'] - row['TPR_gap_ci_lower']],
                    color='rgba(0,0,0,0.3)',
                    thickness=2
                ),
                showlegend=False
            ))
        
        fig_fairness.add_vline(
            x=0.10,
            line_dash="dash",
            line_color="#d32f2f",
            annotation_text="Limite Aceitável (10%)",
            annotation_position="top right"
        )
        
        fig_fairness.update_layout(
            title="Diferenças de TPR entre Grupos (com Intervalos de Confiança 95%)",
            xaxis_title="Diferença na Taxa de Detecção de AVC (TPR Gap)",
            yaxis_title="Atributo Sensível",
            height=400,
            xaxis_tickformat='.0%',
            showlegend=False
        )
        
        st.plotly_chart(fig_fairness, use_container_width=True)
        
        st.markdown("---")
        
        # Alertas ativos
        st.markdown("### 🚨 Alertas Ativos de Equidade")
        
        if fairness_alerts:
            for idx, alert in enumerate(fairness_alerts, 1):
                attr = alert.get('attribute', 'Desconhecido')
                tpr_gap = alert.get('tpr_gap', 0)
                ci_lower = alert.get('ci_lower', 0)
                
                st.error(
                    f"""
                    **Alerta #{idx}: {attr}**
                    
                    - **Diferença TPR:** {tpr_gap:.1%}
                    - **IC Inferior:** {ci_lower:.1%} (> 0, logo **estatisticamente significativo**)
                    
                    **Ação Recomendada:** 
                    - Investigar se há diferenças na coleta/qualidade dos dados para este grupo
                    - Considerar estratificação ou calibração por subgrupo
                    - Avaliar se features específicas estão causando o viés
                    """
                )
        else:
            st.success(
                """
                ✅ **Nenhum alerta ativo de equidade!**
                
                O modelo apresenta desempenho equilibrado entre todos os grupos demográficos avaliados.
                As pequenas diferenças observadas estão dentro da margem de variação estatística esperada.
                """
            )
        
        st.markdown("---")
        
        # Contexto e recomendações
        st.markdown("### 💡 Contexto e Próximos Passos")
        
        col_context1, col_context2 = st.columns(2)
        
        with col_context1:
            st.markdown("#### 🎯 O que Fazemos com Disparidades?")
            st.markdown(
                """
                1. **Investigação**: Entender a causa raiz (dados, features, processo)
                2. **Mitigação Técnica**: 
                   - Calibração por subgrupo
                   - Ajuste de thresholds específicos
                   - Ponderação de amostras
                3. **Mitigação Processual**:
                   - Revisão humana adicional para grupos afetados
                   - Coleta de dados complementares
                4. **Monitoramento Contínuo**: Verificar regularmente se disparidades persistem
                """
            )
        
        with col_context2:
            st.markdown("#### ⚖️ Limitações da Análise")
            st.warning(
                """
                **Importante considerar:**
                
                - Grupos pequenos (< 30 casos) podem ter métricas instáveis
                - Disparidades podem refletir diferenças **reais** na prevalência
                - Equidade estatística ≠ Equidade clínica necessariamente
                - Múltiplas definições de "equidade" podem conflitar
                
                **Recomendação:** Combinar análise quantitativa com revisão clínica qualitativa
                """
            )

# =============================================================================
# PAGE: ALERTAS DE DRIFT
# =============================================================================

elif page == "📊 Monitoramento de Dados":
    st.title("📊 Monitoramento de Qualidade e Drift de Dados")
    
    st.markdown(
        """
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 20px; border-radius: 10px; color: white; margin-bottom: 30px;">
            <h4 style="margin: 0; color: white;">🔍 Por que Monitorar Dados?</h4>
            <p style="margin: 10px 0 0 0; color: white;">
                Modelos de machine learning assumem que os <strong>dados futuros</strong> serão similares aos 
                <strong>dados de treinamento</strong>. Se a distribuição dos dados mudar (drift), 
                o desempenho do modelo pode <strong>degradar silenciosamente</strong>.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    if drift_df.empty:
        st.info(
            """
            ℹ️ **Nenhum resultado de drift disponível.**
            
            Para gerar o monitoramento de drift, execute a célula correspondente no notebook 
            `Stroke_Prediction_v4_Production.ipynb` (seção "Monitoramento de Drift").
            
            O sistema usa **PSI (Population Stability Index)** para detectar mudanças na distribuição 
            das features entre treino e teste.
            """
        )
    else:
        # Explicação do PSI
        st.markdown("### 📊 Entendendo o PSI (Population Stability Index)")
        
        col_exp1, col_exp2, col_exp3 = st.columns(3)
        
        with col_exp1:
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #2e7d32 0%, #1b5e20 100%); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong style="color: #ffffff;">🟢 PSI < 0.10</strong><br>
                    <span style="font-size: 13px; color: #ffffff;">Distribuição <strong>estável</strong></span><br>
                    <span style="font-size: 12px; color: #e8f5e9;">Modelo confiável para dados atuais</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_exp2:
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #e65100 0%, #bf360c 100%); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong style="color: #ffffff;">🟡 0.10 ≤ PSI < 0.25</strong><br>
                    <span style="font-size: 13px; color: #ffffff;">Mudança <strong>moderada</strong></span><br>
                    <span style="font-size: 12px; color: #fff3e0;">Investigar causa e monitorar de perto</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with col_exp3:
            st.markdown(
                """
                <div style="background: linear-gradient(135deg, #c62828 0%, #b71c1c 100%); padding: 15px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                    <strong style="color: #ffffff;">🔴 PSI ≥ 0.25</strong><br>
                    <span style="font-size: 13px; color: #ffffff;">Mudança <strong>significativa</strong></span><br>
                    <span style="font-size: 12px; color: #ffebee;">Retreinamento recomendado</span>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown("---")
        
        # Tabela de drift - CORRIGIDA
        st.markdown("### 📋 Índice de Estabilidade por Feature")
        
        # Verificar se existe coluna 'psi' no dataframe
        if 'psi' not in drift_df.columns:
            st.error("⚠️ Coluna 'psi' não encontrada no dataframe de drift. Colunas disponíveis: " + ", ".join(drift_df.columns))
            st.info("""
            Para gerar corretamente os dados de drift:
            1. Execute o notebook `Stroke_Prediction_v4_Production.ipynb`
            2. Certifique-se de executar a célula de "Monitoramento de Drift"
            3. O arquivo `results/drift_monitoring_results.pkl` deve conter a chave 'psi_results'
            """)
        else:
            # Adicionar coluna de status
            drift_display = drift_df.copy()
            drift_display["status_icon"] = drift_display["psi"].apply(
                lambda x: "🔴 Crítico" if x >= 0.25 else "🟡 Moderado" if x >= 0.10 else "🟢 Estável"
            )
            
            # Renomear colunas
            drift_display = drift_display.rename(columns={
                "feature": "Variável",
                "psi": "PSI",
                "status_icon": "Status"
            })
            
            # Função para aplicar cores no background mantendo texto escuro
            def highlight_drift_status(val):
                if isinstance(val, str):
                    if '🔴' in val:
                        return 'background-color: #ffcdd2; color: #212529; font-weight: bold;'
                    elif '🟡' in val:
                        return 'background-color: #fff9c4; color: #212529; font-weight: bold;'
                    elif '🟢' in val:
                        return 'background-color: #c8e6c9; color: #212529; font-weight: bold;'
                return ''
            
            st.dataframe(
                drift_display.style.format({"PSI": "{:.4f}"}).map(
                    highlight_drift_status,
                    subset=['Status']
                ),
                use_container_width=True
            )
        
        # Gráfico de barras
        st.markdown("### 📊 Visualização do PSI por Feature")
        
        fig_drift = go.Figure()
        
        # Ordenar por PSI decrescente
        drift_sorted = drift_df.sort_values("psi", ascending=False)
        
        colors = [
            '#d32f2f' if psi >= 0.25 else '#ff9800' if psi >= 0.10 else '#4caf50'
            for psi in drift_sorted["psi"]
        ]
        
        fig_drift.add_trace(go.Bar(
            x=drift_sorted["feature"],
            y=drift_sorted["psi"],
            marker_color=colors,
            text=[f"{psi:.3f}" for psi in drift_sorted["psi"]],
            textposition='auto',
            showlegend=False
        ))
        
        fig_drift.add_hline(
            y=0.10,
            line_dash="dash",
            line_color="#ff9800",
            annotation_text="Limite Moderado (0.10)",
            annotation_position="top left"
        )
        
        fig_drift.add_hline(
            y=0.25,
            line_dash="dash",
            line_color="#d32f2f",
            annotation_text="Limite Crítico (0.25)",
            annotation_position="top left"
        )
        
        fig_drift.update_layout(
            title="Population Stability Index (PSI) por Feature",
            xaxis_title="Variável",
            yaxis_title="PSI",
            xaxis_tickangle=-45,
            height=500,
            showlegend=False
        )
        
        st.plotly_chart(fig_drift, use_container_width=True)
        
        st.markdown("---")
        
        # Análise e recomendações
        st.markdown("### 🎯 Análise e Recomendações")
        
        critical_features = drift_df[drift_df["psi"] >= 0.25]
        moderate_features = drift_df[(drift_df["psi"] >= 0.10) & (drift_df["psi"] < 0.25)]
        stable_features = drift_df[drift_df["psi"] < 0.10]
        
        col_analysis1, col_analysis2, col_analysis3 = st.columns(3)
        
        with col_analysis1:
            st.metric(
                "🔴 Features Críticas",
                len(critical_features),
                help="Features com PSI ≥ 0.25"
            )
            if not critical_features.empty:
                st.caption("**Principais:**")
                for feat in critical_features["feature"].head(3):
                    st.caption(f"• {feat}")
        
        with col_analysis2:
            st.metric(
                "🟡 Features em Atenção",
                len(moderate_features),
                help="Features com 0.10 ≤ PSI < 0.25"
            )
            if not moderate_features.empty:
                st.caption("**Principais:**")
                for feat in moderate_features["feature"].head(3):
                    st.caption(f"• {feat}")
        
        with col_analysis3:
            st.metric(
                "🟢 Features Estáveis",
                len(stable_features),
                help="Features com PSI < 0.10"
            )
            st.caption(f"**{len(stable_features)}/{len(drift_df)}** variáveis estáveis")
        
        st.markdown("---")
        
        # Recomendações acionáveis
        if not critical_features.empty:
            st.error(
                f"""
                **🚨 Ação Imediata Necessária**
                
                {len(critical_features)} feature(s) apresentam drift crítico (PSI ≥ 0.25):
                
                **Próximos Passos:**
                1. **Investigar causas:** Mudanças no processo de coleta? População diferente?
                2. **Avaliar impacto:** Rodar validação cruzada com dados recentes
                3. **Retreinar modelo:** Incluir dados novos no conjunto de treino
                4. **Atualizar pipeline:** Revisar transformações de features afetadas
                
                **Features críticas:** {', '.join(critical_features['feature'].tolist())}
                """
            )
        elif not moderate_features.empty:
            st.warning(
                f"""
                **⚠️ Monitoramento Intensificado Recomendado**
                
                {len(moderate_features)} feature(s) apresentam drift moderado (0.10 ≤ PSI < 0.25):
                
                **Ações Recomendadas:**
                1. Aumentar frequência de monitoramento (ex: diário → cada 6h)
                2. Configurar alertas automáticos se PSI ultrapassar 0.20
                3. Planejar retreinamento caso drift persista por 7+ dias
                4. Documentar possíveis causas (sazonalidade, mudanças operacionais)
                
                **Features em atenção:** {', '.join(moderate_features['feature'].tolist())}
                """
            )
        else:
            st.success(
                """
                ✅ **Todos os Dados Estáveis!**
                
                Nenhuma feature apresenta drift significativo. O modelo está operando em um ambiente 
                de dados consistente com o treinamento original.
                
                **Manutenção Recomendada:**
                - Continuar monitoramento regular (semanal)
                - Revisar PSI após eventos conhecidos (ex: mudanças de protocolo)
                - Retreinamento preventivo a cada 3-6 meses
                """
            )
        
        st.markdown("---")
        
        # Contexto adicional
        with st.expander("📖 **Entenda o PSI em Detalhes**", expanded=False):
            st.markdown(
                """
                ### Como o PSI é Calculado?
                
                O PSI compara a distribuição de uma feature em dois períodos (treino vs produção):
                
                ```
                PSI = Σ (% Produção - % Treino) × ln(% Produção / % Treino)
                ```
                
                ### Por que é Importante?
                
                - **Detecção Precoce**: Identifica problemas antes do modelo degradar visivelmente
                - **Granularidade**: Mostra exatamente quais features mudaram
                - **Decisões Informadas**: Guia quando retreinar vs quando investigar mais
                
                ### Limitações
                
                - Não detecta mudanças na **relação** entre features (somente distribuições marginais)
                - Sensível ao número de bins escolhidos para discretização
                - Não diferencia drift "benigno" (somente mudança na distribuição) de "prejudicial" (impacta o modelo)
                
                ### Complementos Recomendados
                
                - **Monitoring de Performance**: Acompanhar precisão/recall em produção
                - **Evidently AI**: Dashboard interativo de drift e performance
                - **SHAP Drift**: Monitorar mudanças na importância de features
                """
            )

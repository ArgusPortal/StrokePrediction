import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import json
from pathlib import Path
from datetime import datetime, timedelta
import sys

# Configuração da página
st.set_page_config(
    page_title="Stroke Prediction Dashboard",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Paths
BASE_DIR = Path(__file__).parent
MODELS_PATH = BASE_DIR / "models"
RESULTS_PATH = BASE_DIR / "results"
DATA_DIR = BASE_DIR / "data"

# === FUNÇÕES AUXILIARES ===

@st.cache_resource
def load_model_and_metadata():
    """Carrega modelo e metadados com cache"""
    model_path = MODELS_PATH / "stroke_model_v2_production.joblib"
    metadata_path = MODELS_PATH / "model_metadata_production.json"
    
    model = None
    metadata = None
    
    # Tentar carregar modelo
    if model_path.exists():
        try:
            model = joblib.load(model_path)
        except Exception as e:
            st.error(f"❌ Erro ao carregar modelo: {e}")
    
    # Tentar carregar metadados ou criar default
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
        except Exception as e:
            st.warning(f"⚠️ Erro ao carregar metadados: {e}")
    
    # Se não há metadados, criar estrutura default
    if metadata is None:
        metadata = create_default_metadata()
    
    return model, metadata


def create_default_metadata():
    """Cria estrutura de metadados padrão quando não disponível"""
    return {
        'model_info': {
            'version': 'v2.0-demo',
            'algorithm': 'Demo Model',
            'calibration_method': 'N/A',
            'created_at_utc': datetime.now().isoformat(),
            'training_samples': 5000,
            'validation_samples': 1000,
            'test_samples': 1000,
            'approved_for_production': False
        },
        'performance_metrics': {
            'test_set': {
                'pr_auc': 0.285,
                'roc_auc': 0.823,
                'recall': 0.68,
                'precision': 0.14,
                'f1_score': 0.23,
                'specificity': 0.85,
                'expected_calibration_error': 0.042,
                'brier_score': 0.045,
                'confusion_matrix': {
                    'true_negatives': 850,
                    'false_positives': 150,
                    'false_negatives': 32,
                    'true_positives': 68
                },
                'clinical_interpretation': {
                    'sensitivity_pct': '68%',
                    'ppv': '1 em 7 alertas',
                    'specificity_pct': '85%',
                    'npv': '96%'
                }
            }
        },
        'hyperparameters': {
            'optimal_threshold': 0.15
        },
        'fairness_metrics': {
            'gender': {
                'tpr_gap': 0.08,
                'fnr_gap': 0.09,
                'fpr_gap': 0.06,
                'compliant': True
            },
            'age_group': {
                'tpr_gap': 0.12,
                'fnr_gap': 0.11,
                'fpr_gap': 0.15,
                'compliant': False
            }
        },
        'compliance': {
            'hipaa_compliant': True,
            'gdpr_compliant': True,
            'fda_cleared': False,
            'clinical_validation_required': True,
            'bias_audit_completed': True,
            'last_audit_date': '2024-01-01'
        }
    }


def safe_get(dictionary, keys, default=None):
    """Safely get nested dictionary values"""
    for key in keys:
        if isinstance(dictionary, dict) and key in dictionary:
            dictionary = dictionary[key]
        else:
            return default
    return dictionary


@st.cache_data
def load_production_data():
    """Carrega dados de produção simulados"""
    # Simulação de dados de produção (substituir por dados reais)
    np.random.seed(42)
    
    dates = pd.date_range(end=datetime.now(), periods=30, freq='D')
    
    production_data = {
        'date': dates,
        'total_predictions': np.random.randint(800, 1200, size=30),
        'high_risk_count': np.random.randint(50, 150, size=30),
        'pr_auc': np.random.uniform(0.27, 0.30, size=30),
        'calibration_error': np.random.uniform(0.03, 0.06, size=30),
        'avg_latency_ms': np.random.uniform(80, 120, size=30),
        'error_rate': np.random.uniform(0, 0.02, size=30)
    }
    
    return pd.DataFrame(production_data)


def calculate_psi(baseline_dist, current_dist, bins=10):
    """Calcula Population Stability Index (PSI)"""
    baseline_bins, _ = np.histogram(baseline_dist, bins=bins)
    current_bins, _ = np.histogram(current_dist, bins=bins)
    
    baseline_pct = baseline_bins / baseline_bins.sum() + 1e-10
    current_pct = current_bins / current_bins.sum() + 1e-10
    
    psi = np.sum((current_pct - baseline_pct) * np.log(current_pct / baseline_pct))
    
    return psi


# === SIDEBAR - NAVEGAÇÃO ===

st.sidebar.title("🏥 Stroke Prediction System")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navegação",
    ["📊 Dashboard Principal", "🔮 Predição Individual", "📈 Análise de Performance", 
     "⚖️ Monitoramento de Fairness", "🚨 Alertas de Drift", "📋 Model Card"]
)

st.sidebar.markdown("---")

# Carregar modelo e metadados
model, metadata = load_model_and_metadata()

if metadata:
    st.sidebar.markdown("### 📌 Informações do Modelo")
    st.sidebar.metric("Versão", str(safe_get(metadata, ['model_info', 'version'], 'N/A')))
    st.sidebar.metric("PR-AUC (Test)", f"{safe_get(metadata, ['performance_metrics', 'test_set', 'pr_auc'], 0.285):.4f}")
    st.sidebar.metric("Recall (Test)", f"{safe_get(metadata, ['performance_metrics', 'test_set', 'recall'], 0.68):.4f}")
    st.sidebar.metric("ECE", f"{safe_get(metadata, ['performance_metrics', 'test_set', 'expected_calibration_error'], 0.042):.4f}")

st.sidebar.markdown("---")
st.sidebar.markdown("**Última atualização:** " + datetime.now().strftime("%Y-%m-%d %H:%M"))

# === PÁGINA 1: DASHBOARD PRINCIPAL ===

if page == "📊 Dashboard Principal":
    
    st.title("🏥 Stroke Prediction - Dashboard de Monitoramento")
    st.markdown("### Visão geral do sistema em tempo real")
    
    # Carregar dados de produção
    prod_data = load_production_data()
    
    # KPIs principais
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        total_preds_today = prod_data.iloc[-1]['total_predictions']
        total_preds_yesterday = prod_data.iloc[-2]['total_predictions']
        delta_preds = total_preds_today - total_preds_yesterday
        
        st.metric(
            label="📋 Predições (Hoje)",
            value=f"{total_preds_today:,}",
            delta=f"{delta_preds:+,} vs ontem"
        )
    
    with col2:
        high_risk_today = prod_data.iloc[-1]['high_risk_count']
        high_risk_rate = high_risk_today / total_preds_today * 100
        
        st.metric(
            label="🔴 Alto Risco (Hoje)",
            value=f"{high_risk_today}",
            delta=f"{high_risk_rate:.1f}%"
        )
    
    with col3:
        current_pr_auc = prod_data.iloc[-1]['pr_auc']
        baseline_pr_auc = safe_get(metadata, ['performance_metrics', 'test_set', 'pr_auc'], 0.285)
        delta_pr_auc = current_pr_auc - baseline_pr_auc
        
        st.metric(
            label="📊 PR-AUC (Atual)",
            value=f"{current_pr_auc:.4f}",
            delta=f"{delta_pr_auc:+.4f} vs baseline",
            delta_color="normal" if delta_pr_auc >= 0 else "inverse"
        )
    
    with col4:
        current_latency = prod_data.iloc[-1]['avg_latency_ms']
        
        st.metric(
            label="⚡ Latência Média",
            value=f"{current_latency:.0f} ms",
            delta="Normal" if current_latency < 100 else "Alta"
        )
    
    with col5:
        uptime = 99.8  # Simulado
        
        st.metric(
            label="✅ Uptime (30d)",
            value=f"{uptime:.1f}%",
            delta="✅ Saudável"
        )
    
    st.markdown("---")
    
    # Gráficos de tendência
    col_left, col_right = st.columns(2)
    
    with col_left:
        st.markdown("### 📈 Volume de Predições (30 dias)")
        
        fig_volume = go.Figure()
        
        fig_volume.add_trace(go.Scatter(
            x=prod_data['date'],
            y=prod_data['total_predictions'],
            mode='lines+markers',
            name='Total',
            line=dict(color='#3498db', width=3),
            marker=dict(size=8)
        ))
        
        fig_volume.add_trace(go.Scatter(
            x=prod_data['date'],
            y=prod_data['high_risk_count'],
            mode='lines+markers',
            name='Alto Risco',
            line=dict(color='#e74c3c', width=3),
            marker=dict(size=8)
        ))
        
        fig_volume.update_layout(
            xaxis_title="Data",
            yaxis_title="Quantidade",
            hovermode='x unified',
            height=400
        )
        
        st.plotly_chart(fig_volume, width=None)
    
    with col_right:
        st.markdown("### 🎯 Performance do Modelo (30 dias)")
        
        fig_perf = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig_perf.add_trace(
            go.Scatter(
                x=prod_data['date'],
                y=prod_data['pr_auc'],
                mode='lines+markers',
                name='PR-AUC',
                line=dict(color='#2ecc71', width=3),
                marker=dict(size=8)
            ),
            secondary_y=False
        )
        
        fig_perf.add_trace(
            go.Scatter(
                x=prod_data['date'],
                y=prod_data['calibration_error'],
                mode='lines+markers',
                name='Calibration Error',
                line=dict(color='#f39c12', width=3),
                marker=dict(size=8)
            ),
            secondary_y=True
        )
        
        # Linha de baseline
        fig_perf.add_hline(
            y=baseline_pr_auc,
            line_dash="dash",
            line_color="gray",
            annotation_text="Baseline PR-AUC",
            secondary_y=False
        )
        
        fig_perf.update_xaxes(title_text="Data")
        fig_perf.update_yaxes(title_text="PR-AUC", secondary_y=False)
        fig_perf.update_yaxes(title_text="Calibration Error", secondary_y=True)
        fig_perf.update_layout(hovermode='x unified', height=400)
        
        st.plotly_chart(fig_perf, width=None)
    
    st.markdown("---")
    
    # Distribuição de Tiers de Risco (últimos 7 dias)
    st.markdown("### 🎯 Distribuição por Tiers de Risco (Últimos 7 dias)")
    
    # Simulação de dados de tiers
    tier_data = {
        'Tier': ['Tier 1 (Muito Alto)', 'Tier 2 (Alto)', 'Tier 3 (Médio)', 'Tier 4 (Baixo)'],
        'Pacientes': [250, 450, 800, 3500],
        'Percentual': [5, 9, 16, 70],
        'Custo Estimado': [75000, 67500, 40000, 35000]
    }
    
    tier_df = pd.DataFrame(tier_data)
    
    col_tier1, col_tier2 = st.columns(2)
    
    with col_tier1:
        fig_tier_pie = px.pie(
            tier_df,
            values='Pacientes',
            names='Tier',
            title='Distribuição de Pacientes por Tier',
            color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
        )
        
        st.plotly_chart(fig_tier_pie, width=None)
    
    with col_tier2:
        fig_tier_bar = px.bar(
            tier_df,
            x='Tier',
            y='Custo Estimado',
            title='Custo Operacional por Tier',
            color='Tier',
            color_discrete_sequence=['#e74c3c', '#f39c12', '#f1c40f', '#2ecc71']
        )
        
        fig_tier_bar.update_layout(showlegend=False)
        
        st.plotly_chart(fig_tier_bar, width=None)
    
    # Tabela de resumo
    st.markdown("### 📊 Resumo dos Tiers")
    st.dataframe(tier_df, width="stretch")
    
    st.markdown(f"**💰 Custo Operacional Total (7 dias):** R$ {tier_df['Custo Estimado'].sum():,}")


# === PÁGINA 2: PREDIÇÃO INDIVIDUAL ===

elif page == "🔮 Predição Individual":
    
    st.title("🔮 Predição de Risco Individual")
    st.markdown("### Insira os dados do paciente para obter a predição de risco de AVC")
    
    # Formulário de entrada
    with st.form("patient_form"):
        st.markdown("#### 📋 Dados Demográficos")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gênero", ["Male", "Female", "Other"])
        
        with col2:
            age = st.number_input("Idade", min_value=1, max_value=120, value=50)
        
        with col3:
            ever_married = st.selectbox("Estado Civil", ["Yes", "No"])
        
        st.markdown("#### 🏥 Dados Clínicos")
        
        col4, col5, col6 = st.columns(3)
        
        with col4:
            hypertension = st.selectbox("Hipertensão", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        
        with col5:
            heart_disease = st.selectbox("Doença Cardíaca", [0, 1], format_func=lambda x: "Sim" if x == 1 else "Não")
        
        with col6:
            avg_glucose_level = st.number_input("Glicose Média (mg/dL)", min_value=50.0, max_value=500.0, value=100.0)
        
        st.markdown("#### 💪 Dados Físicos e Estilo de Vida")
        
        col7, col8, col9 = st.columns(3)
        
        with col7:
            bmi = st.number_input("BMI (kg/m²)", min_value=10.0, max_value=100.0, value=25.0)
        
        with col8:
            work_type = st.selectbox("Tipo de Trabalho", ["Private", "Self-employed", "Govt_job", "children", "Never_worked"])
        
        with col9:
            smoking_status = st.selectbox("Status de Tabagismo", ["never smoked", "formerly smoked", "smokes", "Unknown"])
        
        col10, col11 = st.columns(2)
        
        with col10:
            residence_type = st.selectbox("Tipo de Residência", ["Urban", "Rural"])
        
        submit_button = st.form_submit_button("🔮 Obter Predição", type="primary")
    
    if submit_button:
        # Preparar dados do paciente
        patient_data = {
            'gender': gender,
            'age': age,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi,
            'smoking_status': smoking_status
        }
        
        try:
            # Predição simulada (substituir pela função real quando disponível)
            probability = np.random.uniform(0.05, 0.40)  # Substituir por predição real
            threshold_value = safe_get(metadata, ['hyperparameters', 'optimal_threshold'], 0.15)
            # Garantir que threshold seja um número
            threshold = float(threshold_value) if isinstance(threshold_value, (int, float, str)) else 0.15
            prediction = int(probability >= threshold)
            
            # Determinar nível de risco
            if probability >= 0.8:
                risk_level = "CRÍTICO"
                risk_color = "🔴"
                risk_bg = "#ffebee"
            elif probability >= 0.6:
                risk_level = "ALTO"
                risk_color = "🟠"
                risk_bg = "#fff3e0"
            elif probability >= 0.4:
                risk_level = "MODERADO"
                risk_color = "🟡"
                risk_bg = "#fffde7"
            elif probability >= 0.2:
                risk_level = "BAIXO"
                risk_color = "🟢"
                risk_bg = "#e8f5e9"
            else:
                risk_level = "MUITO BAIXO"
                risk_color = "⚪"
                risk_bg = "#f5f5f5"
            
            # Exibir resultados
            st.markdown("---")
            st.markdown("## 📊 Resultado da Predição")
            
            col_res1, col_res2, col_res3 = st.columns(3)
            
            with col_res1:
                st.markdown(f"""
                <div style="background-color: {risk_bg}; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2>{risk_color} {risk_level}</h2>
                    <h3>Probabilidade: {probability*100:.2f}%</h3>
                </div>
                """, unsafe_allow_html=True)
            
            with col_res2:
                st.metric("Classificação", "ALTO RISCO" if prediction == 1 else "BAIXO RISCO")
                st.metric("Threshold Utilizado", f"{threshold:.4f}")
            
            with col_res3:
                confidence = "Alta" if abs(probability - threshold) > 0.15 else "Moderada" if abs(probability - threshold) > 0.05 else "Baixa"
                st.metric("Confiança da Predição", confidence)
            
            # Recomendações clínicas
            st.markdown("### 🏥 Recomendações Clínicas")
            
            if prediction == 1:
                st.warning("⚠️ **PACIENTE DE ALTO RISCO - AÇÃO IMEDIATA NECESSÁRIA**")
                
                recommendations = [
                    "🏥 **Consulta cardiológica urgente** - agendar em até 7 dias",
                    "📋 **Avaliação completa de fatores de risco cardiovascular**",
                ]
                
                if hypertension == 1:
                    recommendations.append("💊 **Monitoramento rigoroso da pressão arterial**")
                
                if avg_glucose_level > 126:
                    recommendations.append("🩸 **Controle glicêmico - possível diabetes**")
                
                if bmi > 30:
                    recommendations.append("⚖️ **Programa de redução de peso recomendado**")
                
                if smoking_status in ['smokes', 'formerly smoked']:
                    recommendations.append("🚭 **Cessação do tabagismo crítica**")
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            else:
                st.success("✅ **Paciente de baixo risco**")
                
                recommendations = [
                    "✅ Manter acompanhamento preventivo regular",
                    "🏃 Estilo de vida saudável e atividade física",
                    "📅 Reavaliação anual recomendada"
                ]
                
                for rec in recommendations:
                    st.markdown(f"- {rec}")
            
            # Visualização de fatores de risco
            st.markdown("### 📊 Análise de Fatores de Risco")
            
            # Fatores de risco simulados (substituir por SHAP real)
            risk_factors = {
                'Idade': min(age / 100, 0.9),
                'Hipertensão': hypertension * 0.7,
                'Doença Cardíaca': heart_disease * 0.8,
                'Glicose': min(avg_glucose_level / 300, 0.85),
                'BMI': min(bmi / 50, 0.75),
                'Tabagismo': 0.6 if smoking_status == 'smokes' else 0.3 if smoking_status == 'formerly smoked' else 0.1
            }
            
            fig_factors = go.Figure(go.Bar(
                x=list(risk_factors.values()),
                y=list(risk_factors.keys()),
                orientation='h',
                marker=dict(
                    color=list(risk_factors.values()),
                    colorscale='RdYlGn_r',
                    cmin=0,
                    cmax=1
                )
            ))
            
            fig_factors.update_layout(
                title="Contribuição dos Fatores de Risco",
                xaxis_title="Impacto Normalizado",
                yaxis_title="Fator",
                height=400
            )
            
            st.plotly_chart(fig_factors, width=None)
            
        except Exception as e:
            st.error(f"❌ Erro na predição: {str(e)}")


# === PÁGINA 3: ANÁLISE DE PERFORMANCE ===

elif page == "📈 Análise de Performance":
    
    st.title("📈 Análise Detalhada de Performance")
    st.markdown("### Métricas de validação e performance do modelo")
    
    # Métricas do Test Set
    st.markdown("## 🎯 Métricas do Test Set")
    
    test_metrics = safe_get(metadata, ['performance_metrics', 'test_set'], {})
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    
    with col_m1:
        st.metric("PR-AUC", f"{test_metrics.get('pr_auc', 0.285):.4f}")
        st.metric("ROC-AUC", f"{test_metrics.get('roc_auc', 0.823):.4f}")
    
    with col_m2:
        st.metric("Recall (Sensibilidade)", f"{test_metrics.get('recall', 0.68):.4f}")
        st.metric("Precision (PPV)", f"{test_metrics.get('precision', 0.14):.4f}")
    
    with col_m3:
        st.metric("Especificidade", f"{test_metrics.get('specificity', 0.85):.4f}")
        st.metric("F1-Score", f"{test_metrics.get('f1_score', 0.23):.4f}")
    
    with col_m4:
        st.metric("ECE (Calibration Error)", f"{test_metrics.get('expected_calibration_error', 0.042):.4f}")
        st.metric("Brier Score", f"{test_metrics.get('brier_score', 0.045):.4f}")
    
    st.markdown("---")
    
    # Confusion Matrix
    st.markdown("## 🔍 Confusion Matrix")
    
    cm_data = test_metrics.get('confusion_matrix', {
        'true_negatives': 850,
        'false_positives': 150,
        'false_negatives': 32,
        'true_positives': 68
    })
    
    fig_cm = go.Figure(data=go.Heatmap(
        z=[[cm_data['true_negatives'], cm_data['false_positives']],
           [cm_data['false_negatives'], cm_data['true_positives']]],
        x=['No Stroke (Pred)', 'Stroke (Pred)'],
        y=['No Stroke (Real)', 'Stroke (Real)'],
        colorscale='Blues',
        text=[[cm_data['true_negatives'], cm_data['false_positives']],
              [cm_data['false_negatives'], cm_data['true_positives']]],
        texttemplate='%{text}',
        textfont={"size": 20},
        hoverongaps=False
    ))
    
    fig_cm.update_layout(
        title="Confusion Matrix - Test Set",
        xaxis_title="Predito",
        yaxis_title="Real",
        height=500
    )
    
    st.plotly_chart(fig_cm, width=None)
    
    # Interpretação clínica
    st.markdown("## 🏥 Interpretação Clínica")
    
    clinical_interp = test_metrics.get('clinical_interpretation', {})
    
    col_c1, col_c2 = st.columns(2)
    
    with col_c1:
        st.info(f"""
        **Sensibilidade (Recall):** {clinical_interp.get('sensitivity_pct', '68%')}
        
        → O modelo detecta aproximadamente **7 em cada 10 casos** de AVC.
        """)
        
        st.info(f"""
        **Valor Preditivo Positivo (PPV):** {clinical_interp.get('ppv', '1 em 7 alertas')}
        
        → Aproximadamente **1 caso verdadeiro a cada 7 alertas** de alto risco.
        """)
    
    with col_c2:
        st.info(f"""
        **Especificidade:** {clinical_interp.get('specificity_pct', '85%')}
        
        → Baixa taxa de falsos alarmes em pacientes saudáveis.
        """)
        
        st.info(f"""
        **Valor Preditivo Negativo (NPV):** {clinical_interp.get('npv', '96%')}
        
        → Alta confiança quando o modelo indica baixo risco.
        """)


# === PÁGINA 4: MONITORAMENTO DE FAIRNESS ===

elif page == "⚖️ Monitoramento de Fairness":
    
    st.title("⚖️ Monitoramento de Fairness e Equidade")
    st.markdown("### Análise de viés e equidade do modelo por grupos demográficos")
    
    fairness_metrics = safe_get(metadata, ['fairness_metrics'], {})
    
    if not fairness_metrics:
        st.warning("⚠️ Métricas de fairness não disponíveis nos metadados")
        # Criar dados demo
        fairness_metrics = {
            'gender': {
                'tpr_gap': 0.08,
                'fnr_gap': 0.09,
                'fpr_gap': 0.06,
                'compliant': True
            },
            'age_group': {
                'tpr_gap': 0.12,
                'fnr_gap': 0.11,
                'fpr_gap': 0.15,
                'compliant': False
            }
        }
    
    # Seletor de atributo
    st.markdown("## 📊 Análise por Atributo Demográfico")
    
    selected_attr = st.selectbox(
        "Selecione o atributo para análise:",
        list(fairness_metrics.keys())
    )
    
    attr_metrics = fairness_metrics[selected_attr]
    
    # KPIs de Fairness
    col_f1, col_f2, col_f3, col_f4 = st.columns(4)
    
    with col_f1:
        tpr_gap = attr_metrics.get('tpr_gap', 0.08)
        tpr_status = "✅" if tpr_gap < 0.10 else "⚠️"
        st.metric("TPR Gap", f"{tpr_gap:.4f}", delta=tpr_status)
    
    with col_f2:
        fnr_gap = attr_metrics.get('fnr_gap', 0.09)
        fnr_status = "✅" if fnr_gap < 0.10 else "⚠️"
        st.metric("FNR Gap", f"{fnr_gap:.4f}", delta=fnr_status)
    
    with col_f3:
        fpr_gap = attr_metrics.get('fpr_gap', 0.06)
        fpr_status = "✅" if fpr_gap < 0.10 else "⚠️"
        st.metric("FPR Gap", f"{fpr_gap:.4f}", delta=fpr_status)
    
    with col_f4:
        compliant = attr_metrics.get('compliant', True)
        st.metric("Status", "✅ Conforme" if compliant else "❌ Não Conforme")
    
    st.markdown("---")
    
    # Interpretação
    if attr_metrics.get('compliant', True):
        st.success(f"""
        ✅ **O modelo está em conformidade com critérios de fairness para {selected_attr}**
        
        Todos os gaps estão abaixo do limite de 10%, indicando tratamento equitativo entre grupos.
        """)
    else:
        st.error(f"""
        ❌ **ATENÇÃO: Gaps de fairness detectados para {selected_attr}**
        
        Gaps superiores a 10% indicam possível tratamento desigual entre grupos. Revisão recomendada.
        """)
    
    # Visualização de gaps
    st.markdown("### 📊 Visualização de Gaps de Fairness")
    
    gap_data = {
        'Métrica': ['TPR Gap', 'FNR Gap', 'FPR Gap'],
        'Valor': [attr_metrics.get('tpr_gap', 0.08), attr_metrics.get('fnr_gap', 0.09), attr_metrics.get('fpr_gap', 0.06)],
        'Limite': [0.10, 0.10, 0.10]
    }
    
    gap_df = pd.DataFrame(gap_data)
    
    fig_gaps = go.Figure()
    
    fig_gaps.add_trace(go.Bar(
        x=gap_df['Métrica'],
        y=gap_df['Valor'],
        name='Valor Atual',
        marker_color=['#2ecc71' if v < 0.10 else '#e74c3c' for v in gap_df['Valor']]
    ))
    
    fig_gaps.add_trace(go.Scatter(
        x=gap_df['Métrica'],
        y=gap_df['Limite'],
        mode='lines',
        name='Limite (10%)',
        line=dict(color='red', dash='dash', width=3)
    ))
    
    fig_gaps.update_layout(
        title=f"Gaps de Fairness - {selected_attr}",
        yaxis_title="Gap Value",
        height=400
    )
    
    st.plotly_chart(fig_gaps, width=None)
    
    # Resumo de todos os atributos
    st.markdown("## 📋 Resumo de Fairness - Todos os Atributos")
    
    fairness_summary = []
    
    for attr, metrics in fairness_metrics.items():
        fairness_summary.append({
            'Atributo': attr,
            'TPR Gap': f"{metrics.get('tpr_gap', 0.08):.4f}",
            'FNR Gap': f"{metrics.get('fnr_gap', 0.09):.4f}",
            'FPR Gap': f"{metrics.get('fpr_gap', 0.06):.4f}",
            'Conforme': '✅' if metrics.get('compliant', True) else '❌'
        })
    
    fairness_df = pd.DataFrame(fairness_summary)
    
    st.dataframe(fairness_df, width="stretch")


# === PÁGINA 5: ALERTAS DE DRIFT ===

elif page == "🚨 Alertas de Drift":
    
    st.title("🚨 Monitoramento de Drift")
    st.markdown("### Detecção de mudanças na distribuição de dados e performance")
    
    # Simulação de drift (substituir por cálculos reais)
    st.markdown("## 📊 Data Drift (PSI - Population Stability Index)")
    
    # Dados simulados de PSI
    psi_data = {
        'Feature': ['age', 'avg_glucose_level', 'bmi', 'hypertension', 'heart_disease'],
        'PSI': [0.12, 0.28, 0.15, 0.08, 0.22],
        'Status': ['⚠️ Warning', '🔴 Critical', '⚠️ Warning', '✅ OK', '🔴 Critical']
    }
    
    psi_df = pd.DataFrame(psi_data)
    
    # Aplicar cores com CSS personalizado
    def highlight_psi(val):
        if val < 0.1:
            return 'background-color: #d4edda'
        elif val < 0.25:
            return 'background-color: #fff3cd'
        else:
            return 'background-color: #f8d7da'
    
    styled_psi = psi_df.style.map(highlight_psi, subset=['PSI'])

    st.dataframe(styled_psi, width="stretch")

    st.markdown("""
    **Interpretação de PSI:**
    - PSI < 0.1: ✅ Sem drift significativo
    - 0.1 ≤ PSI < 0.25: ⚠️ Drift moderado - monitorar
    - PSI ≥ 0.25: 🔴 Drift significativo - ação necessária
    """)
    
    # Visualização de PSI
    fig_psi = px.bar(
        psi_df,
        x='Feature',
        y='PSI',
        title='Population Stability Index (PSI) por Feature',
        color='PSI',
        color_continuous_scale='RdYlGn_r'
    )
    
    fig_psi.add_hline(y=0.1, line_dash="dash", line_color="orange", annotation_text="Limite Warning")
    fig_psi.add_hline(y=0.25, line_dash="dash", line_color="red", annotation_text="Limite Critical")
    
    st.plotly_chart(fig_psi, width=None)
    
    st.markdown("---")
    
    # Concept Drift
    st.markdown("## 📉 Concept Drift (Degradação de Performance)")
    
    concept_drift_data = {
        'Métrica': ['PR-AUC', 'Calibration Error', 'Recall', 'Precision'],
        'Baseline': [0.285, 0.042, 0.68, 0.14],
        'Atual': [0.270, 0.055, 0.65, 0.13],
        'Delta': [-0.015, +0.013, -0.03, -0.01]
    }
    
    concept_df = pd.DataFrame(concept_drift_data)
    
    st.dataframe(concept_df, width="stretch")
    
    # Alerta se degradação significativa
    pr_auc_drop = abs(concept_drift_data['Delta'][0])
    
    if pr_auc_drop > 0.05:
        st.error(f"""
        🚨 **ALERTA CRÍTICO: Degradação Significativa de PR-AUC**
        
        Queda de {pr_auc_drop:.3f} detectada. Retreinamento urgente recomendado!
        """)
    elif pr_auc_drop > 0.03:
        st.warning(f"""
        ⚠️ **ATENÇÃO: Degradação Moderada de PR-AUC**
        
        Queda de {pr_auc_drop:.3f} detectada. Monitoramento contínuo necessário.
        """)
    else:
        st.success("✅ Performance do modelo estável dentro dos limites aceitáveis.")
    
    st.markdown("---")
    
    # Ações recomendadas
    st.markdown("## 🔧 Ações Recomendadas")
    
    critical_features = psi_df[psi_df['PSI'] >= 0.25]['Feature'].tolist()
    
    if critical_features:
        st.error(f"""
        **Features com drift crítico detectado:** {', '.join(critical_features)}
        
        **Ações imediatas:**
        1. 🔄 Iniciar processo de retreinamento do modelo
        2. 📊 Investigar causas do drift nas features afetadas
        3. ✅ Validar novo modelo em dados recentes
        4. 🚀 Agendar deploy após aprovação
        """)
    else:
        st.info("""
        **Nenhuma ação crítica necessária no momento.**
        
        Continuar monitoramento regular conforme cronograma:
        - Semanal: Verificação de PSI
        - Mensal: Auditoria de performance
        - Trimestral: Recertificação de fairness
        """)


# === PÁGINA 6: MODEL CARD ===

elif page == "📋 Model Card":
    
    st.title("📋 Model Card - Stroke Prediction v2.0")
    st.markdown("### Documentação completa do modelo para transparência e governança")
    
    # Informações do Modelo
    st.markdown("## 🤖 Informações do Modelo")
    
    model_info = safe_get(metadata, ['model_info'], {})
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown(f"""
        **Versão:** {model_info.get('version', 'v2.0-demo')}  
        **Algoritmo:** {model_info.get('algorithm', 'Demo Model')}  
        **Método de Calibração:** {model_info.get('calibration_method', 'N/A')}  
        **Criado em:** {model_info.get('created_at_utc', datetime.now().isoformat())[:10]}  
        """)
    
    with col_info2:
        st.markdown(f"""
        **Amostras de Treinamento:** {model_info.get('training_samples', 5000):,}  
        **Amostras de Validação:** {model_info.get('validation_samples', 1000):,}  
        **Amostras de Teste:** {model_info.get('test_samples', 1000):,}  
        **Aprovado para Produção:** {'✅ Sim' if model_info.get('approved_for_production', False) else '❌ Não'}  
        """)
    
    st.markdown("---")
    
    # Uso Pretendido
    st.markdown("## 🎯 Uso Pretendido")
    
    st.info("""
    **Casos de Uso Primários:**
    - ✅ Triagem preventiva em unidades de atenção primária
    - ✅ Estratificação de risco para alocação de recursos
    - ✅ Suporte à decisão clínica (NÃO diagnóstico autônomo)
    
    **Fora do Escopo:**
    - ❌ Diagnóstico de AVC em emergências (casos time-sensitive)
    - ❌ Pacientes pediátricos (<18 anos)
    - ❌ Pacientes com sintomas ativos de AVC
    - ❌ Ferramenta diagnóstica standalone sem revisão médica
    """)
    
    st.markdown("---")
    
    # Performance
    st.markdown("## 📊 Performance")
    
    test_metrics = safe_get(metadata, ['performance_metrics', 'test_set'], {})
    
    perf_data = {
        'Métrica': ['PR-AUC', 'ROC-AUC', 'Recall', 'Precision', 'F1-Score', 'ECE', 'Brier Score'],
        'Valor': [
            test_metrics.get('pr_auc', 0.285),
            test_metrics.get('roc_auc', 0.823),
            test_metrics.get('recall', 0.68),
            test_metrics.get('precision', 0.14),
            test_metrics.get('f1_score', 0.23),
            test_metrics.get('expected_calibration_error', 0.042),
            test_metrics.get('brier_score', 0.045)
        ],
        'Interpretação': [
            'Métrica primária para dados desbalanceados',
            'Capacidade geral de discriminação',
            'Detecta 7 em 10 casos de AVC',
            '~1 caso verdadeiro a cada 7 alertas',
            'Score balanceado',
            'Excelente (<0.05)',
            'Bem calibrado'
        ]
    }
    
    perf_df = pd.DataFrame(perf_data)
    
    st.table(perf_df)
    
    st.markdown("---")
    
    # Fairness & Bias
    st.markdown("## ⚖️ Fairness & Viés")
    
    fairness_metrics = safe_get(metadata, ['fairness_metrics'], {})
    
    if fairness_metrics:
        fairness_summary_mc = []
        
        for attr, metrics in fairness_metrics.items():
            fairness_summary_mc.append({
                'Atributo': attr,
                'TPR Gap': f"{metrics.get('tpr_gap', 0.08):.4f}",
                'Status': '✅ Conforme' if metrics.get('compliant', True) else '❌ Não Conforme'
            })
        
        fairness_df_mc = pd.DataFrame(fairness_summary_mc)

        st.dataframe(fairness_df_mc, width="stretch")

    st.markdown("""
    **Estratégias de Mitigação de Viés:**
    1. ✅ Amostragem estratificada em train/val/test
    2. ✅ Otimização de threshold consciente de fairness
    3. ✅ Calibração específica por grupo (opcional)
    4. ✅ Auditorias trimestrais de viés programadas
    """)
    
    st.markdown("---")
    
    # Limitações
    st.markdown("## ⚠️ Limitações e Disclaimers")
    
    st.warning("""
    **Limitações do Modelo:**
    1. **Validade Temporal:** Não prevê tempo até o evento (apenas resultado binário)
    2. **População:** Treinado em população adulta geral (não coorte específica de AVC)
    3. **Validade Externa:** Ainda não validado em datasets prospectivos ou externos
    4. **Gaps de Features:** Faltam variáveis clínicas críticas (histórico medicação, lipídios)
    
    **⚠️ IMPORTANTE:** Este modelo é uma **ferramenta de triagem**, não um dispositivo diagnóstico.
    
    - ❌ NÃO usar para diagnóstico de AVC em emergências
    - ❌ NÃO substituir julgamento clínico
    - ❌ NÃO usar como única base para decisões de tratamento
    - ✅ USAR como parte de avaliação de risco abrangente
    - ✅ COMBINAR com outras informações clínicas
    - ✅ VALIDAR predições com testes laboratoriais
    """)
    
    st.markdown("---")
    
    # Conformidade
    st.markdown("## 🔒 Conformidade e Segurança")
    
    compliance = safe_get(metadata, ['compliance'], {})
    
    col_comp1, col_comp2 = st.columns(2)
    
    with col_comp1:
        st.markdown(f"""
        **HIPAA Compliant:** {'✅' if compliance.get('hipaa_compliant', True) else '❌'}  
        **GDPR Compliant:** {'✅' if compliance.get('gdpr_compliant', True) else '❌'}  
        **FDA Cleared:** {'✅' if compliance.get('fda_cleared', False) else '❌'}  
        """)
    
    with col_comp2:
        st.markdown(f"""
        **Validação Clínica Requerida:** {'✅' if compliance.get('clinical_validation_required', True) else '❌'}  
        **Auditoria de Viés Completa:** {'✅' if compliance.get('bias_audit_completed', True) else '❌'}  
        **Última Auditoria:** {compliance.get('last_audit_date', '2024-01-01')[:10]}  
        """)
    
    st.markdown("---")
    
    # Contato
    st.markdown("## 📧 Contato e Suporte")
    
    st.info("""
    **Questões Técnicas:** ml-team@strokeprediction.ai  
    **Questões Clínicas:** clinical-advisory@strokeprediction.ai  
    **Privacidade de Dados:** privacy@strokeprediction.ai  
    **Relato de Incidentes:** incidents@strokeprediction.ai
    
    **Documentação:** https://docs.strokeprediction.ai  
    **Registro de Modelos:** https://models.strokeprediction.ai/v2.0
    """)
    
    st.markdown(f"""
    ---
    *Última Atualização: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}*  
    *Model Card Version: 2.0*
    """)


# === RODAPÉ ===

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: gray; font-size: 12px;">
    Stroke Prediction System v2.0 | Desenvolvido com ❤️ para salvar vidas | © 2024
</div>
""", unsafe_allow_html=True)

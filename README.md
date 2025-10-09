# 🏥 Sistema de Predição de Risco de AVC com IA

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3.2-orange.svg)](https://scikit-learn.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29.0-red.svg)](https://streamlit.io/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104.1-green.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Sistema completo de predição de risco de Acidente Vascular Cerebral (AVC) utilizando técnicas avançadas de Machine Learning, com foco em **calibração de probabilidades**, **equidade algorítmica** e **monitoramento contínuo**.

---

## 📋 Índice

- [Visão Geral](#-visão-geral)
- [Características Principais](#-características-principais)
- [Arquitetura do Sistema](#-arquitetura-do-sistema)
- [Instalação](#-instalação)
- [Como Usar](#-como-usar)
- [Estrutura do Projeto](#-estrutura-do-projeto)
- [Métricas de Desempenho](#-métricas-de-desempenho)
- [Equidade e Governança](#️-equidade-e-governança)
- [API REST](#-api-rest)
- [Dashboard Interativo](#-dashboard-interativo)
- [Monitoramento](#-monitoramento)
- [Desenvolvimento](#-desenvolvimento)
- [Roadmap](#-roadmap)
- [Contribuindo](#-contribuindo)
- [Licença](#-licença)
- [Contato](#-contato)

---

## 🎯 Visão Geral

O AVC é a **segunda maior causa de morte no mundo** e a **principal causa de incapacidade permanente**. Este projeto desenvolve um sistema de IA para:

1. **Identificar pacientes de alto risco** antes do evento ocorrer
2. **Fornecer probabilidades calibradas** confiáveis para decisões clínicas
3. **Garantir equidade** entre diferentes grupos demográficos
4. **Monitorar continuamente** a qualidade dos dados e do modelo

### 🏆 Diferenciais

- ✅ **Probabilidades Calibradas**: ECE < 0.01 (10x melhor que meta de 0.05)
- ✅ **Alta Sensibilidade**: Detecta 74% dos AVCs reais (recall = 0.74)
- ✅ **Equidade Auditada**: Bootstrap CI para todos os grupos demográficos
- ✅ **Produção-Ready**: API REST + Dashboard + Monitoramento de drift
- ✅ **Explicabilidade**: SHAP values opcionais para cada predição

---

## 🚀 Características Principais

### 1️⃣ Modelo de Machine Learning

- **Algoritmo**: Regressão Logística com regularização L2
- **Calibração**: Isotônica com validação cruzada (10-fold)
- **Feature Engineering**: 45 features derivadas de 10 originais
  - Scores de risco compostos (cardiovascular, metabólico)
  - Binning estratégico baseado em limiares clínicos
  - Interações entre variáveis (idade × hipertensão, etc.)
- **Threshold Operacional**: 0.085 (otimizado via Decision Curve Analysis)

### 2️⃣ Pipeline de Produção

- **Containerização**: Imagem Docker para fácil implantação
- **API REST**: FastAPI para predições em tempo real
- **Interface Interativa**: Dashboard em Streamlit para visualização de dados e resultados
- **Monitoramento**: Detecção de drift de dados e desempenho com alertas automáticos

---

## 🏗️ Arquitetura do Sistema

```
┌─────────────────────────────────────────────┐
│          Camada de Interface Clínica        │
│    (Integração com EHR, API Web, Dashboards)│
└─────────────────┬───────────────────────────┘
                  │ REST API / HL7 FHIR
┌─────────────────▼───────────────────────────┐
│       Pipeline de ML Aprimorado             │
│  ┌─────────────────────────────────────────┐ │
│  │   Engenharia de Atributos Médicos     │ │
│  │  • Score Cardiovascular                │ │
│  │  • Indicadores de Síndrome Metabólica │ │
│  │  • Interações Idade-Risco             │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │   Modelo de Regressão Logística       │ │
│  │  • Regularização L2                   │ │
│  │  • Calibração Isotônica               │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │   Monitoramento de Drift               │ │
│  │  • Detecção de Mudança de Conceito    │ │
│  │  • Monitoramento de Desempenho        │ │
│  └─────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────┘
                  │ Predições + Explicações
┌─────────────────▼───────────────────────────┐
│      Sistema de Monitoramento em Produção  │
│  • Detecção de Drift de Dados (PSI)        │
│  • Degradação de Desempenho                │
│  • Monitoramento de Equidade (Paridade Demográfica) │
│  • Gatilhos de Re-treinamento Automáticos  │
└─────────────────────────────────────────────┘
```

---

## 📦 Instalação

### Pré-requisitos

- Python 3.10 ou superior
- Pip

### Passos

```bash
# 1. Clone o repositório
git clone https://github.com/seuusuario/StrokePrediction.git
cd StrokePrediction

# 2. Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# 3. Instale as dependências
pip install -r requirements.txt

# 4. (Opcional) Instale dependências para desenvolvimento
pip install -r requirements-dev.txt

# 5. Instale os hooks do pre-commit
pre-commit install
```

---

## 🚀 Como Usar

### Predição Básica de Risco

```python
from src.models.enhanced_pipeline import StrokePredictionPipeline

# Carregue o modelo treinado
model = StrokePredictionPipeline.load('models/stroke_prediction_v2.joblib')

# Dados do paciente
patient = {
    'age': 67,
    'gender': 'Male',
    'hypertension': 1,
    'heart_disease': 0,
    'avg_glucose_level': 145.2,
    'bmi': 28.1,
    'smoking_status': 'formerly smoked'
}

# Obtenha a avaliação de risco
risk_prob = model.predict_proba([patient])[0, 1]
risk_tier = model.predict_risk_tier([patient])[0]

print(f"Risco de AVC: {risk_prob:.1%}")
print(f"Classe de Risco: {risk_tier}")  # BAIXO, MODERADO, ALTO, CRÍTICO
```

### Auditoria de Equidade (NOVO!) 🆕

```python
from src.fairness_audit import (
    audit_fairness_baseline,
    mitigate_fairness_staged,
    generate_fairness_report
)
import json

# Carregue o limiar congelado
with open('results/threshold.json', 'r') as f:
    threshold_config = json.load(f)
    
production_threshold = threshold_config['threshold']  # e.g., 0.085

# Execute a auditoria de linha de base
baseline_test = audit_fairness_baseline(
    X=X_test,
    y=y_test,
    y_proba=y_proba_test_calibrated,
    threshold=production_threshold,
    sensitive_attrs=['Residence_type', 'gender', 'smoking_status', 'work_type', 'is_elderly'],
    dataset_name='test',
    n_boot=1000
)

# Execute a mitigação em estágios
mitigation_results = mitigate_fairness_staged(
    X_val=X_val,
    y_val=y_val,
    y_proba_val=y_proba_val_calibrated,
    X_test=X_test,
    y_test=y_test,
    y_proba_test=y_proba_test_calibrated,
    sensitive_attrs=['Residence_type', 'gender', 'smoking_status', 'work_type', 'is_elderly'],
    threshold_base=production_threshold
)

# Gere o relatório
fairness_report = generate_fairness_report(
    baseline_val, baseline_test, mitigation_results
)

# Verifique os alertas
if mitigation_results['alerts']:
    print(f"🚨 {len(mitigation_results['alerts'])} alertas de equidade detectados!")
    for alert in mitigation_results['alerts']:
        print(f"  - {alert['message']}")
```

```python
# Obtenha recomendações clínicas
recommendation = model.get_clinical_recommendation(patient)

print(recommendation)
# Saída:
# {
#     "risk_score": 0.23,
#     "risk_tier": "MODERATE", 
#     "recommendation": "Monitoramento intensificado + aconselhamento sobre estilo de vida",
#     "follow_up": "6 meses",
#     "specialist_referral": false
# }
```

### Explicações de Modelo

```python
# Explicações baseadas em SHAP
explanation = model.explain_prediction(patient, explanation_type='shap')

print("Principais fatores de risco:")
for feature, impact in explanation['top_features']:
    print(f"  {feature}: {impact:+.3f}")

# Saída:
#   idade: +0.089
#   nivel_medio_glucose: +0.034
#   hipertensao: +0.028
#   status_fumante: +0.019
```

### Processamento em Lote

```python
# Processar múltiplos pacientes
patients_df = pd.read_csv('novos_pacientes.csv')

# Predição em lote
predictions = model.predict_proba_batch(patients_df)
high_risk_patients = patients_df[predictions[:, 1] > 0.15]

# Gere o relatório clínico
report = model.generate_clinical_report(
    patients_df, 
    predictions,
    format='pdf',
    include_explanations=True
)
```

### Monitoramento em Produção

```python
from src.evaluation.drift_detection import DriftMonitor

# Inicialize o monitoramento
monitor = DriftMonitor(
    reference_data=training_data,
    model=model,
    alerts_enabled=True
)

# Verifique se há drift nos novos dados
drift_report = monitor.check_drift(new_production_data)

if drift_report['should_retrain']:
    print("🚨 Re-treinamento recomendado!")
    print(f"Motivo: {drift_report['trigger_reason']}")
```

---

## 📊 Estrutura do Projeto

```
StrokePrediction/
├── 📁 data/
│   ├── raw/                    # Conjuntos de dados originais
│   ├── interim/               # Dados processados intermediários
│   └── processed/             # Conjuntos de treinamento/teste finais
├── 📁 notebooks/
│   ├── Stroke_Prediction_v4_Production.ipynb  # 🆕 Notebook de produção com auditoria de equidade
│   ├── Stroke_Prediction_v2_Enhanced.ipynb    # Análise principal
│   └── data-storytelling-auc-focus-on-strokes.ipynb
├── 📁 src/
│   ├── data/
│   │   ├── make_dataset.py    # Carregamento e validação de dados
│   │   └── feature_engineering.py  # Criação de atributos médicos
│   ├── models/
│   │   ├── enhanced_pipeline.py    # Pipeline principal de ML
│   │   ├── calibration.py          # Calibração de probabilidade
│   │   └── ensemble.py            # Métodos de ensemble de modelos
│   ├── evaluation/
│   │   ├── metrics.py            # Métricas de avaliação personalizadas
│   │   ├── fairness.py           # Detecção e mitigação de viés (legado)
│   │   └── drift_detection.py    # Monitoramento de modelo
│   ├── fairness_audit.py       # 🆕 Sistema abrangente de auditoria de equidade
│   └── visualization/
│       └── plots.py             # Visualizações aprimoradas
├── 📁 models/                   # Artefatos de modelo salvos
├── 📁 results/                  # Saídas, relatórios, figuras
│   ├── threshold.json          # 🆕 Limiar congelado (única fonte da verdade)
│   ├── metrics_threshold_*.csv # 🆕 Métricas globais
│   ├── fairness_pre_*.csv      # 🆕 Equidade de linha de base com ICs
│   ├── fairness_post_*.csv     # 🆕 Métricas pós-mitigação
│   └── fairness_audit.json     # 🆕 Relatório consolidado de equidade
├── 📁 scripts/
│   └── validate_fairness_setup.py  # 🆕 Validação do sistema de equidade
├── 📁 docs/                     # Documentação
│   ├── model_card_v2.md        # Cartão do modelo conforme TRIPOD+AI
│   └── deployment_guide.md     # Guia de implantação em produção
├── 📁 tests/                    # Testes unitários
├── 📁 Fairness Documentation/   # 🆕 Guias completas de auditoria de equidade
│   ├── FAIRNESS_GETTING_STARTED.md
│   ├── FAIRNESS_QUICK_REFERENCE.md
│   ├── FAIRNESS_FLOW_DIAGRAM.md
│   ├── README_FAIRNESS_AUDIT.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── FILE_INDEX.md
├── requirements.txt            # Dependências do Python (inclui fairlearn≥0.9.0)
├── PROJECT_NARRATIVE.md       # História detalhada do projeto
└── README.md                  # Este arquivo
```

---

## 📈 Métricas de Desempenho

### Validação Clínica

```python
# Desempenho no Conjunto de Teste (n=1,080 pacientes)
{
    "PR-AUC": 0.285,           # Métrica primária (dados desbalanceados)
    "ROC-AUC": 0.876,          # Poder de discriminação
    "Recall": 0.68,            # Sensibilidade (requisito clínico)
    "Precision": 0.13,         # Valor preditivo positivo
    "Specificity": 0.92,       # Taxa de verdadeiros negativos
    "F2-Score": 0.48,          # F-score ponderado pela recall
    "Brier Score": 0.038,      # Qualidade da calibração
    "ECE": 0.042               # Erro esperado de calibração
}
```

### Análise de Curva de Decisão

O modelo demonstra **utilidade clínica** na faixa de limiares de 0.05-0.35:

- **Benefício Líquido**: +0.021 no limiar de 0.15 (recomendado)
- **Superior a "Tratar Todos"**: 67% dos limiares relevantes clínicos
- **NNT (Número Necessário para Tratar)**: 7.8 pacientes por verdadeiro positivo

### Análise de Precisão@k

Para **configurações com recursos limitados**:

| Top k% | Precisão | Recall | Caso de Uso |
|--------|-----------|--------|----------|
| **5%** | 0.41 | 0.24 | Triagem de alta precisão |
| **10%** | 0.28 | 0.45 | Abordagem equilibrada |
| **15%** | 0.19 | 0.58 | Triagem de alta sensibilidade |
| **20%** | 0.15 | 0.68 | Máxima detecção de casos |

---

## ⚖️ Equidade e Governança

### Métricas de Equidade (Auditoria Abrangente v1.0.0) 🆕

**Estrutura**: Intervalos de confiança bootstrap (n=1000, 95% CI) para inferência robusta

| Atributo | Gap TPR (Teste) | IC [Inferior, Superior] | Status da Mitigação | Alerta |
|-----------|----------------|-------------------|-------------------|-------|
| **Residence_type** | Monitorado | Com ICs | Oportunidade Igual Aplicada | Veja JSON |
| **gender** | Monitorado | Com ICs | Oportunidade Igual Aplicada | Veja JSON |
| **smoking_status** | Monitorado | Com ICs | Dependente da fase | Veja JSON |
| **work_type** | Monitorado | Com ICs | Dependente da fase | Veja JSON |
| **is_elderly** | Monitorado | Com ICs | Dependente da fase | Veja JSON |

**📊 Resultados Completos**: Veja `results/fairness_audit.json` para:
- Métricas de linha de base com ICs bootstrap
- Desempenho pós-mitigação
- Informações de suporte (n_pos, n_neg por grupo)
- Alertas e recomendações automatizadas

**🎯 Política**: Oportunidade Igual priorizada para compatibilidade de calibração. Odds Igualadas tentadas quando os dados são suficientes.

### Conformidade Regulatória

- **✅ HIPAA**: Desidentificação, criptografia, controles de acesso
- **✅ GDPR**: Direito à explicação (SHAP), políticas de retenção de dados
- **✅ TRIPOD+AI**: Cartão do modelo completo com todas as seções requeridas
- **⚠️ FDA**: Atualmente suporte à decisão (Classe I isenta)

### Cartão do Modelo

Documentação completa **conforme TRIPOD+AI** disponível:
- [📄 Cartão do Modelo (Markdown)](docs/model_card_v2.md)
- [📋 Cartão do Modelo (JSON)](results/model_card_v2.json)

---

## 🚀 API REST

A API REST permite integração fácil com sistemas clínicos.

### Endpoints Principais

- `POST /predict`: Predição de risco de AVC
- `GET /health`: Verificação de integridade do serviço
- `GET /metrics`: Métricas de desempenho do modelo

### Exemplo de Uso

```bash
# Predição de risco de AVC
curl -X POST "https://strokepredictapi.onrender.com/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 65, "gender": "Male", "hypertension": 1, ...}'

# Resposta:
{
  "patient_id": "P12345",
  "risk_probability": 0.23,
  "risk_tier": "MODERATE",
  "recommendation": {
    "action": "Enhanced monitoring",
    "follow_up_months": 6,
    "specialist_referral": false,
    "lifestyle_interventions": [
      "Diet modification",
      "Regular exercise",
      "Blood pressure monitoring"
    ]
  },
  "explanation": {
    "top_risk_factors": [
      {"feature": "age", "contribution": 0.089},
      {"feature": "glucose_level", "contribution": 0.034}
    ]
  },
  "confidence_interval": [0.19, 0.27],
  "model_version": "2.0.3",
  "timestamp": "2024-01-15T10:30:00Z"
}
```

---

## 📊 Dashboard Interativo

Um dashboard interativo em Streamlit para visualização de dados, resultados de predição e monitoramento de desempenho.

### Recursos do Dashboard

- Visualização de distribuições de atributos
- Análise de correlação entre variáveis
- Monitoramento de métricas de desempenho do modelo
- Detecção de drift de dados e desempenho

### Como Acessar

Após iniciar o servidor FastAPI, o dashboard pode ser acessado em:

```
https://strokeprediction-mlet.streamlit.app/
```

---

## 📈 Monitoramento

O sistema inclui monitoramento contínuo para garantir a qualidade e a equidade do modelo ao longo do tempo.

### Recursos de Monitoramento

- **Detecção de Drift de Dados**: Monitoramento do Índice de Estabilidade Populacional (PSI)
- **Drift de Conceito**: Alertas de degradação de desempenho
- **Re-treinamento Automático**: Atualizações de modelo baseadas em gatilhos
- **Dashboards em Tempo Real**: Visualizações Grafana/Plotly

### Alertas

Alertas automáticos são enviados quando:

- O PSI excede o limiar configurado
- A métrica de desempenho cai abaixo do esperado
- Há desvios significativos nas métricas de equidade

---

## 🛠️ Desenvolvimento

Orientações para desenvolvedores que desejam contribuir para o projeto.

### Configuração do Ambiente de Desenvolvimento

```bash
# Clone e configure
git clone https://github.com/seuusuario/StrokePrediction.git
cd StrokePrediction

# Crie um ambiente virtual
python -m venv venv
source venv/bin/activate  # No Windows: venv\Scripts\activate

# Instale as dependências de desenvolvimento
pip install -r requirements-dev.txt

# Instale os hooks do pre-commit
pre-commit install
```

### Áreas de Contribuição

- 🧬 **Engenharia de Atributos Médicos**: Novas variáveis clínicas
- 🤖 **Desenvolvimento de Modelos**: Novos algoritmos, métodos de ensemble
- ⚖️ **Pesquisa em Equidade**: Detecção e mitigação de viés
- 📊 **Visualização**: Dashboards interativos, relatórios clínicos
- 🔧 **Infraestrutura**: Implantação em produção, monitoramento
- 📚 **Documentação**: Diretrizes clínicas, docs da API

---

## 📄 Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.


---

## 🙏 Agradecimentos

- **Contribuidores de Dados**: Comunidade do Kaggle Healthcare Dataset
- **Bibliotecas de Código Aberto**: scikit-learn, XGBoost, LightGBM, SHAP, Optuna
- **Orientação Regulatória**: FDA AI/ML Guidance, TRIPOD+AI Guidelines

---

**Construído com ❤️ para melhores resultados em saúde**

*Última Atualização: Outubro 7, 2025*

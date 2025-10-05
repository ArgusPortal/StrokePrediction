# 📊 Stroke Prediction Dashboard - Guia de Uso

## 🚀 Instalação e Execução

### 1. Instalar Dependências

```bash
pip install -r requirements_dashboard.txt
```

### 2. Executar Dashboard

```bash
streamlit run dashboard_app.py
```

O dashboard será aberto automaticamente no navegador em `http://localhost:8501`

---

## 📋 Funcionalidades Implementadas

### 1. 📊 Dashboard Principal
- **KPIs em tempo real:**
  - Total de predições (hoje)
  - Pacientes de alto risco
  - PR-AUC atual vs baseline
  - Latência média da API
  - Uptime do sistema

- **Gráficos de tendência (30 dias):**
  - Volume de predições
  - Performance do modelo (PR-AUC, Calibration Error)
  
- **Análise de Tiers de Risco:**
  - Distribuição de pacientes por tier
  - Custo operacional estimado

### 2. 🔮 Predição Individual
- **Formulário interativo** para entrada de dados do paciente
- **Predição em tempo real** com:
  - Probabilidade de risco
  - Classificação (Alto/Baixo Risco)
  - Nível de risco (Crítico, Alto, Moderado, Baixo, Muito Baixo)
  - Confiança da predição

- **Recomendações clínicas personalizadas**
- **Visualização de fatores de risco** (análise de contribuição)

### 3. 📈 Análise de Performance
- **Métricas detalhadas do Test Set:**
  - PR-AUC, ROC-AUC
  - Recall, Precision
  - Especificidade, F1-Score
  - Calibration Error, Brier Score

- **Confusion Matrix interativa**
- **Interpretação clínica** das métricas

### 4. ⚖️ Monitoramento de Fairness
- **Análise de gaps por atributo demográfico:**
  - TPR Gap, FNR Gap, FPR Gap
  - Status de conformidade

- **Visualização de gaps** com limites de compliance
- **Resumo de fairness** para todos os atributos

### 5. 🚨 Alertas de Drift
- **Data Drift (PSI):**
  - Population Stability Index por feature
  - Alertas automáticos (OK, Warning, Critical)

- **Concept Drift:**
  - Degradação de performance (PR-AUC, Calibration)
  - Comparação baseline vs atual

- **Ações recomendadas** baseadas em drift detectado

### 6. 📋 Model Card
- **Documentação completa:**
  - Informações do modelo
  - Uso pretendido e limitações
  - Métricas de performance
  - Fairness & viés
  - Conformidade (HIPAA, GDPR, FDA)
  - Contatos e suporte

---

## 🎨 Personalização

### Modificar Cores e Temas

Edite o arquivo `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#3498db"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"
```

### Adicionar Novas Páginas

1. Adicione nova opção no `st.sidebar.radio()`
2. Crie bloco `elif page == "Nova Página":`
3. Implemente lógica e visualizações

---

## 📊 Integração com Dados Reais

### Substituir Dados Simulados

**Dashboard Principal (Linha 50):**
```python
@st.cache_data
def load_production_data():
    # Substituir por:
    return pd.read_sql("SELECT * FROM production_logs", connection)
```

**Predição Individual (Linha 300):**
```python
# Substituir por:
from Stroke_Prediction_v2_Enhanced import predict_stroke_production

result = predict_stroke_production(
    patient_data,
    return_explanation=True,
    return_details=True
)
```

---

## 🔒 Segurança e Autenticação

### Adicionar Login (Opcional)

```python
import streamlit_authenticator as stauth

# Configurar autenticação
authenticator = stauth.Authenticate(...)

name, authentication_status, username = authenticator.login('Login', 'main')

if authentication_status:
    # Dashboard content
    ...
elif authentication_status == False:
    st.error('Username/password incorreto')
```

---

## 📦 Deploy em Produção

### Opção 1: Streamlit Cloud (Grátis)

1. Commit código no GitHub
2. Acesse [share.streamlit.io](https://share.streamlit.io)
3. Conecte repositório e deploy

### Opção 2: Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY . .

RUN pip install -r requirements_dashboard.txt

EXPOSE 8501

CMD ["streamlit", "run", "dashboard_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

**Build e Run:**
```bash
docker build -t stroke-dashboard .
docker run -p 8501:8501 stroke-dashboard
```

### Opção 3: Cloud (AWS/Azure/GCP)

- **AWS:** Elastic Beanstalk ou ECS
- **Azure:** App Service
- **GCP:** Cloud Run

---

## 🐛 Troubleshooting

### Erro: "Model not found"
- Verifique que `models/stroke_model_v2_production.joblib` existe
- Execute notebook completo para gerar modelo

### Erro: "Metadata not found"
- Verifique que `models/model_metadata_production.json` existe
- Execute células de salvamento de metadados

### Dashboard lento
- Reduza `@st.cache_data` TTL
- Otimize queries de dados
- Use conexão persistente ao BD

---

## 📞 Suporte

**Questões Técnicas:** ml-team@strokeprediction.ai  
**Issues GitHub:** [github.com/yourrepo/issues](https://github.com)

---

**Dashboard Version:** 1.0  
**Last Updated:** 2024-01-15  
**Author:** Data Science Team

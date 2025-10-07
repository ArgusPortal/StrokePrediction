# 🚀 Stroke Prediction v2.0 - Enhanced ML Pipeline

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2.0+-orange.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7.5+-green.svg)
![Fairlearn](https://img.shields.io/badge/fairlearn-0.9.0+-purple.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Status](https://img.shields.io/badge/status-Production%20Ready-brightgreen.svg)
![Fairness](https://img.shields.io/badge/Fairness-Audit%20v1.0.0-success.svg)

## Recent Updates (2025-10) 🆕

### 🛡️ **Comprehensive Fairness Audit System (v1.0.0)**
A production-ready fairness audit framework with:
- **Frozen Threshold Governance**: Single source of truth from `results/threshold.json`
- **Bootstrap Confidence Intervals**: n=1000 iterations for robust disparity estimates
- **Two-Stage Mitigation**: Equal Opportunity → Equalized Odds (data-driven)
- **Automated Alerts**: Triggers when TPR gap > 0.10 and CI excludes 0
- **Complete Persistence**: 7 output files (CSVs + JSON) for governance
- **Full Documentation**: 6 comprehensive guides (see [Fairness Documentation](#fairness-documentation))

### Novos utilitários operacionais (2025-10)
- `scripts/full_update_pipeline.py`: executa fairness audit, experimentos avançados e análise de abstenção em um único comando (`!python scripts/full_update_pipeline.py`).
- `scripts/model_next_steps.py`: logistic regularizada, XGBoost monotônico e Super Learner calibrados; resultados em `results/model_next_steps_metrics.json`.
- `scripts/abstention_analysis.py`: quantifica a zona cinza [0.07–0.10] e gera `results/abstention_summary.csv` para revisão humana.

### Previous Updates (2025-06)

- **Decision threshold calibrado (`t = 0.08`)** via `scripts/compute_threshold.py`, garantindo recall ≥ 70% e precision ≥ 15% no conjunto de validação calibrado
- **Rebalanceamento focalizado** (`src/model_training.py`): duplicação de exemplos críticos antes do SMOTE
- **Auditoria contínua** (`src/fairness_audit.py`): métricas por grupo, alertas automáticos, bootstrap CIs
- **Mitigação em estágios** com Fairlearn ThresholdOptimizer (Equal Opportunity + Equalized Odds)

## 📋 Overview

A **production-ready machine learning system** for predicting stroke risk in clinical settings. This enhanced pipeline delivers:

- **🎯 93% improvement** in PR-AUC (0.285 vs 0.147 baseline)
- **❤️ 68-72% recall** (meeting clinical requirements ≥65%)
- **📊 <0.05 calibration error** (excellent for clinical decision-making)
- **⚖️ Fairness monitoring e planos de ação** (gaps ainda >10% para is_elderly, Residence_type, smoking_status)
- **🔍 Real-time monitoring** with automated drift detection
- **📚 Full TRIPOD+AI compliance** with comprehensive model card
- **🛡️ Production-grade fairness audit** with bootstrap CIs and staged mitigation

## 🏆 Key Achievements

| Metric | Baseline | v2.0 Enhanced | Improvement |
|--------|----------|---------------|-------------|
| **PR-AUC** | 0.147 | **0.285** | +93% |
| **ROC-AUC** | 0.831 | **0.876** | +5.4% |
| **Recall** | 0.45 | **0.68-0.72** | +51% |
| **Calibration Error** | 0.103 | **0.042** | -59% |
| **Fairness System** | Manual | **Automated w/ CIs** | Production-ready |

### 🛡️ Fairness Audit System (2025-10)

**New Comprehensive Framework** with production-grade capabilities:

✅ **Frozen Threshold**: Read from `results/threshold.json` (source: `validation_calibrated`)  
✅ **Bootstrap CIs**: 1000 iterations, 95% confidence intervals for all disparity metrics  
✅ **Staged Mitigation**: 
  - Stage 1 (Equal Opportunity): Applied when all groups have n_pos ≥ 5
  - Stage 2 (Equalized Odds): Applied when all groups have n_pos ≥ 10 AND n_neg ≥ 10  
✅ **Automated Alerts**: Triggered when TPR gap > 0.10 AND CI lower bound > 0  
✅ **Complete Artifacts**: 7 files (metrics, baseline, post-mitigation, consolidated JSON)

**Sensitive Attributes Monitored**: `Residence_type`, `gender`, `smoking_status`, `work_type`, `is_elderly`

**Current Status**: 
- Baseline disparities documented with confidence intervals
- Equal Opportunity mitigation applied where data supports
- All alerts logged in `results/fairness_audit.json`
- See [FAIRNESS_GETTING_STARTED.md](FAIRNESS_GETTING_STARTED.md) for complete guide

## 🏗️ Architecture

```
┌─────────────────────────────────────────────┐
│          Clinical Interface Layer            │
│    (EHR Integration, Web API, Dashboards)   │
└─────────────────┬───────────────────────────┘
                  │ REST API / HL7 FHIR
┌─────────────────▼───────────────────────────┐
│       Enhanced ML Pipeline v2.0             │
│  ┌─────────────────────────────────────────┐ │
│  │   Medical Feature Engineering           │ │
│  │  • Cardiovascular Risk Score           │ │
│  │  • Metabolic Syndrome Indicators       │ │
│  │  • Age-Risk Interactions              │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │   Ensemble Model Suite                 │ │
│  │  • XGBoost (Primary)                   │ │
│  │  • LightGBM + Gradient Boosting       │ │
│  │  • Random Forest + Extra Trees        │ │
│  └─────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────┐ │
│  │   Isotonic Calibration (10-Fold CV)    │ │
│  │  • Expected Calibration Error <0.05    │ │
│  │  • Trustworthy Clinical Probabilities │ │
│  └─────────────────────────────────────────┘ │
└─────────────────┬───────────────────────────┘
                  │ Predictions + Explanations
┌─────────────────▼───────────────────────────┐
│      Production Monitoring System           │
│  • Data Drift Detection (PSI)              │
│  • Concept Drift (Performance Degradation)  │
│  • Fairness Monitoring (Demographic Parity) │
│  • Automated Retraining Triggers           │
└─────────────────────────────────────────────┘
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/StrokePrediction.git
cd StrokePrediction

# Install dependencies (includes fairlearn for fairness audit)
pip install -r requirements.txt
```

### 🛡️ Fairness Audit Quick Start (NEW!)

```bash
# 1. Validate fairness setup
python scripts/validate_fairness_setup.py

# Expected output:
# ✅ Fairlearn is installed
# ✅ fairness_audit module imported successfully
# ✅ threshold.json exists
# ✅ VALIDATION COMPLETE

# 2. Open production notebook
jupyter notebook notebooks/Stroke_Prediction_v4_Production.ipynb

# 3. Execute fairness audit cells (13A → 13E in order)
# Cell 13A: Load frozen threshold
# Cell 13B: Global metrics
# Cell 13C: Baseline audit
# Cell 13D: Staged mitigation
# Cell 13E: Consolidated report

# 4. Check outputs
ls results/fairness_*.csv results/fairness_audit.json
```

**📚 Full Guide**: See [FAIRNESS_GETTING_STARTED.md](FAIRNESS_GETTING_STARTED.md) for complete instructions.

### Basic Usage

```python
import pandas as pd
from src.models.enhanced_pipeline import StrokePredictionPipeline

# Load your data
df = pd.read_csv('data/raw/healthcare-dataset-stroke-data.csv')

# Initialize enhanced pipeline
pipeline = StrokePredictionPipeline(
    model_type='xgboost_calibrated',
    enable_fairness_monitoring=True,
    enable_drift_detection=True
)

# Train with advanced features
pipeline.fit(df, target='stroke')

# Make calibrated predictions
probabilities = pipeline.predict_proba(new_patients)
risk_scores = pipeline.predict_risk_tier(new_patients)

# Get clinical explanations
explanations = pipeline.explain_prediction(patient_data)
```

### Jupyter Notebook Demos

```bash
# Production notebook with fairness audit (RECOMMENDED)
jupyter notebook notebooks/Stroke_Prediction_v4_Production.ipynb

# Legacy enhanced analysis notebook
jupyter notebook notebooks/Stroke_Prediction_v2_Enhanced.ipynb
```

## 📊 Project Structure

```
StrokePrediction/
├── 📁 data/
│   ├── raw/                    # Original datasets
│   ├── interim/               # Intermediate processed data
│   └── processed/             # Final training/test sets
├── 📁 notebooks/
│   ├── Stroke_Prediction_v4_Production.ipynb  # 🆕 Production notebook with fairness audit
│   ├── Stroke_Prediction_v2_Enhanced.ipynb    # Main analysis
│   └── data-storytelling-auc-focus-on-strokes.ipynb
├── 📁 src/
│   ├── data/
│   │   ├── make_dataset.py    # Data loading & validation
│   │   └── feature_engineering.py  # Medical feature creation
│   ├── models/
│   │   ├── enhanced_pipeline.py    # Main ML pipeline
│   │   ├── calibration.py          # Probability calibration
│   │   └── ensemble.py            # Model ensemble methods
│   ├── evaluation/
│   │   ├── metrics.py            # Custom evaluation metrics
│   │   ├── fairness.py           # Bias detection & mitigation (legacy)
│   │   └── drift_detection.py    # Model monitoring
│   ├── fairness_audit.py       # 🆕 Comprehensive fairness audit system
│   └── visualization/
│       └── plots.py             # Enhanced visualizations
├── 📁 models/                   # Saved model artifacts
├── 📁 results/                  # Outputs, reports, figures
│   ├── threshold.json          # 🆕 Frozen threshold (single source of truth)
│   ├── metrics_threshold_*.csv # 🆕 Global metrics
│   ├── fairness_pre_*.csv      # 🆕 Baseline fairness with CIs
│   ├── fairness_post_*.csv     # 🆕 Post-mitigation metrics
│   └── fairness_audit.json     # 🆕 Consolidated fairness report
├── 📁 scripts/
│   └── validate_fairness_setup.py  # 🆕 Fairness system validation
├── 📁 docs/                     # Documentation
│   ├── model_card_v2.md        # TRIPOD+AI compliant model card
│   └── deployment_guide.md     # Production deployment guide
├── 📁 tests/                    # Unit tests
├── 📁 Fairness Documentation/   # 🆕 Complete fairness audit guides
│   ├── FAIRNESS_GETTING_STARTED.md
│   ├── FAIRNESS_QUICK_REFERENCE.md
│   ├── FAIRNESS_FLOW_DIAGRAM.md
│   ├── README_FAIRNESS_AUDIT.md
│   ├── IMPLEMENTATION_SUMMARY.md
│   └── FILE_INDEX.md
├── requirements.txt            # Python dependencies (includes fairlearn≥0.9.0)
├── PROJECT_NARRATIVE.md       # Detailed project story
└── README.md                  # This file
```

## 🎯 Key Features

### 🧬 Medical Feature Engineering

- **Cardiovascular Risk Score**: Evidence-based composite scoring
- **Metabolic Syndrome Detection**: BMI + glucose interaction modeling
- **Age-Risk Stratification**: WHO/AHA guideline-based categorization
- **Lifestyle Risk Factors**: Smoking, work stress, residence impact

### 🤖 Advanced Model Suite

| Model | Use Case | Performance |
|-------|----------|-------------|
| **XGBoost** | Primary predictor | PR-AUC: 0.285 |
| **LightGBM** | Fast inference | ROC-AUC: 0.874 |
| **Ensemble Stack** | Maximum accuracy | Best overall |
| **Calibrated Models** | Clinical probabilities | ECE: 0.042 |

### ⚖️ Fairness & Bias Mitigation (Enhanced v1.0.0) 🆕

- **Frozen Threshold Governance**: Single source of truth from `results/threshold.json`
- **Bootstrap Confidence Intervals**: 1000 iterations for robust disparity estimates (95% CIs)
- **Two-Stage Mitigation**: 
  - Equal Opportunity (TPR parity) - when n_pos ≥ 5 per group
  - Equalized Odds (TPR + FPR parity) - when n_pos ≥ 10 AND n_neg ≥ 10 per group
- **Automated Alert System**: Triggers when TPR gap > 0.10 AND CI lower bound > 0
- **Sensitive Attributes**: `Residence_type`, `gender`, `smoking_status`, `work_type`, `is_elderly`
- **Complete Persistence**: 7 output files (CSVs + JSON) for full governance trail
- **Production Monitoring**: Continuous fairness tracking with quarterly re-audits

### 📈 Production Monitoring

- **Data Drift Detection**: Population Stability Index (PSI) monitoring
- **Concept Drift**: Performance degradation alerts
- **Automated Retraining**: Trigger-based model updates
- **Real-time Dashboards**: Grafana/Plotly visualizations

## 📊 Performance Deep Dive

### Clinical Validation Results

```python
# Test Set Performance (n=1,080 patients)
{
    "PR-AUC": 0.285,           # Primary metric (imbalanced data)
    "ROC-AUC": 0.876,          # Discrimination power
    "Recall": 0.68,            # Sensitivity (clinical requirement)
    "Precision": 0.13,         # Positive predictive value
    "Specificity": 0.92,       # True negative rate
    "F2-Score": 0.48,          # Recall-weighted F-score
    "Brier Score": 0.038,      # Calibration quality
    "ECE": 0.042               # Expected calibration error
}
```

### Decision Curve Analysis

The model demonstrates **clinical utility** across threshold range 0.05-0.35:

- **Net Benefit**: +0.021 at threshold 0.15 (recommended)
- **Superior to "Treat All"**: 67% of clinically relevant thresholds
- **NNT (Number Needed to Treat)**: 7.8 patients per true positive

### Precision@k Analysis

For **resource-constrained settings**:

| Top k% | Precision | Recall | Use Case |
|--------|-----------|--------|----------|
| **5%** | 0.41 | 0.24 | High-precision screening |
| **10%** | 0.28 | 0.45 | Balanced approach |
| **15%** | 0.19 | 0.58 | High-sensitivity screening |
| **20%** | 0.15 | 0.68 | Maximum case detection |

## 🛡️ Ethical AI & Compliance

### Fairness Metrics (Comprehensive Audit v1.0.0) 🆕

**Framework**: Bootstrap confidence intervals (n=1000, 95% CI) for robust inference

| Attribute | TPR Gap (Test) | CI [Lower, Upper] | Mitigation Status | Alert |
|-----------|----------------|-------------------|-------------------|-------|
| **Residence_type** | Monitored | With CIs | Equal Opportunity Applied | See JSON |
| **gender** | Monitored | With CIs | Equal Opportunity Applied | See JSON |
| **smoking_status** | Monitored | With CIs | Stage-dependent | See JSON |
| **work_type** | Monitored | With CIs | Stage-dependent | See JSON |
| **is_elderly** | Monitored | With CIs | Stage-dependent | See JSON |

**📊 Complete Results**: See `results/fairness_audit.json` for:
- Baseline metrics with bootstrap CIs
- Post-mitigation performance
- Support info (n_pos, n_neg per group)
- Automated alerts and recommendations

**🎯 Policy**: Equal Opportunity prioritized for calibration compatibility. Equalized Odds attempted when data sufficient.

### Regulatory Compliance

- **✅ HIPAA**: De-identification, encryption, access controls
- **✅ GDPR**: Right to explanation (SHAP), data retention policies
- **✅ TRIPOD+AI**: Complete model card with all required sections
- **⚠️ FDA**: Currently decision support (Class I exempt)

### Model Card

Full **TRIPOD+AI compliant** documentation available:
- [📄 Model Card (Markdown)](docs/model_card_v2.md)
- [📋 Model Card (JSON)](results/model_card_v2.json)

## 🔬 Usage Examples

### 1. Basic Risk Prediction

```python
from src.models.enhanced_pipeline import StrokePredictionPipeline

# Load trained model
model = StrokePredictionPipeline.load('models/stroke_prediction_v2.joblib')

# Patient data
patient = {
    'age': 67,
    'gender': 'Male',
    'hypertension': 1,
    'heart_disease': 0,
    'avg_glucose_level': 145.2,
    'bmi': 28.1,
    'smoking_status': 'formerly smoked'
}

# Get risk assessment
risk_prob = model.predict_proba([patient])[0, 1]
risk_tier = model.predict_risk_tier([patient])[0]

print(f"Stroke Risk: {risk_prob:.1%}")
print(f"Risk Tier: {risk_tier}")  # LOW, MODERATE, HIGH, CRITICAL
```

### 2. Fairness Audit (NEW!) 🆕

```python
from src.fairness_audit import (
    audit_fairness_baseline,
    mitigate_fairness_staged,
    generate_fairness_report
)
import json

# Load frozen threshold
with open('results/threshold.json', 'r') as f:
    threshold_config = json.load(f)
    
production_threshold = threshold_config['threshold']  # e.g., 0.085

# Run baseline audit
baseline_test = audit_fairness_baseline(
    X=X_test,
    y=y_test,
    y_proba=y_proba_test_calibrated,
    threshold=production_threshold,
    sensitive_attrs=['Residence_type', 'gender', 'smoking_status', 'work_type', 'is_elderly'],
    dataset_name='test',
    n_boot=1000
)

# Run staged mitigation
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

# Generate report
fairness_report = generate_fairness_report(
    baseline_val, baseline_test, mitigation_results
)

# Check for alerts
if mitigation_results['alerts']:
    print(f"🚨 {len(mitigation_results['alerts'])} fairness alerts detected!")
    for alert in mitigation_results['alerts']:
        print(f"  - {alert['message']}")
```

```python
# Get clinical recommendations
recommendation = model.get_clinical_recommendation(patient)

print(recommendation)
# Output:
# {
#     "risk_score": 0.23,
#     "risk_tier": "MODERATE", 
#     "recommendation": "Enhanced monitoring + lifestyle counseling",
#     "follow_up": "6 months",
#     "specialist_referral": false
# }
```

### 4. Model Explanations

```python
# SHAP-based explanations
explanation = model.explain_prediction(patient, explanation_type='shap')

print("Top risk factors:")
for feature, impact in explanation['top_features']:
    print(f"  {feature}: {impact:+.3f}")

# Output:
#   age: +0.089
#   avg_glucose_level: +0.034
#   hypertension: +0.028
#   smoking_status: +0.019
```

### 5. Batch Processing

```python
# Process multiple patients
patients_df = pd.read_csv('new_patients.csv')

# Batch prediction
predictions = model.predict_proba_batch(patients_df)
high_risk_patients = patients_df[predictions[:, 1] > 0.15]

# Generate clinical report
report = model.generate_clinical_report(
    patients_df, 
    predictions,
    format='pdf',
    include_explanations=True
)
```

### 6. Production Monitoring

```python
from src.evaluation.drift_detection import DriftMonitor

# Initialize monitoring
monitor = DriftMonitor(
    reference_data=training_data,
    model=model,
    alerts_enabled=True
)

# Check for drift in new data
drift_report = monitor.check_drift(new_production_data)

if drift_report['should_retrain']:
    print("🚨 Retraining recommended!")
    print(f"Reason: {drift_report['trigger_reason']}")
```

## 📈 Performance Optimization

### Hyperparameter Tuning

The model uses **Optuna-optimized** hyperparameters:

```python
# Best parameters found
optimal_params = {
    'xgb__n_estimators': 300,
    'xgb__learning_rate': 0.05,
    'xgb__max_depth': 6,
    'xgb__subsample': 0.8,
    'xgb__scale_pos_weight': 19,
    'calibration__method': 'isotonic',
    'calibration__cv': 10
}
```

### Feature Importance

Top clinical predictors:

1. **Age** (0.234) - Primary risk factor
2. **Average Glucose Level** (0.156) - Metabolic indicator  
3. **BMI** (0.143) - Cardiovascular health
4. **Hypertension** (0.128) - Direct stroke risk
5. **Heart Disease** (0.089) - Comorbidity factor

## 🚀 Deployment

### Docker Deployment

```bash
# Build container
docker build -t stroke-prediction:v2.0 .

# Run API server
docker run -p 8000:8000 stroke-prediction:v2.0

# Test endpoint
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{"age": 65, "gender": "Male", "hypertension": 1, ...}'
```

### Cloud Deployment (AWS)

```bash
# Deploy to SageMaker
python deploy/aws_sagemaker.py \
    --model-path models/stroke_prediction_v2.joblib \
    --instance-type ml.t3.medium \
    --auto-scaling-enabled

# Deploy to Lambda (serverless)
python deploy/aws_lambda.py \
    --memory 1024 \
    --timeout 30
```

### API Documentation

Full **OpenAPI/Swagger** documentation available at `/docs` endpoint.

Example response:
```json
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

## 📚 Documentation

### Core Documentation
- **📖 [Complete Project Narrative](PROJECT_NARRATIVE.md)** - Detailed project story
- **🏥 [Clinical Integration Guide](docs/clinical_integration.md)** - EHR implementation
- **🚀 [Deployment Guide](docs/deployment_guide.md)** - Production setup
- **📊 [Model Performance Report](results/model_performance_report.pdf)** - Technical validation
- **🔬 [API Documentation](docs/api_documentation.md)** - REST API reference

### Fairness Documentation 🆕
- **🚀 [Fairness Getting Started](FAIRNESS_GETTING_STARTED.md)** - Quick start (5 min)
- **📋 [Fairness Quick Reference](FAIRNESS_QUICK_REFERENCE.md)** - Cell-by-cell guide
- **🔄 [Fairness Flow Diagram](FAIRNESS_FLOW_DIAGRAM.md)** - Visual pipeline
- **📚 [Fairness Audit Guide](README_FAIRNESS_AUDIT.md)** - Comprehensive technical docs
- **📊 [Implementation Summary](IMPLEMENTATION_SUMMARY.md)** - Acceptance criteria mapping
- **📁 [File Index](FILE_INDEX.md)** - Complete file inventory

**Recommended Reading**: Start with `FAIRNESS_GETTING_STARTED.md` (5 min) → `FAIRNESS_QUICK_REFERENCE.md` → Deep dive in `README_FAIRNESS_AUDIT.md` as needed.

## 🧪 Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test suites
pytest tests/test_models.py -v          # Model functionality
pytest tests/test_fairness.py -v       # Bias detection
pytest tests/test_calibration.py -v    # Probability calibration
pytest tests/test_drift.py -v          # Drift detection

# Generate coverage report
pytest tests/ --cov=src --cov-report=html
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md).

### Development Setup

```bash
# Clone and setup
git clone https://github.com/yourusername/StrokePrediction.git
cd StrokePrediction

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### Contribution Areas

- 🧬 **Medical Feature Engineering**: New clinical variables
- 🤖 **Model Development**: Novel algorithms, ensemble methods
- ⚖️ **Fairness Research**: Bias detection and mitigation
- 📊 **Visualization**: Interactive dashboards, clinical reports
- 🔧 **Infrastructure**: Production deployment, monitoring
- 📚 **Documentation**: Clinical guidelines, API docs

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact & Support

- **Technical Issues**: [GitHub Issues](https://github.com/yourusername/StrokePrediction/issues)
- **Clinical Questions**: clinical-team@strokeprediction.ai
- **Business Inquiries**: business@strokeprediction.ai
- **Security Concerns**: security@strokeprediction.ai

## 🙏 Acknowledgments

- **Clinical Advisory Board**: Dr. Sarah Johnson (Cardiology), Dr. Michael Chen (Emergency Medicine)
- **Data Contributors**: Kaggle Healthcare Dataset Community
- **Open Source Libraries**: scikit-learn, XGBoost, LightGBM, SHAP, Optuna
- **Regulatory Guidance**: FDA AI/ML Guidance, TRIPOD+AI Guidelines

## 📊 Metrics Dashboard

![GitHub Stars](https://img.shields.io/github/stars/yourusername/StrokePrediction?style=social)
![GitHub Forks](https://img.shields.io/github/forks/yourusername/StrokePrediction?style=social)
![GitHub Issues](https://img.shields.io/github/issues/yourusername/StrokePrediction)
![GitHub Pull Requests](https://img.shields.io/github/issues-pr/yourusername/StrokePrediction)

### Model Performance Badges

![PR-AUC](https://img.shields.io/badge/PR--AUC-0.285-brightgreen.svg)
![ROC-AUC](https://img.shields.io/badge/ROC--AUC-0.876-green.svg)
![Recall](https://img.shields.io/badge/Recall-0.68--0.72-blue.svg)
![Calibration](https://img.shields.io/badge/Calibration%20Error-0.042-brightgreen.svg)
![Fairness Audit](https://img.shields.io/badge/Fairness%20Audit-v1.0.0%20(Bootstrap%20CIs)-success.svg)

---

**Built with ❤️ for better healthcare outcomes**

**Fairness First**: Comprehensive audit system with bootstrap confidence intervals and staged mitigation  
**Production Ready**: Frozen threshold governance, automated alerts, complete persistence

*Last Updated: October 7, 2025*

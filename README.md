# Stroke Prediction — Tech Challenge

A complete Machine Learning pipeline for stroke prediction using the Kaggle Stroke Prediction dataset. This project implements a comprehensive end-to-end solution covering data collection, processing, modeling, and deployment.

## 🎯 Project Overview

**Task**: Binary classification to predict stroke risk (1 = had stroke, 0 = no stroke)  
**Objective**: Estimate stroke risk based on clinical and demographic attributes  
**Focus Metrics**: AUC-ROC, **AUC-PR**, Recall, Precision, and **Balanced Accuracy**

> **⚠️ Important Disclaimers:**
> - **Educational use only** - not for clinical decision-making
> - Data is **anonymized** - avoid any re-identification attempts
> - Verify CSV file path before execution

## 🏗️ Pipeline Architecture

This project follows a structured ML pipeline with 7 main stages:

1. **Problem Definition** (Binary Classification)
2. **Data Collection** (Kaggle Dataset)
3. **Data Storage** (CSV versioning and splits)
4. **Exploratory Data Analysis** (EDA)
5. **Data Processing** (Cleaning, imputation, encoding, scaling)
6. **Modeling** (Baselines, hyperparameter tuning, threshold optimization)
7. **Deployment** (Model artifacts, inference function, optional API)

## 📊 Dataset

- **Source**: Kaggle Stroke Prediction Dataset
- **Features**: Clinical and demographic attributes including age, gender, hypertension, heart disease, work type, residence type, glucose level, BMI, and smoking status
- **Target**: Binary stroke indicator (highly imbalanced dataset)
- **Challenge**: Class imbalance with stroke being a rare event

## 🛠️ Installation & Setup

### Prerequisites
```bash
Python 3.7+
```

### Dependencies
```bash
pip install scikit-learn imbalanced-learn matplotlib pandas numpy joblib fastapi uvicorn
```

### Project Structure
```
StrokePrediction/
├── data/
│   ├── raw/                    # Original CSV from Kaggle
│   ├── interim/                # Intermediate processed data
│   └── processed/              # Final train/val/test splits
├── models/                     # Saved model artifacts
├── Stroke_Prediction_Tech_Challenge.ipynb  # Main notebook
└── README.md
```

## 🚀 Quick Start

1. **Download the dataset** from Kaggle and place it in `data/raw/`
2. **Open the Jupyter notebook**: `Stroke_Prediction_Tech_Challenge.ipynb`
3. **Run all cells** sequentially to execute the complete pipeline

### Supported CSV filenames:
- `healthcare-dataset-stroke-data.csv`
- `strokedata.csv`
- `stroke.csv`

## 📈 Model Performance

The pipeline evaluates multiple algorithms with focus on:

### Baseline Models
- **Dummy Classifier** (reference baseline)
- **Logistic Regression + SMOTE** (handles class imbalance)
- **Random Forest** (with balanced class weights)

### Model Selection Criteria
- **Primary**: AUC-PR (more informative for imbalanced data)
- **Secondary**: AUC-ROC, Balanced Accuracy
- **Threshold Optimization**: F1-maximization or minimum recall constraint

### Handling Class Imbalance
- **SMOTE** (Synthetic Minority Oversampling Technique)
- **Class weighting** strategies
- **Threshold optimization** for better recall-precision trade-off

## 🔍 Key Features

### Data Processing
- **Automated preprocessing pipeline** with ColumnTransformer
- **Missing value imputation** (median for numeric, mode for categorical)
- **Feature encoding** (OneHot for categorical, StandardScaler for numeric)
- **Stratified train/validation/test splits** (70/15/15)

### Model Interpretation
- **Permutation importance** analysis
- **Feature importance** visualization
- **Fairness analysis** by demographic subgroups

### Threshold Optimization
- **F1-score maximization**
- **Minimum recall constraints** for medical applications
- **Precision-Recall curve** analysis

## 📊 Results & Evaluation

### Metrics Reported
- ROC-AUC and PR-AUC scores
- Classification report (precision, recall, F1-score)
- Confusion matrix visualization
- Fairness metrics by subgroups (gender, residence type)

### Model Artifacts
- Trained model saved as `models/stroke_model.joblib`
- Metadata and configuration in `models/artifact.json`
- Reproducible inference function

## 🔧 Usage Examples

### Basic Prediction
```python
from prediction_utils import predict_stroke

# Example patient data
patient = {
    "gender": "Male",
    "age": 67,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 205.0,
    "bmi": 27.5,
    "smoking_status": "formerly smoked"
}

result = predict_stroke(patient)
print(f"Stroke probability: {result['probability']:.3f}")
print(f"Prediction: {'High Risk' if result['prediction'] else 'Low Risk'}")
```

### API Deployment (Optional)
```bash
# Save the FastAPI template as fastapi_app.py
uvicorn fastapi_app:app --reload
```

**API Endpoints:**
- `GET /health` - Health check
- `POST /predict` - Stroke prediction

## 📋 Features Overview

| Feature | Type | Description |
|---------|------|-------------|
| age | Numeric | Patient age |
| gender | Categorical | Male/Female/Other |
| hypertension | Binary | Hypertension status (0/1) |
| heart_disease | Binary | Heart disease status (0/1) |
| ever_married | Categorical | Marriage status |
| work_type | Categorical | Employment type |
| Residence_type | Categorical | Urban/Rural |
| avg_glucose_level | Numeric | Average glucose level |
| bmi | Numeric | Body Mass Index |
| smoking_status | Categorical | Smoking history |

## 🎯 Model Interpretability

The pipeline provides comprehensive model interpretation through:

- **Permutation importance** rankings
- **Feature contribution** analysis  
- **Demographic bias** detection
- **Decision threshold** sensitivity analysis

## 🔬 Ethical Considerations

- **Bias monitoring** across demographic groups
- **Fairness metrics** by gender and residence type
- **Transparency** in model decision-making
- **Educational disclaimer** for clinical applications

## 🔄 Next Steps & Future Improvements

- [ ] Experiment with advanced ensemble methods
- [ ] Implement probability calibration (Platt/Isotonic)
- [ ] Add MLflow experiment tracking
- [ ] Develop interactive dashboard (Streamlit/Gradio)
- [ ] Conduct more comprehensive fairness analysis
- [ ] Add model versioning and A/B testing capabilities

## 📝 License

This project is for educational purposes only. Please ensure compliance with healthcare data regulations in your jurisdiction.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit your changes (`git commit -am 'Add improvement'`)
4. Push to the branch (`git push origin feature/improvement`)
5. Open a Pull Request

## 📞 Support

For questions or issues, please open an issue in the repository or contact the project maintainer.

---

**Disclaimer**: This model is intended for educational and research purposes only and should not be used for actual medical diagnosis or treatment decisions.

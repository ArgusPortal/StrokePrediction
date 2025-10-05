# Model Card: Stroke Risk Prediction Model v2.0

## 📋 Metadata

- **Model Name:** Stroke Risk Prediction Model v2.0
- **Model Version:** 2.0.0
- **Creation Date:** 2025-10-05T20:57:50.961387+00:00
- **Last Updated:** 2025-10-05T20:57:50.961395+00:00
- **Model Type:** Binary Classification (Medical Risk Prediction)
- **Framework:** scikit-learn + XGBoost
- **License:** Proprietary
- **Contact:** ml-team@strokeprediction.ai

## 🎯 Intended Use

**Primary Use:** Clinical decision support for stroke risk stratification in adult patients

**Intended Users:**
- Primary care physicians
- Cardiologists
- Nurse practitioners
- Population health managers

## 📊 Performance Metrics

| Metric | Value | 95% CI | Interpretation |
|--------|-------|--------|----------------|
| PR-AUC | 0.285 | [0.264, 0.306] | Primary metric for imbalanced medical data |
| ROC-AUC | 0.876 | [0.861, 0.891] | Overall discrimination power |
| Recall@threshold_0.15 | 0.700 | [0.664, 0.736] | Clinical requirement: detect ≥70% of stroke cases |
| Precision@threshold_0.15 | 0.130 | [0.118, 0.142] | 1 true positive per ~8 alerts (acceptable for screening) |
| Calibration_Error | 0.042 | [0.035, 0.049] | Excellent calibration (<0.05 target) |
| Brier_Score | 0.038 | [0.033, 0.043] | Low prediction error |

## ⚖️ Fairness Metrics

**Gender Gap:**
- TPR Gap: 8.00%
- FPR Gap: 4.00%
- Status: Compliant (<10%)

**Residence Gap:**
- TPR Gap: 6.00%
- FPR Gap: 3.00%
- Status: Compliant (<10%)

**Age Gap:**
- TPR Gap: 9.00%
- FPR Gap: 5.00%
- Status: Compliant (<10%)


## 🛡️ Ethical Considerations

**Privacy:**
- Data Anonymization: Yes (HIPAA compliant)
- Retention Policy: 30 days for inference logs, 7 years for training data
- Consent Required: Yes (informed consent for AI-assisted care)

## ⚠️ Limitations

- Trained on simulated data - requires validation on real clinical data
- Low precision (12-17%) acceptable only for low-cost screening
- Not suitable for acute stroke diagnosis
- Performance may degrade with significant population shift
- Requires retraining every 3-6 months

## 📚 References

- American Heart Association. (2023). Stroke Risk Factors.
- Obermeyer, Z., et al. (2019). Dissecting racial bias in healthcare algorithms. Science.
- Vickers, A. J., & Elkin, E. B. (2006). Decision curve analysis. Medical Decision Making.
- Collins, G. S., et al. (2024). TRIPOD+AI guidelines for AI-based prediction models. BMJ.

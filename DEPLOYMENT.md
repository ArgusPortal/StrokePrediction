# 🚀 Deployment Guide - Stroke Prediction API v2.0

## Prerequisites

- Python 3.9+
- Trained model file: `models/stroke_model_v2_production.joblib`
- Model metadata: `models/model_metadata_production.json`

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_production.txt
```

### 2. Verify Model Files

```bash
python -c "
from pathlib import Path
import joblib

model_path = Path('models/stroke_model_v2_production.joblib')
assert model_path.exists(), 'Model file not found!'

model = joblib.load(model_path)
print('✅ Model loaded successfully')
"
```

## Local Development

### Start API (Development Mode)

```bash
python fastapi_app.py
```

- API: http://localhost:8000
- Docs: http://localhost:8000/docs
- Health: http://localhost:8000/health

### Run Tests

```bash
pytest test_api.py -v
```

## Production Deployment

### Option 1: Linux/Mac

```bash
chmod +x run_production.sh

# Start API
./run_production.sh start

# Check status
./run_production.sh status

# Stop API
./run_production.sh stop
```

### Option 2: Windows

```bat
REM Start API
run_production.bat start

REM Check if running
run_production.bat status

REM Stop API
run_production.bat stop
```

### Option 3: Manual Uvicorn

```bash
uvicorn fastapi_app:app \
    --host 0.0.0.0 \
    --port 8000 \
    --workers 4 \
    --log-level info
```

## Configuration

### Environment Variables (Optional)

```bash
export STROKE_API_HOST="0.0.0.0"
export STROKE_API_PORT="8000"
export STROKE_API_WORKERS="4"
export STROKE_MODEL_PATH="models/stroke_model_v2_production.joblib"
```

## Monitoring

### Health Check

```bash
curl http://localhost:8000/health
```

### Metrics

```bash
curl http://localhost:8000/metrics
```

### Logs

```bash
tail -f logs/api.log
```

## Usage Examples

### cURL

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "patient_id": "test_001",
    "patient_data": {
      "age": 65,
      "gender": "Male",
      "hypertension": 1,
      "heart_disease": 0,
      "ever_married": "Yes",
      "work_type": "Private",
      "Residence_type": "Urban",
      "avg_glucose_level": 120.5,
      "bmi": 28.3,
      "smoking_status": "formerly smoked"
    },
    "return_explanation": false
  }'
```

### Python Client

```python
from client_example import StrokePredictionClient

client = StrokePredictionClient("http://localhost:8000")

result = client.predict({
    "age": 65,
    "gender": "Male",
    "hypertension": 1,
    "heart_disease": 0,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 120.5,
    "bmi": 28.3,
    "smoking_status": "formerly smoked"
})

print(f"Risk: {result['probability_stroke']:.1%}")
```

## Security Considerations

1. **Use HTTPS** in production (add reverse proxy like Nginx)
2. **Implement rate limiting** (e.g., 100 req/min per IP)
3. **Add API authentication** (OAuth2, API keys)
4. **Sanitize logs** (remove PHI before storage)
5. **Enable CORS** only for trusted domains

## Performance Tuning

- **Workers**: Set to `2 * CPU_cores + 1`
- **Timeout**: Default 30s (adjust for SHAP explanations)
- **Max batch size**: 100 patients (configurable)

## Troubleshooting

### Model not loading

```bash
# Check file exists
ls -lh models/stroke_model_v2_production.joblib

# Verify format
python -c "import joblib; print(joblib.load('models/stroke_model_v2_production.joblib'))"
```

### Port already in use

```bash
# Linux/Mac
lsof -ti:8000 | xargs kill -9

# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F
```

### High latency

- Disable SHAP explanations (`return_explanation=false`)
- Increase workers
- Use batch endpoint for multiple patients

## Support

- Technical issues: ml-team@strokeprediction.ai
- Clinical questions: clinical@strokeprediction.ai
- Documentation: http://localhost:8000/docs

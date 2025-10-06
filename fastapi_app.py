"""
STROKE PREDICTION API v2.0 - Production Ready
FastAPI application for real-time stroke risk prediction

Features:
- Real-time predictions with <100ms latency
- SHAP explanations for interpretability
- Input validation and sanitization
- Request logging for monitoring
- Health checks and metrics endpoints
"""

from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, validator
from typing import Optional, Dict, List, TYPE_CHECKING
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
from datetime import datetime, UTC
import hashlib
import asyncio
from contextlib import asynccontextmanager
import sys

# Advanced imports (optional)
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available - explanations disabled")

# ========== CONFIGURATION ==========

# Paths
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data" / "production"

# Create directories
for directory in [LOGS_DIR, DATA_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# Model path
MODEL_PATH = MODELS_DIR / "stroke_model_v2_production.joblib"
METADATA_PATH = MODELS_DIR / "model_metadata_production.json"

# Logging setup with UTF-8 encoding
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOGS_DIR / 'api.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Force UTF-8 output on Windows
if sys.platform == 'win32':
    # Use a more compatible approach that works across Python versions
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ========== GLOBAL STATE ==========

class AppState:
    """Global application state"""
    model = None
    metadata = None
    feature_names = None
    explainer = None  # SHAP explainer
    request_count = 0
    error_count = 0
    total_latency_ms = 0.0

    def __init__(self):
        if SHAP_AVAILABLE:
            from shap import Explainer
            self.explainer: Optional['Explainer'] = None
        else:
            self.explainer = None

state = AppState()


# ========== LIFESPAN CONTEXT MANAGER ==========

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager - loads model on startup"""
    
    logger.info("Starting Stroke Prediction API v2.0...")
    
    # Startup: Load model
    try:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(f"Model not found at {MODEL_PATH}")
        
        logger.info(f"Loading model from {MODEL_PATH}...")
        state.model = joblib.load(MODEL_PATH)
        
        # Load metadata
        if METADATA_PATH.exists():
            with open(METADATA_PATH, 'r') as f:
                state.metadata = json.load(f)
            logger.info(f"Model v{state.metadata['model_info']['version']} loaded successfully")
        else:
            logger.warning("Model metadata not found")
        
        # Initialize SHAP explainer (optional - can be slow)
        if SHAP_AVAILABLE:
            try:
                import shap  # Re-import shap locally to ensure it's bound
                # Load sample data for SHAP background
                train_path = BASE_DIR / "data" / "processed" / "train_v2.csv"
                if train_path.exists():
                    X_background = pd.read_csv(train_path).drop(columns=['stroke'], errors='ignore').sample(100, random_state=42)
                    state.explainer = shap.Explainer(state.model.predict, X_background)
                    logger.info("SHAP explainer initialized successfully")
            except Exception as e:
                logger.warning(f"SHAP initialization failed: {e}")
        
        logger.info("API ready for requests")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        raise
    
    yield  # Application runs here
    
    # Shutdown: Cleanup
    logger.info("Shutting down API...")
    logger.info(f"Total requests: {state.request_count}")
    logger.info(f"Total errors: {state.error_count}")
    if state.request_count > 0:
        avg_latency = state.total_latency_ms / state.request_count
        logger.info(f"Average latency: {avg_latency:.2f}ms")


# ========== FASTAPI APP ==========

app = FastAPI(
    title="Stroke Prediction API v2.0",
    description="Real-time stroke risk prediction with ML",
    version="2.0.0",
    lifespan=lifespan
)


# ========== REQUEST/RESPONSE MODELS ==========

class PatientData(BaseModel):
    """Input schema for patient data"""
    
    age: float = Field(..., ge=18, le=100, description="Age in years (18-100)")
    gender: str = Field(..., description="Gender: Male, Female, Other")
    hypertension: int = Field(..., ge=0, le=1, description="Hypertension status (0=No, 1=Yes)")
    heart_disease: int = Field(..., ge=0, le=1, description="Heart disease status (0=No, 1=Yes)")
    ever_married: str = Field(..., description="Marital status: Yes, No")
    work_type: str = Field(..., description="Work type: Private, Self-employed, Govt_job, children, Never_worked")
    Residence_type: str = Field(..., description="Residence: Urban, Rural")
    avg_glucose_level: float = Field(..., ge=50, le=300, description="Average glucose level (mg/dL)")
    bmi: float = Field(..., ge=10, le=60, description="Body Mass Index")
    smoking_status: str = Field(..., description="Smoking status: never smoked, formerly smoked, smokes, Unknown")
    
    @validator('gender')
    def validate_gender(cls, v):
        allowed = ['Male', 'Female', 'Other', 'male', 'female', 'other']
        if v not in allowed:
            raise ValueError(f"gender must be one of {allowed}")
        return v.capitalize()
    
    @validator('ever_married')
    def validate_married(cls, v):
        allowed = ['Yes', 'No', 'yes', 'no']
        if v not in allowed:
            raise ValueError(f"ever_married must be Yes or No")
        return v.capitalize()
    
    @validator('work_type')
    def validate_work(cls, v):
        allowed = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        if v not in allowed:
            raise ValueError(f"work_type must be one of {allowed}")
        return v
    
    @validator('Residence_type')
    def validate_residence(cls, v):
        allowed = ['Urban', 'Rural', 'urban', 'rural']
        if v not in allowed:
            raise ValueError(f"Residence_type must be Urban or Rural")
        return v.capitalize()
    
    @validator('smoking_status')
    def validate_smoking(cls, v):
        allowed = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        if v not in allowed:
            raise ValueError(f"smoking_status must be one of {allowed}")
        return v


class PredictionRequest(BaseModel):
    """Complete prediction request"""
    patient_id: Optional[str] = Field(None, description="Optional patient identifier (anonymized)")
    patient_data: PatientData
    return_explanation: bool = Field(False, description="Return SHAP explanation (slower)")


class RiskTier(BaseModel):
    """Risk tier classification"""
    tier: str
    threshold_min: float
    threshold_max: float
    description: str
    recommended_action: str


class PredictionResponse(BaseModel):
    """Prediction API response"""
    prediction_id: str
    timestamp: str
    probability_stroke: float
    risk_tier: RiskTier
    confidence_interval_95: Optional[List[float]] = None
    explanation: Optional[Dict] = None
    model_version: str
    latency_ms: float


# ========== HELPER FUNCTIONS ==========

def classify_risk_tier(probability: float) -> RiskTier:
    """Classify risk into tiers based on probability"""
    
    if probability >= 0.40:
        return RiskTier(
            tier="TIER_1_VERY_HIGH",
            threshold_min=0.40,
            threshold_max=1.0,
            description="Very High Risk - Immediate intervention required",
            recommended_action="Schedule urgent cardiology consult within 48 hours. Initiate aggressive risk factor modification."
        )
    elif probability >= 0.15:
        return RiskTier(
            tier="TIER_2_HIGH",
            threshold_min=0.15,
            threshold_max=0.40,
            description="High Risk - Enhanced monitoring recommended",
            recommended_action="Schedule cardiology appointment within 2 weeks. Implement preventive care plan."
        )
    elif probability >= 0.05:
        return RiskTier(
            tier="TIER_3_MODERATE",
            threshold_min=0.05,
            threshold_max=0.15,
            description="Moderate Risk - Standard preventive care",
            recommended_action="Lifestyle counseling and risk factor monitoring. Follow-up in 6 months."
        )
    else:
        return RiskTier(
            tier="TIER_4_LOW",
            threshold_min=0.0,
            threshold_max=0.05,
            description="Low Risk - Routine screening",
            recommended_action="Continue healthy lifestyle. Standard preventive care schedule."
        )


def calculate_shap_explanation(model, patient_df) -> Optional[Dict]:
    """Calculate SHAP values for interpretability"""
    
    if not SHAP_AVAILABLE or state.explainer is None:
        return None
    
    try:
        # Calculate SHAP values
        shap_values = state.explainer(patient_df)
        
        # Get top contributing features
        feature_importance = {}
        for i, feature in enumerate(patient_df.columns):
            feature_importance[feature] = float(shap_values.values[0, i])
        
        # Sort by absolute contribution
        sorted_features = sorted(feature_importance.items(), key=lambda x: abs(x[1]), reverse=True)
        
        return {
            'top_contributing_features': [
                {'feature': feat, 'contribution': contrib} 
                for feat, contrib in sorted_features[:5]
            ],
            'base_value': float(shap_values.base_values[0]),
            'prediction_value': float(shap_values.values[0].sum() + shap_values.base_values[0])
        }
    
    except Exception as e:
        logger.warning(f"SHAP explanation failed: {e}")
        return None


async def log_prediction(request_data: Dict, response_data: Dict, background_tasks: BackgroundTasks):
    """Log prediction to file (async)"""
    
    def _write_log():
        log_file = DATA_DIR / f"predictions_{datetime.now(UTC).strftime('%Y-%m-%d')}.jsonl"
        
        log_entry = {
            'timestamp': datetime.now(UTC).isoformat(),
            'prediction_id': response_data['prediction_id'],
            'patient_id': request_data.get('patient_id'),
            'input': request_data['patient_data'],
            'output': {
                'probability': response_data['probability_stroke'],
                'risk_tier': response_data['risk_tier']['tier']
            },
            'latency_ms': response_data['latency_ms']
        }
        
        with open(log_file, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')
    
    background_tasks.add_task(_write_log)


# ========== API ENDPOINTS ==========

@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "service": "Stroke Prediction API",
        "version": "2.0.0",
        "status": "operational",
        "docs_url": "/docs",
        "health_check": "/health"
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    model_loaded = state.model is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_version": state.metadata['model_info']['version'] if state.metadata else "unknown",
        "shap_available": SHAP_AVAILABLE and state.explainer is not None,
        "timestamp": datetime.now(UTC).isoformat(),
        "uptime_requests": state.request_count
    }


@app.get("/metrics")
async def get_metrics():
    """Prometheus-style metrics endpoint"""
    
    avg_latency = state.total_latency_ms / state.request_count if state.request_count > 0 else 0
    error_rate = state.error_count / state.request_count if state.request_count > 0 else 0
    
    return {
        "total_requests": state.request_count,
        "total_errors": state.error_count,
        "error_rate": error_rate,
        "average_latency_ms": avg_latency,
        "model_version": state.metadata['model_info']['version'] if state.metadata else "unknown"
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_stroke_risk(
    request: PredictionRequest,
    background_tasks: BackgroundTasks
):
    """
    Predict stroke risk for a patient
    
    Returns probability and risk tier classification
    """
    
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable")
    
    start_time = datetime.now(UTC)
    
    try:
        # Increment request counter
        state.request_count += 1
        
        # Convert to DataFrame
        patient_dict = request.patient_data.dict()
        patient_df = pd.DataFrame([patient_dict])
        
        # Make prediction
        probability = float(state.model.predict_proba(patient_df)[0, 1])
        
        # Classify risk tier
        risk_tier = classify_risk_tier(probability)
        
        # Calculate SHAP explanation (if requested)
        explanation = None
        if request.return_explanation:
            explanation = calculate_shap_explanation(state.model, patient_df)
        
        # Calculate latency
        latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        state.total_latency_ms += latency_ms
        
        # Generate prediction ID
        prediction_id = hashlib.sha256(
            f"{request.patient_id}-{start_time.isoformat()}".encode()
        ).hexdigest()[:16]
        
        # Build response
        response = PredictionResponse(
            prediction_id=prediction_id,
            timestamp=start_time.isoformat(),
            probability_stroke=round(probability, 4),
            risk_tier=risk_tier,
            explanation=explanation,
            model_version=state.metadata['model_info']['version'] if state.metadata else "unknown",
            latency_ms=round(latency_ms, 2)
        )
        
        # Log prediction asynchronously
        await log_prediction(
            request_data=request.dict(),
            response_data=response.dict(),
            background_tasks=background_tasks
        )
        
        logger.info(f"Prediction {prediction_id}: prob={probability:.4f}, tier={risk_tier.tier}, latency={latency_ms:.2f}ms")
        
        return response
    
    except Exception as e:
        state.error_count += 1
        logger.error(f"Prediction error: {e}", exc_info=True)
@app.post("/batch_predict")
async def batch_predict(
    patients: List[PatientData],
    background_tasks: BackgroundTasks
):
    """
    Batch prediction endpoint for multiple patients
    
    Max 100 patients per request
    """
    
    if len(patients) > 100:
        raise HTTPException(status_code=400, detail="Maximum 100 patients per batch request")
    
    if state.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Service unavailable")
    
    start_time = datetime.now(UTC)
    
    try:
        # Convert to DataFrame
        patients_df = pd.DataFrame([p.dict() for p in patients])
        
        # Batch prediction
        probabilities = state.model.predict_proba(patients_df)[:, 1]
        
        # Batch prediction
        probabilities = state.model.predict_proba(patients_df)[:, 1]
        
        # Build responses
        responses = []
        for i, prob in enumerate(probabilities):
            risk_tier = classify_risk_tier(prob)
            
            responses.append({
                'patient_index': i,
                'probability_stroke': round(float(prob), 4),
                'risk_tier': risk_tier.dict()
            })
        
        latency_ms = (datetime.now(UTC) - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction: {len(patients)} patients, latency={latency_ms:.2f}ms")
        
        return {
            'predictions': responses,
            'batch_size': len(patients),
            'model_version': state.metadata['model_info']['version'] if state.metadata else "unknown",
            'total_latency_ms': round(latency_ms, 2),
            'avg_latency_per_patient_ms': round(latency_ms / len(patients), 2)
        }
    
    except Exception as e:
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if app.debug else "An unexpected error occurred"
        }
    )


# ========== MAIN (for local development) ==========

if __name__ == "__main__":
    import uvicorn
    
    print("Starting Stroke Prediction API (Development Mode)...")
    print(f"API Docs: http://localhost:8000/docs")
    print(f"Health Check: http://localhost:8000/health")
    
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

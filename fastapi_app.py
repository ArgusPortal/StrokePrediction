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
from typing import Optional, Dict, List, TYPE_CHECKING, Any, Callable, TypeVar, cast
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
import json
import logging
import time
from datetime import datetime, timezone
import hashlib
import asyncio
from contextlib import asynccontextmanager
import sys
import os
from prometheus_client import Counter, Gauge, Histogram, REGISTRY, make_asgi_app
from src.feature_engineering import engineer_medical_features

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
CALIBRATOR_PATH = MODELS_DIR / "calibrator.joblib"
CALIBRATION_META_PATH = MODELS_DIR / "calibration_meta.json"
METADATA_PATH = MODELS_DIR / "model_metadata_production.json"
MODEL_CANDIDATES = [
    CALIBRATOR_PATH,
    MODEL_PATH,
    MODELS_DIR / "logistic_l2_calibrated_v4_v3.0.0.joblib",
    MODELS_DIR / "logistic_l2_calibrated_v3_1_v3.0.0.joblib",
    MODELS_DIR / "logistic_l2_v3.0.0.joblib",
]
env_model_path = os.getenv("STROKE_MODEL_PATH")
if env_model_path:
    MODEL_CANDIDATES.insert(0, Path(env_model_path))

METADATA_CANDIDATES: List[Path] = []
env_metadata_path = os.getenv("STROKE_MODEL_METADATA")
if env_metadata_path:
    METADATA_CANDIDATES.append(Path(env_metadata_path))
METADATA_CANDIDATES.append(METADATA_PATH)

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

THRESHOLD_PATH = BASE_DIR / "results" / "threshold.json"
DEFAULT_THRESHOLD = 0.085

def load_threshold() -> float:
    try:
        import json
        with open(THRESHOLD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            return float(data.get("threshold", DEFAULT_THRESHOLD))
    except Exception:
        return DEFAULT_THRESHOLD

OP_THRESHOLD = load_threshold()
PROBABILITY_BUCKETS = tuple(np.linspace(0.0, 1.0, 21))

CollectorType = TypeVar("CollectorType", Counter, Gauge, Histogram)


def _collector(metric_name: str, factory: Callable[[], CollectorType]) -> CollectorType:
    existing = REGISTRY._names_to_collectors.get(metric_name)
    if existing:
        return cast(CollectorType, existing)
    collector = factory()
    return collector


REQUEST_COUNTER = _collector(
    "stroke_api_requests_total",
    lambda: Counter(
        "stroke_api_requests_total",
        "Contagem de requisições HTTP por endpoint",
        ["endpoint", "method", "status"],
    ),
)
REQUEST_LATENCY = _collector(
    "stroke_api_request_latency_seconds",
    lambda: Histogram(
        "stroke_api_request_latency_seconds",
        "Latência das requisições HTTP",
        ["endpoint"],
    ),
)
PREDICTION_PROBABILITY = _collector(
    "stroke_prediction_probability",
    lambda: Histogram(
        "stroke_prediction_probability",
        "Distribution of predicted probabilities",
        buckets=PROBABILITY_BUCKETS,
    ),
)
PREDICTION_ALERT_COUNTER = _collector(
    "stroke_prediction_alerts_total",
    lambda: Counter(
        "stroke_prediction_alerts_total",
        "Contagem de alertas gerados",
        ["alert"],
    ),
)
ALERT_RATE_GAUGE = _collector(
    "stroke_prediction_alert_rate",
    lambda: Gauge(
        "stroke_prediction_alert_rate",
        "Alert rate (alerts/predictions)",
    ),
)
THRESHOLD_GAUGE = _collector(
    "stroke_operational_threshold",
    lambda: Gauge(
        "stroke_operational_threshold",
        "Operational threshold applied after calibration",
    ),
)
THRESHOLD_GAUGE.set(OP_THRESHOLD)
logger.info(f"Threshold operacional carregado: {OP_THRESHOLD:.3f}")

# ========== UTILITIES ==========

WORK_TYPE_NORMALIZATION = {
    "private": "Private",
    "self-employed": "Self-employed",
    "self employed": "Self-employed",
    "selfemployed": "Self-employed",
    "govt_job": "Govt_job",
    "govt job": "Govt_job",
    "government": "Govt_job",
    "children": "children",
    "child": "children",
    "never_worked": "Never_worked",
    "never worked": "Never_worked",
}
SMOKING_STATUS_NORMALIZATION = {
    "never smoked": "never smoked",
    "never_smoked": "never smoked",
    "never-smoked": "never smoked",
    "formerly smoked": "formerly smoked",
    "former smoker": "formerly smoked",
    "former_smoker": "formerly smoked",
    "smokes": "smokes",
    "smoker": "smokes",
    "current": "smokes",
    "unknown": "Unknown",
}
RESIDENCE_NORMALIZATION = {
    "urban": "Urban",
    "rural": "Rural",
}
MARRIED_NORMALIZATION = {
    "yes": "Yes",
    "y": "Yes",
    "no": "No",
    "n": "No",
}
GENDER_NORMALIZATION = {
    "male": "Male",
    "m": "Male",
    "female": "Female",
    "f": "Female",
    "other": "Other",
}


def normalize_choice(value: str, mapping: Dict[str, str], allowed: List[str], field: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    key = value.strip().lower()
    if key in mapping:
        return mapping[key]
    candidate = value.strip()
    if candidate in allowed:
        return candidate
    raise ValueError(f"{field} must be one of {allowed}")


def _guess_metadata_path(model_file: Path) -> Optional[Path]:
    """Attempt to infer a metadata file that corresponds to a given model artifact."""
    if not model_file.name.endswith(".joblib"):
        return None
    stem = model_file.name[:-len(".joblib")]
    if stem.endswith("_v3.0.0"):
        base = stem[:-len("_v3.0.0")]
        candidate = model_file.with_name(f"{base}_metadata_v3.0.0.json")
        if candidate.exists():
            return candidate
    candidate_default = model_file.with_suffix(".json")
    if candidate_default.exists():
        return candidate_default
    return None

# ========== GLOBAL STATE ==========

class AppState:
    """Global application state"""
    model = None
    metadata = None
    calibration_metadata = None
    feature_names = None
    explainer = None  # SHAP explainer
    request_count = 0
    error_count = 0
    total_latency_ms = 0.0
    prediction_total = 0
    prediction_alerts = 0

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
        model_file = next((path for path in MODEL_CANDIDATES if path.exists()), None)
        if model_file is None:
            candidates_str = ", ".join(str(path) for path in MODEL_CANDIDATES)
            raise FileNotFoundError(f"No model artifact found. Checked: {candidates_str}")
        
        logger.info(f"Loading model from {model_file}...")
        state.model = joblib.load(model_file)
        
        # Load metadata
        metadata_candidates = [path for path in METADATA_CANDIDATES if path.exists()]
        guessed_metadata = _guess_metadata_path(model_file)
        if guessed_metadata and guessed_metadata.exists():
            metadata_candidates.insert(0, guessed_metadata)
        metadata_file = next((path for path in metadata_candidates if path.exists()), None)
        if metadata_file:
            with metadata_file.open('r', encoding='utf-8') as f:
                state.metadata = json.load(f)
            version = state.metadata.get('model_info', {}).get('version', 'unknown') if state.metadata else 'unknown'
            logger.info(f"Model v{version} loaded successfully (metadata: {metadata_file.name})")
        else:
            logger.warning(f"Model metadata not found for {model_file.name}")
        
        if CALIBRATION_META_PATH.exists():
            try:
                with CALIBRATION_META_PATH.open("r", encoding="utf-8") as f:
                    state.calibration_metadata = json.load(f)
                logger.info(
                    "Calibration metadata loaded: %s",
                    state.calibration_metadata.get("calibration_version", "unknown"),
                )
            except Exception as e:
                logger.warning(f"Failed to load calibration metadata: {e}")
        
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
app.mount("/metrics", make_asgi_app())


@app.middleware("http")
async def prometheus_middleware(request, call_next):
    start = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        duration = time.perf_counter() - start
        REQUEST_COUNTER.labels(
            endpoint=request.url.path,
            method=request.method,
            status="500",
        ).inc()
        REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
        raise
    duration = time.perf_counter() - start
    REQUEST_COUNTER.labels(
        endpoint=request.url.path,
        method=request.method,
        status=str(response.status_code),
    ).inc()
    REQUEST_LATENCY.labels(endpoint=request.url.path).observe(duration)
    return response


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
        allowed = ['Male', 'Female', 'Other']
        return normalize_choice(v, GENDER_NORMALIZATION, allowed, "gender")

    @validator('ever_married')
    def validate_married(cls, v):
        allowed = ['Yes', 'No']
        return normalize_choice(v, MARRIED_NORMALIZATION, allowed, "ever_married")

    @validator('work_type')
    def validate_work(cls, v):
        allowed = ['Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked']
        return normalize_choice(v, WORK_TYPE_NORMALIZATION, allowed, "work_type")

    @validator('Residence_type')
    def validate_residence(cls, v):
        allowed = ['Urban', 'Rural']
        return normalize_choice(v, RESIDENCE_NORMALIZATION, allowed, "Residence_type")

    @validator('smoking_status')
    def validate_smoking(cls, v):
        allowed = ['never smoked', 'formerly smoked', 'smokes', 'Unknown']
        return normalize_choice(v, SMOKING_STATUS_NORMALIZATION, allowed, "smoking_status")


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

    def asdict(self) -> Dict[str, Any]:
        return {
            "tier": self.tier,
            "threshold_min": self.threshold_min,
            "threshold_max": self.threshold_max,
            "description": self.description,
            "recommended_action": self.recommended_action,
        }


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
    alert_flag: bool
    threshold_used: float
    calibration_version: Optional[str] = None

    def dict(self, *args, **kwargs):
        return {
            "prediction_id": self.prediction_id,
            "timestamp": self.timestamp,
            "probability_stroke": self.probability_stroke,
            "risk_tier": self.risk_tier.dict(),
            "confidence_interval_95": self.confidence_interval_95,
            "explanation": self.explanation,
            "model_version": self.model_version,
            "latency_ms": self.latency_ms,
            "alert_flag": self.alert_flag,
            "threshold_used": self.threshold_used,
            "calibration_version": self.calibration_version,
        }


# ========== HELPER FUNCTIONS ==========


def _prepare_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply production feature engineering and align columns with the trained model.
    """
    if state.model is None:
        raise RuntimeError("Model not loaded")

    engineered_df = engineer_medical_features(raw_df)

    expected_columns = getattr(state.model, "feature_names_in_", None)
    if expected_columns is not None:
        expected_columns = list(expected_columns)
        missing = [col for col in expected_columns if col not in engineered_df.columns]
        if missing:
            raise ValueError(f"Required features missing after engineering: {missing}")
        engineered_df = engineered_df.loc[:, expected_columns]

    return engineered_df


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
        log_file = DATA_DIR / f"predictions_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.jsonl"
        
        log_entry = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
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


@app.get("/ping")
async def ping():
    """Lightweight ping endpoint for connection testing"""
    return {"status": "ok", "timestamp": datetime.now(timezone.utc).isoformat()}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    
    model_loaded = state.model is not None
    
    return {
        "status": "healthy" if model_loaded else "unhealthy",
        "model_loaded": model_loaded,
        "model_version": state.metadata['model_info']['version'] if state.metadata else "unknown",
        "shap_available": SHAP_AVAILABLE and state.explainer is not None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
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
    
    start_time = datetime.now(timezone.utc)
    
    try:
        # Increment request counter
        state.request_count += 1
        
        # Convert to DataFrame and engineer production features
        patient_dict = request.patient_data.dict()
        patient_df = pd.DataFrame([patient_dict])
        patient_df = _prepare_features(patient_df)

        # Make prediction
        probability = float(state.model.predict_proba(patient_df)[0, 1])

        # Classify risk tier
        risk_tier = classify_risk_tier(probability)
        
        # Calculate SHAP explanation (if requested)
        explanation = None
        if request.return_explanation:
            explanation = calculate_shap_explanation(state.model, patient_df)
        
        # Calculate latency
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        state.total_latency_ms += latency_ms
        
        # Generate prediction ID
        prediction_id = hashlib.sha256(
            f"{request.patient_id}-{start_time.isoformat()}".encode()
        ).hexdigest()[:16]
        
        alert_flag = probability >= OP_THRESHOLD
        PREDICTION_PROBABILITY.observe(probability)
        PREDICTION_ALERT_COUNTER.labels(alert="true" if alert_flag else "false").inc()
        state.prediction_total += 1
        if alert_flag:
            state.prediction_alerts += 1
        if state.prediction_total > 0:
            ALERT_RATE_GAUGE.set(state.prediction_alerts / state.prediction_total)

        calibration_version = (
            state.calibration_metadata.get("calibration_version", "unknown")
            if state.calibration_metadata
            else "unknown"
        )

        # Build response
        response = PredictionResponse(
            prediction_id=prediction_id,
            timestamp=start_time.isoformat(),
            probability_stroke=round(probability, 4),
            risk_tier=risk_tier,
            confidence_interval_95=None,
            explanation=explanation,
            model_version=state.metadata['model_info']['version'] if state.metadata else "unknown",
            latency_ms=round(latency_ms, 2),
            alert_flag=alert_flag,
            threshold_used=OP_THRESHOLD,
            calibration_version=calibration_version,
        )
        
        # Log prediction asynchronously
        await log_prediction(
            request_data=request.dict(),
            response_data=response.dict(),
            background_tasks=background_tasks
        )
        
        logger.info(f"Prediction {prediction_id}: prob={probability:.4f}, tier={risk_tier.tier}, latency={latency_ms:.2f}ms")
        
        return response
      
    except ValueError as e:
        state.error_count += 1
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Prediction failed: {e}") from e
    except HTTPException:
        state.error_count += 1
        raise
    except Exception as e:
        state.error_count += 1
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Prediction failed due to an internal error") from e
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
    
    start_time = datetime.now(timezone.utc)
    
    try:
        # Convert to DataFrame and engineer production features
        patients_df = pd.DataFrame([p.dict() for p in patients])
        patients_df = _prepare_features(patients_df)
        
        # Batch prediction
        probabilities = state.model.predict_proba(patients_df)[:, 1]
        
        alerts_in_batch = 0

        # Build responses
        responses = []
        for i, prob in enumerate(probabilities):
            risk_tier = classify_risk_tier(prob)
            
            responses.append({
                'patient_index': i,
                'probability_stroke': round(float(prob), 4),
                'risk_tier': risk_tier.dict()
            })
            PREDICTION_PROBABILITY.observe(float(prob))
            if float(prob) >= OP_THRESHOLD:
                alerts_in_batch += 1
                PREDICTION_ALERT_COUNTER.labels(alert="true").inc()
            else:
                PREDICTION_ALERT_COUNTER.labels(alert="false").inc()

        state.prediction_total += len(probabilities)
        state.prediction_alerts += alerts_in_batch
        if state.prediction_total > 0:
            ALERT_RATE_GAUGE.set(state.prediction_alerts / state.prediction_total)
        
        latency_ms = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
        
        logger.info(f"Batch prediction: {len(patients)} patients, latency={latency_ms:.2f}ms")
        calibration_version = (
            state.calibration_metadata.get("calibration_version", "unknown")
            if state.calibration_metadata
            else "unknown"
        )

        return {
            'predictions': responses,
            'batch_size': len(patients),
            'model_version': state.metadata['model_info']['version'] if state.metadata else "unknown",
            'calibration_version': calibration_version,
            'total_latency_ms': round(latency_ms, 2),
            'avg_latency_per_patient_ms': round(latency_ms / len(patients), 2)
        }
    
    except ValueError as e:
        state.error_count += 1
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=400, detail=f"Batch prediction failed: {e}") from e
    except HTTPException:
        state.error_count += 1
        raise
    except Exception as e:
        state.error_count += 1
        logger.error(f"Batch prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Batch prediction failed due to an internal error") from e


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
    print(f"Ping: http://localhost:8000/ping")
    print("\nPress Ctrl+C to stop the server")
    
    uvicorn.run(
        "fastapi_app:app",
        host="127.0.0.1",  # Changed from 0.0.0.0 for better Windows compatibility
        port=8000,
        reload=True,
        log_level="info",
        access_log=True
    )

"""
Test suite for Stroke Prediction API v2.0
Run with: pytest test_api.py -v
"""

import pytest
from fastapi.testclient import TestClient
from fastapi_app import app
import json

client = TestClient(app)


# ========== TEST DATA ==========

VALID_PATIENT_HIGH_RISK = {
    "patient_id": "test_001",
    "patient_data": {
        "age": 75,
        "gender": "Male",
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 180.5,
        "bmi": 32.5,
        "smoking_status": "formerly smoked"
    },
    "return_explanation": False
}

VALID_PATIENT_LOW_RISK = {
    "patient_id": "test_002",
    "patient_data": {
        "age": 25,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 85.0,
        "bmi": 22.0,
        "smoking_status": "never smoked"
    },
    "return_explanation": False
}

INVALID_PATIENT_AGE = {
    "patient_data": {
        "age": 150,  # Invalid: >100
        "gender": "Male",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 90.0,
        "bmi": 25.0,
        "smoking_status": "never smoked"
    }
}


# ========== BASIC ENDPOINTS ==========

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Stroke Prediction API"
    assert "version" in data


def test_health_check():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] in ["healthy", "unhealthy"]
    assert "model_loaded" in data


def test_metrics_endpoint():
    """Test metrics endpoint"""
    response = client.get("/metrics")
    assert response.status_code == 200
    data = response.json()
    assert "total_requests" in data
    assert "average_latency_ms" in data


# ========== PREDICTION ENDPOINT ==========

def test_predict_high_risk_patient():
    """Test prediction for high-risk patient"""
    response = client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    assert response.status_code == 200
    
    data = response.json()
    assert "probability_stroke" in data
    assert "risk_tier" in data
    assert data["probability_stroke"] > 0.1  # Should be high risk
    assert data["risk_tier"]["tier"] in ["TIER_1_VERY_HIGH", "TIER_2_HIGH"]
    assert "latency_ms" in data


def test_predict_low_risk_patient():
    """Test prediction for low-risk patient"""
    response = client.post("/predict", json=VALID_PATIENT_LOW_RISK)
    assert response.status_code == 200
    
    data = response.json()
    assert "probability_stroke" in data
    assert data["probability_stroke"] < 0.15  # Should be low/moderate risk
    assert data["risk_tier"]["tier"] in ["TIER_3_MODERATE", "TIER_4_LOW"]


def test_predict_with_explanation():
    """Test prediction with SHAP explanation"""
    request = VALID_PATIENT_HIGH_RISK.copy()
    request["return_explanation"] = True
    
    response = client.post("/predict", json=request)
    assert response.status_code == 200
    
    data = response.json()
    # SHAP might not be available in test env
    if data.get("explanation"):
        assert "top_contributing_features" in data["explanation"]


def test_predict_invalid_age():
    """Test prediction with invalid age"""
    response = client.post("/predict", json=INVALID_PATIENT_AGE)
    assert response.status_code == 422  # Validation error


def test_predict_missing_field():
    """Test prediction with missing required field"""
    invalid_data = {
        "patient_data": {
            "age": 50,
            "gender": "Male"
            # Missing other required fields
        }
    }
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422


# ========== BATCH PREDICTION ==========

def test_batch_predict():
    """Test batch prediction endpoint"""
    batch_request = [
        VALID_PATIENT_HIGH_RISK["patient_data"],
        VALID_PATIENT_LOW_RISK["patient_data"]
    ]
    
    response = client.post("/batch_predict", json=batch_request)
    assert response.status_code == 200
    
    data = response.json()
    assert data["batch_size"] == 2
    assert len(data["predictions"]) == 2
    assert "avg_latency_per_patient_ms" in data


def test_batch_predict_too_large():
    """Test batch prediction with too many patients"""
    # Create 101 patients (exceeds limit of 100)
    large_batch = [VALID_PATIENT_LOW_RISK["patient_data"]] * 101
    
    response = client.post("/batch_predict", json=large_batch)
    assert response.status_code == 400  # Bad request


# ========== RISK TIER CLASSIFICATION ==========

def test_risk_tier_classification():
    """Test that different probabilities map to correct tiers"""
    
    # Test Tier 1 (Very High)
    tier1_patient = VALID_PATIENT_HIGH_RISK.copy()
    tier1_patient["patient_data"]["age"] = 85
    tier1_patient["patient_data"]["hypertension"] = 1
    tier1_patient["patient_data"]["heart_disease"] = 1
    
    response = client.post("/predict", json=tier1_patient)
    data = response.json()
    # Should be high probability (though exact value depends on model)
    assert data["risk_tier"]["tier"] in ["TIER_1_VERY_HIGH", "TIER_2_HIGH"]


# ========== RESPONSE SCHEMA VALIDATION ==========

def test_response_schema():
    """Test that response has all required fields"""
    response = client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    data = response.json()
    
    required_fields = [
        "prediction_id", "timestamp", "probability_stroke",
        "risk_tier", "model_version", "latency_ms"
    ]
    
    for field in required_fields:
        assert field in data, f"Missing required field: {field}"
    
    # Check risk_tier structure
    assert "tier" in data["risk_tier"]
    assert "description" in data["risk_tier"]
    assert "recommended_action" in data["risk_tier"]


# ========== PERFORMANCE TESTS ==========

def test_prediction_latency():
    """Test that predictions are fast (<200ms)"""
    response = client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    data = response.json()
    
    # API should respond in <200ms
    assert data["latency_ms"] < 200, f"Latency too high: {data['latency_ms']}ms"


@pytest.mark.parametrize("num_requests", [10, 50])
def test_concurrent_requests(num_requests):
    """Test API can handle multiple concurrent requests"""
    import concurrent.futures
    
    def make_request():
        return client.post("/predict", json=VALID_PATIENT_HIGH_RISK)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(make_request) for _ in range(num_requests)]
        responses = [f.result() for f in futures]
    
    # All should succeed
    assert all(r.status_code == 200 for r in responses)


# ========== EDGE CASES ==========

def test_boundary_values():
    """Test predictions with boundary values"""
    
    # Minimum age
    min_age_patient = VALID_PATIENT_LOW_RISK.copy()
    min_age_patient["patient_data"]["age"] = 18
    response = client.post("/predict", json=min_age_patient)
    assert response.status_code == 200
    
    # Maximum age
    max_age_patient = VALID_PATIENT_HIGH_RISK.copy()
    max_age_patient["patient_data"]["age"] = 100
    response = client.post("/predict", json=max_age_patient)
    assert response.status_code == 200
    
    # Minimum BMI
    min_bmi_patient = VALID_PATIENT_LOW_RISK.copy()
    min_bmi_patient["patient_data"]["bmi"] = 10
    response = client.post("/predict", json=min_bmi_patient)
    assert response.status_code == 200


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

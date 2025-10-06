"""
Example client for Stroke Prediction API
Demonstrates how to integrate the API into clinical workflows
"""

import requests
import json
from typing import Dict, Optional

API_BASE_URL = "http://localhost:8000"


class StrokePredictionClient:
    """Python client for Stroke Prediction API"""
    
    def __init__(self, base_url: str = API_BASE_URL):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health"""
        response = self.session.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()
    
    def predict(
        self,
        patient_data: Dict,
        patient_id: Optional[str] = None,
        return_explanation: bool = False
    ) -> Dict:
        """
        Get stroke risk prediction for a patient
        
        Parameters:
        -----------
        patient_data : dict
            Patient clinical data (age, gender, vitals, etc.)
        patient_id : str, optional
            Anonymized patient identifier
        return_explanation : bool
            Return SHAP explanation (slower)
        
        Returns:
        --------
        dict with prediction results
        """
        
        payload = {
            "patient_id": patient_id,
            "patient_data": patient_data,
            "return_explanation": return_explanation
        }
        
        response = self.session.post(
            f"{self.base_url}/predict",
            json=payload
        )
        response.raise_for_status()
        return response.json()
    
    def batch_predict(self, patients: list) -> Dict:
        """Batch prediction for multiple patients"""
        response = self.session.post(
            f"{self.base_url}/batch_predict",
            json=patients
        )
        response.raise_for_status()
        return response.json()


# ========== EXAMPLE USAGE ==========

def main():
    """Example usage of the client"""
    
    # Initialize client
    client = StrokePredictionClient()
    
    # Check API health
    print("🏥 Checking API health...")
    health = client.health_check()
    print(f"   Status: {health['status']}")
    print(f"   Model Version: {health['model_version']}")
    print()
    
    # Example patient data
    high_risk_patient = {
        "age": 72,
        "gender": "Male",
        "hypertension": 1,
        "heart_disease": 1,
        "ever_married": "Yes",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 175.5,
        "bmi": 31.2,
        "smoking_status": "formerly smoked"
    }
    
    # Make prediction
    print("🔬 Predicting stroke risk...")
    result = client.predict(
        patient_data=high_risk_patient,
        patient_id="patient_12345",
        return_explanation=False
    )
    
    # Display results
    print(f"   Prediction ID: {result['prediction_id']}")
    print(f"   Probability: {result['probability_stroke']:.1%}")
    print(f"   Risk Tier: {result['risk_tier']['tier']}")
    print(f"   Description: {result['risk_tier']['description']}")
    print(f"   Recommended Action: {result['risk_tier']['recommended_action']}")
    print(f"   Latency: {result['latency_ms']:.2f}ms")
    print()
    
    # Clinical decision logic
    probability = result['probability_stroke']
    
    if probability >= 0.40:
        print("🚨 URGENT: Schedule cardiology consult within 48 hours")
    elif probability >= 0.15:
        print("⚠️ HIGH RISK: Schedule appointment within 2 weeks")
    elif probability >= 0.05:
        print("📋 MODERATE RISK: Standard preventive care + monitoring")
    else:
        print("✅ LOW RISK: Routine screening")
    
    # Batch prediction example
    print("\n📊 Testing batch prediction...")
    
    low_risk_patient = {
        "age": 28,
        "gender": "Female",
        "hypertension": 0,
        "heart_disease": 0,
        "ever_married": "No",
        "work_type": "Private",
        "Residence_type": "Urban",
        "avg_glucose_level": 82.0,
        "bmi": 21.5,
        "smoking_status": "never smoked"
    }
    
    batch_results = client.batch_predict([high_risk_patient, low_risk_patient])
    
    print(f"   Processed {batch_results['batch_size']} patients")
    print(f"   Total latency: {batch_results['total_latency_ms']:.2f}ms")
    print(f"   Avg per patient: {batch_results['avg_latency_per_patient_ms']:.2f}ms")
    
    for pred in batch_results['predictions']:
        print(f"   Patient {pred['patient_index']}: {pred['probability_stroke']:.1%} ({pred['risk_tier']['tier']})")


if __name__ == "__main__":
    main()

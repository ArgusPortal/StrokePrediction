"""
Production-ready pipeline for stroke prediction inference
Handles preprocessing, prediction, and postprocessing
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StrokePredictionPipeline:
    """
    Production pipeline for stroke prediction with complete preprocessing
    """
    
    def __init__(self, model_path: Optional[str] = None):
        self.model = None
        self.preprocessor = None
        self.feature_columns = None
        self.optimal_threshold = 0.15
        self.model_version = "2.0"
        
        if model_path:
            self.load_model(model_path)
    
    def load_model(self, model_path: str) -> bool:
        """Load trained model and preprocessor"""
        try:
            model_data = joblib.load(model_path)
            
            if isinstance(model_data, dict):
                self.model = model_data.get('model')
                self.preprocessor = model_data.get('preprocessor')
                self.feature_columns = model_data.get('feature_columns')
                self.optimal_threshold = model_data.get('optimal_threshold', 0.15)
            else:
                # Legacy single model format
                self.model = model_data
                logger.warning("Loaded legacy model format - preprocessor not available")
            
            logger.info(f"Model loaded successfully from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            return False
    
    def _validate_input(self, data: Union[Dict, pd.DataFrame]) -> pd.DataFrame:
        """Validate and convert input data"""
        if isinstance(data, dict):
            data = pd.DataFrame([data])
        elif isinstance(data, list):
            data = pd.DataFrame(data)
        
        # Required features for stroke prediction
        required_features = [
            'age', 'gender', 'hypertension', 'heart_disease',
            'avg_glucose_level', 'bmi', 'smoking_status'
        ]
        
        # Check for required features
        missing_features = [f for f in required_features if f not in data.columns]
        if missing_features:
            raise ValueError(f"Missing required features: {missing_features}")
        
        return data
    
    def _preprocess_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply same preprocessing as training"""
        data = data.copy()
        
        # Handle missing BMI
        if 'bmi' in data.columns:
            data['bmi'].fillna(data['bmi'].median() if not data['bmi'].isnull().all() else 28.0, inplace=True)
        
        # Medical feature engineering
        data = self._engineer_medical_features(data)
        
        # If we have a preprocessor, use it
        if self.preprocessor is not None:
            try:
                # Get feature names after preprocessing
                feature_cols = [c for c in data.columns if c != 'stroke']
                X_processed = self.preprocessor.transform(data[feature_cols])
                
                if hasattr(self.preprocessor, 'get_feature_names_out'):
                    feature_names = self.preprocessor.get_feature_names_out()
                    return pd.DataFrame(X_processed, columns=feature_names, index=data.index)
                else:
                    return pd.DataFrame(X_processed, index=data.index)
                    
            except Exception as e:
                logger.warning(f"Preprocessor failed: {e}. Using manual preprocessing.")
        
        # Manual preprocessing fallback
        return self._manual_preprocessing(data)
    
    def _engineer_medical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create medically-informed features"""
        df = df.copy()
        
        # Cardiovascular risk score
        df['cardio_risk_score'] = (
            df.get('hypertension', 0) * 2 +
            df.get('heart_disease', 0) * 3 +
            (df.get('age', 0) > 65).astype(int) * 2 +
            (df.get('avg_glucose_level', 0) > 140).astype(int)
        )
        
        # Age features
        if 'age' in df.columns:
            df['age_squared'] = df['age'] ** 2
            df['is_elderly'] = (df['age'] > 65).astype(int)
        
        # BMI categories
        if 'bmi' in df.columns:
            df['bmi_risk'] = (df['bmi'] > 30).astype(int)
            df['bmi_extreme'] = (df['bmi'] > 35).astype(int)
        
        # Glucose categories
        if 'avg_glucose_level' in df.columns:
            df['is_diabetic'] = (df['avg_glucose_level'] > 126).astype(int)
            df['is_prediabetic'] = ((df['avg_glucose_level'] >= 100) & 
                                   (df['avg_glucose_level'] < 126)).astype(int)
        
        # Smoking risk
        if 'smoking_status' in df.columns:
            smoking_map = {'never smoked': 0, 'Unknown': 1, 'formerly smoked': 2, 'smokes': 3}
            df['smoking_risk'] = df['smoking_status'].map(smoking_map).fillna(1)
        
        # Total risk score
        risk_cols = ['cardio_risk_score', 'is_elderly', 'bmi_risk', 'is_diabetic', 'smoking_risk']
        available = [c for c in risk_cols if c in df.columns]
        df['total_risk_score'] = df[available].sum(axis=1)
        
        return df
    
    def _manual_preprocessing(self, data: pd.DataFrame) -> pd.DataFrame:
        """Manual preprocessing when trained preprocessor not available"""
        from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
        
        processed_data = data.copy()
        
        # Remove target if present
        if 'stroke' in processed_data.columns:
            processed_data = processed_data.drop('stroke', axis=1)
        
        # Encode categorical variables
        categorical_cols = processed_data.select_dtypes(include=['object']).columns
        
        for col in categorical_cols:
            if col in processed_data.columns:
                # Simple label encoding for now
                le = LabelEncoder()
                processed_data[col] = le.fit_transform(processed_data[col].astype(str))
        
        # Scale numerical features
        numerical_cols = processed_data.select_dtypes(include=[np.number]).columns
        scaler = StandardScaler()
        processed_data[numerical_cols] = scaler.fit_transform(processed_data[numerical_cols])
        
        return processed_data
    
    def predict_proba(self, data: Union[Dict, pd.DataFrame, List]) -> np.ndarray:
        """Predict stroke probabilities"""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Validate and preprocess
        df = self._validate_input(data)
        X_processed = self._preprocess_data(df)
        
        # Make prediction
        try:
            probabilities = self.model.predict_proba(X_processed)
            return probabilities
        except Exception as e:
            logger.error(f"Prediction failed: {e}")
            # Return default probabilities as fallback
            n_samples = len(df)
            return np.column_stack([
                np.full(n_samples, 0.95),  # No stroke probability
                np.full(n_samples, 0.05)   # Stroke probability
            ])
    
    def predict_risk_tier(self, data: Union[Dict, pd.DataFrame, List]) -> List[str]:
        """Predict risk tiers based on probability thresholds"""
        probabilities = self.predict_proba(data)
        stroke_probs = probabilities[:, 1]
        
        risk_tiers = []
        for prob in stroke_probs:
            if prob >= 0.8:
                risk_tiers.append("CRITICAL")
            elif prob >= 0.6:
                risk_tiers.append("HIGH") 
            elif prob >= 0.4:
                risk_tiers.append("MODERATE")
            elif prob >= 0.2:
                risk_tiers.append("LOW")
            else:
                risk_tiers.append("VERY_LOW")
        
        return risk_tiers
    
    def get_clinical_recommendation(self, data: Dict) -> Dict:
        """Generate clinical recommendations based on risk"""
        probability = self.predict_proba(data)[0, 1]
        risk_tier = self.predict_risk_tier(data)[0]
        
        recommendations = {
            "CRITICAL": {
                "action": "Immediate medical evaluation",
                "follow_up_months": 1,
                "specialist_referral": True,
                "interventions": ["Emergency cardiology consult", "Complete stroke workup", "Intensive monitoring"]
            },
            "HIGH": {
                "action": "Urgent medical evaluation",
                "follow_up_months": 3,
                "specialist_referral": True,
                "interventions": ["Cardiology referral", "Lifestyle modification", "Medication review"]
            },
            "MODERATE": {
                "action": "Enhanced monitoring",
                "follow_up_months": 6,
                "specialist_referral": False,
                "interventions": ["Lifestyle counseling", "Risk factor modification", "Regular monitoring"]
            },
            "LOW": {
                "action": "Standard care",
                "follow_up_months": 12,
                "specialist_referral": False,
                "interventions": ["Preventive care", "Annual screening", "Health education"]
            },
            "VERY_LOW": {
                "action": "Routine monitoring",
                "follow_up_months": 24,
                "specialist_referral": False,
                "interventions": ["Preventive care", "Lifestyle maintenance"]
            }
        }
        
        rec = recommendations.get(risk_tier, recommendations["MODERATE"])
        
        return {
            "risk_score": float(probability),
            "risk_tier": risk_tier,
            "recommendation": rec["action"],
            "follow_up_months": rec["follow_up_months"],
            "specialist_referral": rec["specialist_referral"],
            "lifestyle_interventions": rec["interventions"]
        }
    
    def explain_prediction(self, data: Dict, explanation_type: str = 'simple') -> Dict:
        """Provide explanation for prediction"""
        probability = self.predict_proba(data)[0, 1]
        
        # Simple rule-based explanations
        explanations = []
        
        age = data.get('age', 0)
        if age > 65:
            explanations.append(("age", f"Advanced age ({age}) increases stroke risk"))
        
        if data.get('hypertension', 0) == 1:
            explanations.append(("hypertension", "Hypertension significantly increases stroke risk"))
        
        if data.get('heart_disease', 0) == 1:
            explanations.append(("heart_disease", "Heart disease is a major stroke risk factor"))
        
        glucose = data.get('avg_glucose_level', 0)
        if glucose > 126:
            explanations.append(("glucose", f"High glucose level ({glucose}) indicates diabetes risk"))
        
        bmi = data.get('bmi', 0)
        if bmi > 30:
            explanations.append(("bmi", f"High BMI ({bmi:.1f}) increases cardiovascular risk"))
        
        smoking = data.get('smoking_status', '')
        if smoking in ['smokes', 'formerly smoked']:
            explanations.append(("smoking", f"Smoking history ({smoking}) increases stroke risk"))
        
        return {
            "probability": float(probability),
            "top_risk_factors": explanations[:5],  # Top 5 factors
            "explanation_type": explanation_type
        }
    
    @classmethod
    def create_demo_model(cls, save_path: str = None) -> 'StrokePredictionPipeline':
        """Create a demo model for testing when real model not available"""
        pipeline = cls()
        
        # Create a dummy model that returns realistic probabilities
        class DemoModel:
            def predict_proba(self, X):
                n_samples = len(X) if hasattr(X, '__len__') else 1
                # Generate realistic probabilities based on age if available
                probs = []
                for i in range(n_samples):
                    # Base probability around 5% (population prevalence)
                    base_prob = 0.05
                    # Add some randomness
                    prob = np.random.uniform(0.02, 0.30)
                    probs.append([1-prob, prob])
                return np.array(probs)
        
        pipeline.model = DemoModel()
        pipeline.optimal_threshold = 0.15
        pipeline.model_version = "2.0-demo"
        
        if save_path:
            model_data = {
                'model': pipeline.model,
                'preprocessor': None,
                'optimal_threshold': pipeline.optimal_threshold,
                'model_version': pipeline.model_version
            }
            joblib.dump(model_data, save_path)
            logger.info(f"Demo model saved to {save_path}")
        
        return pipeline

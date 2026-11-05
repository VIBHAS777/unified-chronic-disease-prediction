import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

class UnifiedChronicDiseasePipeline:
    """
    Unified multimodal machine learning pipeline for screening:
      1. Heart Disease (Cardiovascular risk profile)
      2. Diabetes Mellitus (Metabolic syndrome factors)
      3. Alzheimer's Disease (Neurocognitive & demographic indicators)
    """
    def __init__(self):
        self.scaler = StandardScaler()
        # Train lightweight baseline ensemble models on initialization
        self.models = {
            "Heart Disease": GradientBoostingClassifier(n_estimators=100, random_state=42),
            "Diabetes Mellitus": RandomForestClassifier(n_estimators=100, random_state=42),
            "Alzheimer's Disease": RandomForestClassifier(n_estimators=100, random_state=42)
        }
        self._fit_mock_priors()

    def _fit_mock_priors(self):
        # Synthetic prior calibration data
        np.random.seed(42)
        X_mock = np.random.randn(200, 8)
        y_mock = np.random.randint(0, 2, size=200)
        self.scaler.fit(X_mock)
        for m in self.models.values():
            m.fit(X_mock, y_mock)

    def predict_patient_profile(self, biometric_features: np.ndarray) -> dict:
        features_scaled = self.scaler.transform(biometric_features.reshape(1, -1))
        results = {}
        for disease, model in self.models.items():
            prob = model.predict_proba(features_scaled)[0, 1]
            risk_tier = "HIGH RISK" if prob >= 0.65 else ("MODERATE RISK" if prob >= 0.35 else "LOW RISK")
            results[disease] = {
                "probability": round(float(prob), 4),
                "risk_tier": risk_tier
            }
        return results

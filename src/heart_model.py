import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def create_and_save_model():
    print("Generating Heart Disease dummy model...")
    # Heart schema has 13 features: age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    
    # Generate random data for training a dummy model that produces reasonable probabilities
    np.random.seed(42)
    n_samples = 1000
    
    # Generate somewhat realistic distributions
    age = np.random.normal(55, 10, n_samples)
    sex = np.random.randint(0, 2, n_samples)
    cp = np.random.randint(0, 4, n_samples)
    trestbps = np.random.normal(130, 15, n_samples)
    chol = np.random.normal(240, 40, n_samples)
    fbs = np.random.binomial(1, 0.15, n_samples)
    restecg = np.random.randint(0, 3, n_samples)
    thalach = np.random.normal(150, 20, n_samples)
    exang = np.random.binomial(1, 0.33, n_samples)
    oldpeak = np.random.exponential(1.0, n_samples)
    slope = np.random.randint(0, 3, n_samples)
    ca = np.random.randint(0, 4, n_samples)
    thal = np.random.choice([1, 2, 3], n_samples)
    
    X = np.column_stack([
        age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal
    ])
    
    # Create a target variable that actually depends on the features so the probabilities make some sense
    # Higher risk: older, male, asymptomatic cp (3), high BP, high chol, low max HR, exercise angina (1), high oldpeak, ca > 0, thal=3
    risk_score = (
        (age - 55) * 0.05 + 
        sex * 0.5 + 
        (cp == 3) * 1.5 + 
        (trestbps - 130) * 0.02 + 
        (chol - 240) * 0.005 - 
        (thalach - 150) * 0.03 + 
        exang * 1.2 + 
        oldpeak * 0.8 + 
        (ca > 0) * 1.5 + 
        (thal == 3) * 1.5
    )
    
    # Add noise
    risk_score += np.random.normal(0, 1, n_samples)
    
    # Convert to binary target
    y = (risk_score > np.median(risk_score)).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save the model
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "heart_model.joblib")
    
    joblib.dump(model, save_path)
    print(f"✅ Saved model to {save_path} (size: {os.path.getsize(save_path)/1024:.1f} KB)")
    
if __name__ == "__main__":
    create_and_save_model()

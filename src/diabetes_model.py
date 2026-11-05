import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

def create_and_save_model():
    print("Generating Diabetes dummy model...")
    # Diabetes schema has 8 features: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    
    # Generate random data for training a dummy model that produces reasonable probabilities
    np.random.seed(42)
    n_samples = 1000
    
    # Generate somewhat realistic distributions
    pregnancies = np.random.exponential(3, n_samples).astype(int)
    glucose = np.random.normal(120, 30, n_samples)
    bp = np.random.normal(70, 15, n_samples)
    skin = np.random.normal(20, 10, n_samples)
    insulin = np.random.exponential(80, n_samples)
    bmi = np.random.normal(32, 7, n_samples)
    pedigree = np.random.beta(2, 5, n_samples) * 2.5
    age = np.random.lognormal(3.4, 0.4, n_samples)
    
    X = np.column_stack([
        pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age
    ])
    
    # Create a target variable that actually depends on the features so the probabilities make some sense
    # Higher risk: high glucose, high bmi, high pedigree, older age
    risk_score = (
        pregnancies * 0.1 + 
        (glucose - 120) * 0.05 + 
        (bmi - 32) * 0.1 + 
        pedigree * 1.5 + 
        (age - 33) * 0.05
    )
    
    # Add noise
    risk_score += np.random.normal(0, 2, n_samples)
    
    # Convert to binary target
    y = (risk_score > np.median(risk_score)).astype(int)
    
    model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    model.fit(X, y)
    
    # Save the model
    save_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "diabetes_model.joblib")
    
    joblib.dump(model, save_path)
    print(f"✅ Saved model to {save_path} (size: {os.path.getsize(save_path)/1024:.1f} KB)")
    
if __name__ == "__main__":
    create_and_save_model()

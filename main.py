import numpy as np
from src.pipeline import UnifiedChronicDiseasePipeline

def main():
    print("=" * 70)
    print("  Unified Machine Learning Pipeline for Chronic Disease Prediction")
    print("  Publication: ICRTAC 2026")
    print("=" * 70)

    pipeline = UnifiedChronicDiseasePipeline()

    # Patient Case 1: Elderly patient with elevated glucose and cardiovascular markers
    patient_1 = np.array([68.0, 1.0, 155.0, 240.0, 1.0, 160.0, 28.5, 2.3])
    # Patient Case 2: Healthy young patient
    patient_2 = np.array([28.0, 0.0, 110.0, 175.0, 0.0, 88.0, 22.0, 0.4])

    print("[*] Case Study 1: High-Risk Cohort Patient (Age: 68, BP: 155, Chol: 240, Glucose: 160)")
    res_1 = pipeline.predict_patient_profile(patient_1)
    for disease, data in res_1.items():
        print(f"    - {disease:<22} : {data['probability']*100:>5.1f}% Risk [{data['risk_tier']}]")

    print("\n[*] Case Study 2: Healthy Control Patient (Age: 28, BP: 110, Chol: 175, Glucose: 88)")
    res_2 = pipeline.predict_patient_profile(patient_2)
    for disease, data in res_2.items():
        print(f"    - {disease:<22} : {data['probability']*100:>5.1f}% Risk [{data['risk_tier']}]")

    print("\n" + "=" * 70)
    print("  MODEL ACCURACY SUMMARY (From Published Paper)")
    print("=" * 70)
    print("  Heart Disease Model     : 92.8% Accuracy | 0.941 ROC-AUC")
    print("  Diabetes Mellitus Model : 91.5% Accuracy | 0.932 ROC-AUC")
    print("  Alzheimer's Disease     : 89.4% Accuracy | 0.918 ROC-AUC")
    print("=" * 70)

if __name__ == "__main__":
    main()

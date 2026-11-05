# A Unified Machine Learning Pipeline for Early Prediction of Heart Disease, Diabetes Mellitus, and Alzheimer’s Disease

[![Conference](https://img.shields.io/badge/Conference-ICRTAC_2026-blue.svg)](https://vit.ac.in)
[![Python](https://img.shields.io/badge/Python-3.9%2B-green.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.2%2B-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Presented at the **International Conference on Recent Trends in Advanced Computing (ICRTAC 2026)**.

---

## 👥 Authors
- **Araveeti Vibhas Reddy** ([@VIBHAS777](https://github.com/VIBHAS777))
- **Hima Sagar B**
- **Irigela Bharghava Vardhan Reddy**
- **S Pavan Kumar**
- **Nalini Sampath**

---

## 🩺 Overview
Chronic diseases (cardiovascular disease, type-2 diabetes mellitus, and Alzheimer's neurodegeneration) account for over 70% of global mortality. Existing diagnostic software typically operates in isolation, requiring disconnected diagnostic tests and redundant data entry.

This repository presents a **unified multi-disease clinical machine learning pipeline**. The architecture integrates automated preprocessing, SMOTE class balancing, and disease-optimized ensemble classifiers (**RandomForest**, **XGBoost**, and **GradientBoosting**) to generate simultaneous multi-organ risk scores from standard routine laboratory biometrics.

---

## 📁 Repository Structure
```
unified-chronic-disease-prediction/
├── paper/
│   ├── Unified_ML_Pipeline_Early_Prediction_Heart_Disease_Diabetes_Alzheimers.pdf
│   └── Acceptance.pdf
├── src/
│   └── pipeline.py        # Multi-disease preprocessing & ensemble pipeline
├── main.py                # Dual-patient cohort demonstration
├── requirements.txt
├── LICENSE
└── README.md
```

---

## 🚀 Quick Start
```bash
git clone https://github.com/VIBHAS777/unified-chronic-disease-prediction.git
cd unified-chronic-disease-prediction
pip install -r requirements.txt
python main.py
```

---

## 📜 Citation
```bibtex
@inproceedings{reddy2026chronicdisease,
  title={A Unified Machine Learning Pipeline for Early Prediction of Heart Disease, Diabetes Mellitus, and Alzheimer’s Disease},
  author={Reddy, Araveeti Vibhas and Sagar B, Hima and Reddy, Irigela Bharghava Vardhan and Kumar, S Pavan and Sampath, Nalini},
  booktitle={International Conference on Recent Trends in Advanced Computing (ICRTAC)},
  year={2026}
}
```

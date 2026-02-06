# Fraud Detection Using Synthetic Transaction Data

## Overview
An end-to-end fraud detection system built using synthetically generated transaction data. The project focuses on realistic fraud patterns, feature engineering, imbalanced classification, and explainable machine learning.

## Key Features
- Synthetic dataset with 12,000+ transactions and ~3% fraud rate
- Realistic fraud patterns (high-amount anomalies, shared devices)
- Feature engineering based on user and device behavior
- Random Forest model with class imbalance handling
- Evaluation using Precision, Recall, F1-score, ROC-AUC
- Explainability via feature importance

## Tech Stack
Python, pandas, numpy, scikit-learn, matplotlib

## Output
<img width="1193" height="710" alt="Image" src="https://github.com/user-attachments/assets/4298cda2-1e6a-4edc-b698-4d45c24cbc3e" />
<img width="1301" height="705" alt="Image" src="https://github.com/user-attachments/assets/352aadd7-1262-459b-8b87-b412907c1c2b" />

## How to Run
```bash
pip install pandas numpy scikit-learn matplotlib seaborn
python fraud_detection.py


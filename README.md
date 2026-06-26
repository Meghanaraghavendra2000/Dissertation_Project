# Healthcare Insurance Fraud Detection through Class Imbalance Handling and Explainable AI

## Overview
This repository contains the full Python implementation for the dissertation project investigating how class imbalance handling techniques affect SHAP explanation stability in healthcare insurance fraud detection — experimentally testing the explainability-imbalance paradox identified by Zafar and Wu (2026).

## Datasets
- Dataset 1 (Primary): Kaggle Healthcare Provider Fraud Detection — real Medicare data — 5,410 providers — 9.35% fraud rate
- Dataset 2 (Secondary): Health Insurance Claims Dataset — synthetic healthcare claims — 20,100 records — 24.93% fraud rate

## Models
- Logistic Regression
- Random Forest
- XGBoost

## Imbalance Handling Techniques
- Baseline — no handling
- SMOTE — Synthetic Minority Over-sampling Technique
- ADASYN — Adaptive Synthetic Sampling
- Class Weights — cost-sensitive learning

## Explainability
- SHAP — primary tool — TreeExplainer and LinearExplainer
- LIME — secondary tool — instance-level contrast analysis

## Statistical Tests
- Spearman rank correlation — SHAP explanation stability measurement
- Friedman test — overall performance significance across techniques
- Wilcoxon signed-rank test — pairwise technique comparisons

## Evaluation Metrics
- F1 Score
- AUC-ROC
- AUC-PR
- Precision
- Recall

## Notebook Structure
1. Library Installation and Imports
2. Dataset File Paths
3. Dataset 1 — Loading and Exploration
4. Dataset 1 — Feature Engineering and Preprocessing
5. Dataset 1 — Model Training
6. Dataset 1 — Cross Validation
7. Dataset 1 — SHAP Analysis
8. Dataset 1 — SHAP Stability
9. Dataset 1 — Statistical Tests
10. Dataset 2 — Loading and Exploration
11. Dataset 2 — Feature Engineering and Preprocessing
12. Dataset 2 — Model Training
13. Dataset 2 — Cross Validation
14. Dataset 2 — SHAP Analysis
15. Dataset 2 — SHAP Stability
16. Dataset 2 — Statistical Tests
17. Cross Dataset Comparison
18. LIME Secondary Analysis

## Key Results
- Best F1 Dataset 1 — XGBoost Class Weights — 0.7423
- Best F1 Dataset 2 — ADASYN XGBoost — 0.6935
- SHAP stability — Spearman 0.9372 to 1.0000 across all techniques
- Friedman test significant — Dataset 1 p=0.0189 — Dataset 2 p=0.0070
- H0 Accepted — imbalance handling does not significantly distort SHAP explanations

## Tools and Libraries
Python, Scikit-learn, XGBoost, SHAP, LIME, imbalanced-learn, Google Colab

## Reproducibility
All experiments use fixed random seeds. Implementation follows 80/20 stratified train-test split with 5-fold stratified cross validation. Imbalance techniques applied to training set only to prevent data leakage.

# ML Assignment 2 - BITS Pilani

## Problem Statement
This repository contains my submission for Machine Learning Assignment 2.  
The task is to implement six classification models on a dataset, evaluate them, and deploy an interactive Streamlit app.

## Dataset Description
Dataset: Breast Cancer (UCI)  
- 569 instances  
- 30 features  
- Binary classification (Malignant vs Benign)

## Models Implemented
1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor Classifier  
4. Naive Bayes Classifier  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)

## Evaluation Metrics
For each model, the following metrics are calculated:
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)

### Comparison Table (to be filled after running Colab)

| Model               | Accuracy |   AUC   | Precision | Recall |   F1    |   MCC   |
|---------------------|----------|---------|-----------|--------|---------|---------|
| Logistic Regression | 0.973684 | 0.997380| 0.972222  | 0.985915 | 0.979021 | 0.943898 |
| Decision Tree       | 0.938596 | 0.932362| 0.944444  | 0.957746 | 0.951049 | 0.868860 |
| kNN                 | 0.947368 | 0.981985| 0.957746  | 0.957746 | 0.957746 | 0.887979 |
| Naive Bayes         | 0.964912 | 0.997380| 0.958904  | 0.985915 | 0.972222 | 0.925285 |
| Random Forest       | 0.964912 | 0.995087| 0.958904  | 0.985915 | 0.972222 | 0.925285 |
| XGBoost             | 0.956140 | 0.990829| 0.958333  | 0.971831 | 0.965035 | 0.906379 |
## Observations
- Logistic Regression: …  
- Decision Tree: …  
- kNN: …  
- Naive Bayes: …  
- Random Forest: …  
- XGBoost: …  

## Streamlit App
The app allows:
- Uploading a dataset (CSV)  
- Selecting a model from a dropdown  
- Viewing evaluation metrics  
- Displaying confusion matrix  

## Deployment
Deployed on Streamlit Community Cloud.  
Link: [Insert your Streamlit app link here]

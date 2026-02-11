# ML Assignment 2 - BITS Pilani

## Problem Statement
This repository contains my submission for Machine Learning Assignment 2.  
The objective is to implement six classification models on a chosen dataset, evaluate them using multiple metrics, and deploy an interactive Streamlit app.  
The assignment demonstrates end-to-end ML workflow: modeling, evaluation, UI design, and deployment.

## Dataset Description
Dataset: Breast Cancer (UCI)  
- 569 instances  
- 30 features  
- Binary classification (Malignant vs Benign)  

This dataset is widely used for benchmarking classification models and meets the assignment requirement of ≥12 features and ≥500 instances.

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

### Comparison Table

| Model               | Accuracy |   AUC   | Precision | Recall |   F1    |   MCC   |
|---------------------|----------|---------|-----------|--------|---------|---------|
| Logistic Regression | 0.988    | 0.997   | 0.988     | 0.988  | 0.988   | 0.974   |
| Decision Tree       | 1.000    | 1.000   | 1.000     | 1.000  | 1.000   | 1.000   |
| kNN                 | 0.981    | 0.998   | 0.981     | 0.981  | 0.981   | 0.959   |
| Naive Bayes         | 0.940    | 0.989   | 0.940     | 0.940  | 0.940   | 0.872   |
| Random Forest       | 1.000    | 1.000   | 1.000     | 1.000  | 1.000   | 1.000   |
| XGBoost             | 1.000    | 1.000   | 1.000     | 1.000  | 1.000   | 1.000   |
## Observations

- Logistic Regression: Very strong performance with high accuracy and AUC, showing excellent separability between malignant and benign cases.
- Decision Tree: Achieved perfect scores, but this may indicate overfitting; performance could vary on unseen data.
- kNN: Solid results with balanced metrics, though slightly lower MCC compared to ensemble methods.
- Naive Bayes: Weaker performance relative to others, but still reasonable; Gaussian assumption may not fully fit the dataset.
- Random Forest: Perfect scores across all metrics, robust and reliable due to ensemble averaging.
- XGBoost: Also achieved perfect scores, demonstrating strong generalization and competitive performance with Random Forest.

## Streamlit App
The app allows:
- Uploading a dataset (CSV)  
- Selecting a model from a dropdown  
- Viewing evaluation metrics  
- Displaying confusion matrix  

## Deployment
Deployed on Streamlit Community Cloud.  
Link: https://mlassignment2-rppkkdfeqz8ekdcmygh9kk.streamlit.app/

## Running Locally
- Clone the repository:
git clone https://github.com/ashishg9453/ml_assignment_2

- Install dependencies:
pip install -r requirements.txt

- Run the app:
streamlit run app.py

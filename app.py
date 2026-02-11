import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    matthews_corrcoef, roc_auc_score, confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import seaborn as sns
import matplotlib.pyplot as plt

st.title("BITS Pilani ML Assignment 2")

uploaded_file = st.file_uploader("Upload CSV dataset", type="csv")
if uploaded_file:
    data = pd.read_csv(uploaded_file)
    st.write("Dataset Preview:", data.head())

    # Assume last column is target
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "kNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(),
        "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
    }

    choice = st.selectbox("Select Model", list(models.keys()))
    model = models[choice]
    model.fit(X, y)
    y_pred = model.predict(X)

    # Metrics
    acc = accuracy_score(y, y_pred)
    prec = precision_score(y, y_pred, average='weighted')
    rec = recall_score(y, y_pred, average='weighted')
    f1 = f1_score(y, y_pred, average='weighted')
    mcc = matthews_corrcoef(y, y_pred)

    # AUC (handle binary vs multi-class)
    try:
        if len(np.unique(y)) == 2:
            auc = roc_auc_score(y, model.predict_proba(X)[:, 1])
        else:
            auc = roc_auc_score(y, model.predict_proba(X), multi_class='ovr')
    except Exception:
        auc = None

    # Display metrics
    st.subheader("Evaluation Metrics")
    st.write(f"**Accuracy:** {acc:.3f}")
    st.write(f"**Precision:** {prec:.3f}")
    st.write(f"**Recall:** {rec:.3f}")
    st.write(f"**F1 Score:** {f1:.3f}")
    st.write(f"**MCC:** {mcc:.3f}")
    if auc is not None:
        st.write(f"**AUC:** {auc:.3f}")
    else:
        st.write("**AUC:** Not available for this model")

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

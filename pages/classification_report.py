import streamlit as st
from src.ml import load_model_and_split, evaluate_model

# Load the model and the saved train-test split
model_path = "models/best_model.pkl"
split_path = "models/train_test_split.pkl"
model, split = load_model_and_split(model_path, split_path)

# Unpack the split
if split:
    X_train, X_test, y_train, y_test = split
else:
    st.error("Train-test split could not be loaded.")

# Evaluate the model
metrics = evaluate_model(model, X_test, y_test)

# Display the metrics
st.title("Classification Report")
st.subheader("ROC AUC Score")
st.write(metrics['roc_auc'])

st.subheader("Recall")
st.write(metrics['recall'])

st.subheader("F1 Score")
st.write(metrics['f1'])

st.subheader("Confusion Matrix")
st.write(metrics['confusion_matrix'])

st.subheader("Full Classification Report")
st.text(metrics['classification_report'])


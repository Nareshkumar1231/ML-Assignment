from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score)

ROOT = Path(__file__).resolve().parent
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "gaussian_nb.joblib",
    "Support Vector Machine": "svm.joblib",
}
DEFAULT_DATA_PATH = ROOT / "test_data.csv"

st.set_page_config(page_title="WDBC Classifier Lab", page_icon="🧪", layout="wide")
st.title("Wisconsin Breast Cancer — Model Evaluation Lab")
st.info("Target labels: malignant and benign. The CSV must contain all 30 feature columns plus a target column.")

uploaded = st.file_uploader("Upload test data (CSV)", type=["csv"])

if uploaded is None and DEFAULT_DATA_PATH.exists():
    data = pd.read_csv(DEFAULT_DATA_PATH)
    data_source = "default test_data.csv"
    st.success("Using the default `test_data.csv` from the repository.")
else:
    if uploaded is not None:
        data = pd.read_csv(uploaded)
        data_source = uploaded.name
        st.success(f"Using uploaded file: `{uploaded.name}`")
    else:
        st.warning("No dataset is available. Please upload a CSV file.")
        st.stop()

if "target" not in data.columns:
    st.error("The CSV must contain a 'target' column.")
    st.stop()

if set(data["target"].astype(str).str.lower().unique()) - {"malignant", "benign"}:
    st.error("Target values must be 'malignant' or 'benign'.")
    st.stop()

st.caption(f"Active dataset: {data_source}")
y_true = data["target"].astype(str).str.lower()
X = data.drop(columns=["target"])

model_name = st.selectbox("Choose classifier", list(MODEL_FILES.keys()))
model_path = ROOT / "model" / MODEL_FILES[model_name]
if not model_path.exists():
    st.error(f"Saved model not found: {model_path.name}. Run train_models.py and commit the model folder.")
    st.stop()
model = joblib.load(model_path)
expected = list(model.feature_names_in_) if hasattr(model, "feature_names_in_") else None
if expected and list(X.columns) != expected:
    missing = sorted(set(expected) - set(X.columns))
    extra = sorted(set(X.columns) - set(expected))
    st.error(f"Feature columns/order do not match the training data. Missing: {missing}; extra: {extra}")
    st.stop()

try:
    prediction = model.predict(X)
    probabilities = model.predict_proba(X)
    malignant_index = list(model.classes_).index("malignant")
    y_binary = (y_true == "malignant").astype(int)
    auc = roc_auc_score(y_binary, probabilities[:, malignant_index])
    values = {
        "Accuracy": accuracy_score(y_true, prediction),
        "AUC": auc,
        "Precision": precision_score(y_true, prediction, pos_label="malignant", zero_division=0),
        "Recall": recall_score(y_true, prediction, pos_label="malignant", zero_division=0),
        "F1": f1_score(y_true, prediction, pos_label="malignant", zero_division=0),
        "MCC": matthews_corrcoef(y_true, prediction),
    }
    st.subheader(f"Evaluation — {model_name}")
    cols = st.columns(6)
    for col, (label, value) in zip(cols, values.items()):
        col.metric(label, f"{value:.4f}")

    st.subheader("Confusion matrix")
    labels = ["malignant", "benign"]
    cm = confusion_matrix(y_true, prediction, labels=labels)
    cm_df = pd.DataFrame(cm, index=[f"Actual {x}" for x in labels], columns=[f"Predicted {x}" for x in labels])
    st.dataframe(cm_df, use_container_width=True)

    st.subheader("Classification report")
    report = classification_report(y_true, prediction, labels=labels, output_dict=True, zero_division=0)
    report_df = pd.DataFrame(report).transpose()
    report_df = report_df.rename(index={"accuracy": "overall_accuracy"})
    report_df = report_df.round(4)
    st.dataframe(report_df, use_container_width=True)

    output = X.copy()
    output["actual_target"] = y_true
    output["predicted_target"] = prediction
    st.download_button("Download predictions", output.to_csv(index=False), "predictions.csv", "text/csv")
except Exception as exc:
    st.error(f"Evaluation failed: {exc}")

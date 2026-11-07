"""
Train five classifiers on the UCI Wisconsin Diagnostic Breast Cancer dataset.
Run from the project root: python train_models.py
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "wdbc.csv"
MODEL_DIR = ROOT / "model"
MODEL_DIR.mkdir(exist_ok=True)

df = pd.read_csv(DATA)
X = df.drop(columns=["target"])
y = df["target"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=3000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, min_samples_split=4, random_state=42
    ),
    "KNN": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", GaussianNB())
    ]),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", SVC(kernel="rbf", probability=True, random_state=42))
    ])
}
slugs = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "KNN": "knn",
    "Gaussian Naive Bayes": "gaussian_nb",
    "Support Vector Machine": "svm"
}

rows = []
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    probabilities = model.predict_proba(X_test)
    positive_index = list(model.classes_).index("malignant")
    y_binary = (y_test == "malignant").astype(int)
    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, pred),
        "AUC": roc_auc_score(y_binary, probabilities[:, positive_index]),
        "Precision": precision_score(y_test, pred, pos_label="malignant", zero_division=0),
        "Recall": recall_score(y_test, pred, pos_label="malignant", zero_division=0),
        "F1": f1_score(y_test, pred, pos_label="malignant", zero_division=0),
        "MCC": matthews_corrcoef(y_test, pred)
    })
    joblib.dump(model, MODEL_DIR / f"{slugs[name]}.joblib")

results = pd.DataFrame(rows)
results.to_csv(ROOT / "model_metrics.csv", index=False)
print(results.to_string(index=False))
print("\\nSaved all five models in ./model/")

# Assignment 2 — Comparative Classification and Streamlit Evaluation

## a. Problem statement
Build and compare five supervised classification algorithms for the Wisconsin Diagnostic Breast Cancer (WDBC) task. The target is to classify a record as **malignant** or **benign**. All models use the same stratified 80:20 train/test split. The saved estimators are loaded by the Streamlit app to evaluate a user-uploaded labelled test CSV without retraining.

## b. Dataset description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic)
- **Instances:** 569
- **Input features:** 30 numeric measurements computed from digitized images of fine needle aspirates.
- **Target:** `target` with values `malignant` and `benign`.
- **Split:** 80% training / 20% test; `random_state=42`; stratified by target.
- **Files:** `data/wdbc.csv` is the complete dataset; `test_data.csv` is the default dataset used by the app, and `custom_test_data.csv` is an additional sample file for manual upload testing.
- **Preprocessing:** StandardScaler is included in pipelines for Logistic Regression, KNN, Gaussian Naive Bayes and SVM. Decision Tree uses raw numeric features.

## c. GitHub Repository Link
https://github.com/Nareshkumar1231/ML-Assignment/

## d. Models used and evaluation

Positive class for Precision, Recall and F1 is **malignant**. AUC uses the predicted probability for the malignant class.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9649 | 0.9960 | 0.9750 | 0.9286 | 0.9512 | 0.9245 |
| Decision Tree | 0.9211 | 0.9676 | 0.9459 | 0.8333 | 0.8861 | 0.8299 |
| KNN | 0.9561 | 0.9823 | 0.9744 | 0.9048 | 0.9383 | 0.9058 |
| Gaussian Naive Bayes | 0.9211 | 0.9891 | 0.9231 | 0.8571 | 0.8889 | 0.8292 |
| Support Vector Machine | 0.9737 | 0.9947 | 1.0000 | 0.9286 | 0.9630 | 0.9442 |

### Observations
- **Logistic Regression:** Accuracy=0.965, AUC=0.996, Precision=0.975, Recall=0.929, F1=0.951, MCC=0.925. Compare precision and recall to understand false-positive and false-negative trade-offs.
- **Decision Tree:** Accuracy=0.921, AUC=0.968, Precision=0.946, Recall=0.833, F1=0.886, MCC=0.830. Compare precision and recall to understand false-positive and false-negative trade-offs.
- **KNN:** Accuracy=0.956, AUC=0.982, Precision=0.974, Recall=0.905, F1=0.938, MCC=0.906. Compare precision and recall to understand false-positive and false-negative trade-offs.
- **Gaussian Naive Bayes:** Accuracy=0.921, AUC=0.989, Precision=0.923, Recall=0.857, F1=0.889, MCC=0.829. Compare precision and recall to understand false-positive and false-negative trade-offs.
- **Support Vector Machine:** Accuracy=0.974, AUC=0.995, Precision=1.000, Recall=0.929, F1=0.963, MCC=0.944. Compare precision and recall to understand false-positive and false-negative trade-offs.

**Overall winner on this split (by F1): Support Vector Machine.** This selection is metric-dependent; review AUC, MCC and the confusion matrix before making a final judgement.

## Streamlit application
The application accepts a labelled CSV, provides a model-selection dropdown, displays all six evaluation metrics, a confusion matrix and a classification report, and offers downloadable predictions. It loads the pre-trained `.joblib` estimators and does not retrain on uploaded data. When no file is uploaded, it automatically loads `test_data.csv` from the repository as the default dataset.

## Run on BITS Virtual Lab
1. Upload/extract this repository in your BITS Virtual Lab workspace.
2. Open a terminal in the project root.
3. Install dependencies: `python -m pip install -r requirements.txt`
4. Run training: `python train_models.py` (creates/refreshes the five model files and metrics).
5. Launch the app: `streamlit run app.py`
6. The app automatically loads `test_data.csv`; optionally upload `custom_test_data.csv` or another CSV for comparison.
7. Capture your own screenshot of the app execution inside BITS Virtual Lab for the submission.

## Deploy on Streamlit Community Cloud
1. Push the complete project to your own GitHub repository, including `app.py`, `requirements.txt`, `README.md`, `data/wdbc.csv`, `test_data.csv`, `custom_test_data.csv`, `model/` and `train_models.py`.
2. Open https://streamlit.io/cloud, sign in with GitHub and create a new app.
3. Select your repository, branch and `app.py`, then deploy.
4. Copy the live app URL into your final submission PDF.

## Repository structure
```
aiml_assignment2/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
├── custom_test_data.csv
├── model_metrics.csv
├── data/
│   └── wdbc.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── gaussian_nb.joblib
    └── svm.joblib
```

## Final submission links and evidence
- GitHub repository: [https://github.com/Nareshkumar1231/ML-Assignment/](https://github.com/Nareshkumar1231/ML-Assignment/)
- Live Streamlit app: [https://nareshkumar1231-ml-assignment-app-e8bjgq.streamlit.app/](https://nareshkumar1231-ml-assignment-app-e8bjgq.streamlit.app/)
- BITS Virtual Lab screenshot: **[INSERT YOUR OWN SCREENSHOT]**

Do not submit until you have personally run the code in BITS Virtual Lab, checked the deployed app, and inserted the required screenshot.

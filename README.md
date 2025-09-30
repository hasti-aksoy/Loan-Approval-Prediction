# Loan Approval Prediction 🚀  

This repository contains a complete machine learning pipeline for **loan default prediction**.  
The workflow covers **data preprocessing, feature engineering, exploratory data analysis (EDA), model training, evaluation, and prediction**.  

---

## 📂 Project Structure  

LOAN APPROVAL PREDICTION/
│
├── data/ # Raw and cleaned datasets
│ ├── train.csv
│ ├── test.csv
│ ├── train_cleaned.csv
│ ├── test_cleaned.csv
│ └── sample_submission.csv
│
├── models/ # Trained model artifacts
│ ├── logreg.joblib
│ ├── rf.joblib
│ └── xgb.joblib
│
├── notebooks/ # Jupyter notebooks
│ ├── 01_data_cleaning.ipynb
│ └── 02_eda.ipynb
│
├── reports/ # Reports and visualizations
│ └── figures/
│ ├── univariate_distributions.png
│ ├── feature_vs_target_dti.png
│ ├── feature_vs_target_loan_int_rate.png
│ ├── feature_vs_target_loan_percent_income.png
│ └── ... (other plots)
│
├── src/ # Source code
│ ├── evaluate.py # Evaluation utilities
│ ├── predict.py # Inference script
│ ├── train_logistic.py# Logistic Regression training
│ ├── train_rf.py # Random Forest training
│ └── train_xgb.py # XGBoost training
│
├── submissions/ # Submission files
│ └── submission.csv
│
└── README.md # Project documentation

yaml
Copy code

---

## ⚙️ Installation  

Clone the repo and install dependencies:  

```bash
git clone https://github.com/hasti-aksoy/loan-approval-prediction.git
cd loan-approval-prediction

python -m venv .venv
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

pip install -r requirements.txt
📊 Dataset
Train set: 58,645 rows, 13 original features.

Test set: Provided for predictions.

Target column: loan_status (0 = No Default, 1 = Default).

Cleaned versions (train_cleaned.csv, test_cleaned.csv) are generated during preprocessing.

🚀 Usage
1. Data Cleaning and EDA
Run Jupyter notebooks:

notebooks/01_data_cleaning.ipynb

notebooks/02_eda.ipynb

2. Train Models
Each training script outputs a .joblib model under models/:

bash
Copy code
# Logistic Regression
python src/train_logistic.py --input data/train_cleaned.csv --model-out models/logreg.joblib

# Random Forest
python src/train_rf.py --input data/train_cleaned.csv --model-out models/rf.joblib

# XGBoost
python src/train_xgb.py --input data/train_cleaned.csv --model-out models/xgb.joblib
3. Evaluate Models
Evaluation includes ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1, plus confusion matrix and classification report.

4. Predict on Test Set
bash
Copy code
python src/predict.py --model models/xgb.joblib --input data/test_cleaned.csv --output submissions/submission.csv
📈 Results
Model	ROC-AUC	PR-AUC	Accuracy	F1	Precision	Recall
Logistic Regression	0.9000	0.6816	0.8183	0.5647	0.4285	0.8275
Random Forest	0.9353	0.8457	0.9498	0.7974	0.9369	0.6940
XGBoost	0.9534	0.8711	0.9513	0.8074	0.9243	0.7168

XGBoost achieved the best overall performance.

Logistic Regression had the highest recall (best at catching defaulters).

Random Forest provided a balanced trade-off but slightly below XGBoost.

📌 Future Work
Advanced categorical encodings (embeddings).

Ensemble stacking and hybrid models.

Cost-sensitive learning for imbalanced data.

Explainability tools (e.g., SHAP values).

Temporal validation with out-of-time datasets.

📝 License
This project is licensed under the MIT License.
# Loan Approval Prediction

End-to-end pipeline for loan default prediction: data cleaning, feature engineering, EDA, model training, evaluation, and prediction.

## Project Structure

```text
Loan-Approval-Prediction/
├─ data/                     # Raw & cleaned datasets
│  ├─ train.csv
│  ├─ test.csv
│  ├─ train_cleaned.csv
│  ├─ test_cleaned.csv
│  └─ sample_submission.csv
├─ notebooks/                # Jupyter notebooks
│  ├─ 01_data_cleaning.ipynb
│  └─ 02_eda.ipynb
├─ reports/
│  └─ figures/               # Visualizations
│     ├─ univariate_distributions.png
│     ├─ feature_vs_target_dti.png
│     ├─ feature_vs_target_loan_int_rate.png
│     ├─ feature_vs_target_loan_percent_income.png
│     └─ ... (other plots)
├─ src/                      # Source code
│  ├─ evaluate.py            # Metrics & reports
│  ├─ predict.py             # Inference script
│  ├─ train_logistic.py      # Logistic Regression
│  ├─ train_rf.py            # Random Forest
│  └─ train_xgb.py           # XGBoost / fallback GBM
├─ submissions/
│  └─ submission.csv
├─ models/                   # Trained artifacts (kept locally, not in git)
├─ .gitignore
├─ README.md
└─ requirements.txt (optional)
```

## Installation

Clone and set up environment:

```bash
git clone https://github.com/hasti-aksoy/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction

python -m venv .venv
# Linux/Mac
source .venv/bin/activate
# Windows
.venv\Scripts\activate

# If requirements.txt exists
pip install -r requirements.txt
# Or install minimal deps
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Dataset

- Train: 58,645 rows, 13 original features
- Test: provided for predictions
- Target: `loan_status` (0 = No Default, 1 = Default)
- Cleaned: `data/train_cleaned.csv`, `data/test_cleaned.csv` produced by the cleaning notebook

## Usage

1) Data Cleaning and EDA (Jupyter):

```text
notebooks/01_data_cleaning.ipynb
notebooks/02_eda.ipynb
```

2) Train models (artifacts saved under `models/` locally):

```bash
# Logistic Regression
python src/train_logistic.py --input data/train_cleaned.csv --model-out models/logreg.joblib

# Random Forest
python src/train_rf.py --input data/train_cleaned.csv --model-out models/rf.joblib

# XGBoost (or GBM fallback if xgboost not installed)
python src/train_xgb.py --input data/train_cleaned.csv --model-out models/xgb.joblib
```

3) Evaluate

- Metrics: ROC-AUC, PR-AUC, Accuracy, Precision, Recall, F1
- `print_report` prints confusion matrix and classification report

4) Predict on test set

```bash
python src/predict.py --model models/xgb.joblib --input data/test_cleaned.csv --output submissions/submission.csv
```

## Results (example)

| Model                | ROC-AUC | PR-AUC | Accuracy |   F1  | Precision | Recall |
|----------------------|:-------:|:------:|:--------:|:-----:|:---------:|:------:|
| Logistic Regression  |  0.9000 | 0.6816 |  0.8183  | 0.5647|   0.4285  | 0.8275 |
| Random Forest        |  0.9353 | 0.8457 |  0.9498  | 0.7974|   0.9369  | 0.6940 |
| XGBoost              |  0.9534 | 0.8711 |  0.9513  | 0.8074|   0.9243  | 0.7168 |

- XGBoost achieved the best overall performance.
- Logistic Regression had the highest recall (best at catching defaulters).
- Random Forest offered a strong balance, slightly below XGBoost.

## Future Work

- Advanced categorical encodings (e.g., target/embedding)
- Ensemble stacking and hybrid models
- Cost-sensitive learning for imbalance
- Explainability (e.g., SHAP)
- Temporal validation (out-of-time split)

## License

MIT License


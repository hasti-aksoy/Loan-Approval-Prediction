# src/train_logistic.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

try:
    from .evaluate import compute_metrics, print_report
except Exception:
    from evaluate import compute_metrics, print_report

def load_xy_from_csv(csv_path: Path, target_col: str = "loan_status"):
    df = pd.read_csv(csv_path)
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in {csv_path}")

    # Drop obvious identifiers
    drop_id = [c for c in ("id", "loan_id") if c in df.columns]

    # Keep numeric columns only for baseline logistic regression
    obj_cols = [c for c in df.columns if df[c].dtype == "object"]
    keep_cols = [c for c in df.columns if c not in obj_cols and c not in drop_id and c != target_col]

    X = df[keep_cols].copy()
    y = df[target_col].astype(int).values

    return X, y, keep_cols

def build_pipeline(numeric_cols):
    
    pre = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler(with_mean=True, with_std=True)),
                    ]
                ),
                numeric_cols,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    clf = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",  
        solver="lbfgs",
        n_jobs=None
    )

    pipe = Pipeline(steps=[
        ("preprocess", pre),
        ("clf", clf)
    ])
    return pipe

def main(
    input_csv: Path,
    target_col: str,
    test_size: float,
    random_state: int,
    model_out: Path,
    use_external_splits: bool,
    train_csv: Path,
    valid_csv: Path
):
    if use_external_splits:
        
        X_train, y_train, cols = load_xy_from_csv(train_csv, target_col)
        X_valid, y_valid, _    = load_xy_from_csv(valid_csv, target_col)
        numeric_cols = list(X_train.columns)
    else:
        # Fallback between cleaned and raw if provided path missing
        if not input_csv.exists():
            alt = Path("data/train_cleaned.csv") if input_csv.name != "train_cleaned.csv" else Path("data/train.csv")
            if alt.exists():
                print(f"Input not found at {input_csv}. Falling back to {alt}.")
                input_csv = alt
            else:
                raise FileNotFoundError(f"Input CSV not found: {input_csv}")

        X, y, numeric_cols = load_xy_from_csv(input_csv, target_col)
        X_train, X_valid, y_train, y_valid = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

    pipe = build_pipeline(numeric_cols)
    pipe.fit(X_train, y_train)

    
    y_proba = pipe.predict_proba(X_valid)[:, 1]
    y_pred  = (y_proba >= 0.5).astype(int)

   
    metrics = compute_metrics(y_valid, y_proba, y_pred)
    print("=== Logistic Regression (validation) ===")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}" if isinstance(v, float) and not np.isnan(v) else f"{k}: {v}")

    print()
    print_report(y_valid, y_pred)

  
    model_out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"model": pipe, "numeric_cols": numeric_cols, "target_col": target_col},
        model_out
    )
    print(f"\nSaved model to: {model_out.resolve()}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=str, default="data/train_cleaned.csv",
                   help="Path to a single CSV that contains all rows (will split internally).")
    p.add_argument("--target", type=str, default="loan_status")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model-out", type=str, default="models/logreg.joblib")

    
    p.add_argument("--use-external-splits", action="store_true",
                   help="Use --train-csv and --valid-csv instead of internal split.")
    p.add_argument("--train-csv", type=str, default="data/train.csv")
    p.add_argument("--valid-csv", type=str, default="data/valid.csv")

    args = p.parse_args()

    main(
        input_csv=Path(args.input),
        target_col=args.target,
        test_size=args.test_size,
        random_state=args.seed,
        model_out=Path(args.model_out),
        use_external_splits=args.use_external_splits,
        train_csv=Path(args.train_csv),
        valid_csv=Path(args.valid_csv),
    )

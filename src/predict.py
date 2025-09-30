# src/predict.py
import argparse
from pathlib import Path
import joblib
import pandas as pd
import numpy as np


def _detect_id_column(df: pd.DataFrame) -> str | None:
    for c in ("id", "loan_id"):
        if c in df.columns:
            return c
    return None


def _prepare_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    X = df.copy()
    # Ensure all expected features exist
    for c in feature_cols:
        if c not in X.columns:
            X[c] = np.nan
    # Keep only expected features in order
    X = X[feature_cols]
    # Coerce to numeric
    for c in X.columns:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    return X


def main(model_path: Path, input_csv: Path, sample_sub_path: Path | None, output_csv: Path):
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    bundle = joblib.load(model_path)
    pipe = bundle["model"]
    feature_cols = bundle.get("numeric_cols") or bundle.get("feature_cols")
    if feature_cols is None:
        raise KeyError("Model bundle missing 'numeric_cols' or 'feature_cols'.")

    if not input_csv.exists():
        alt = Path("data/test_cleaned.csv") if input_csv.name != "test_cleaned.csv" else Path("data/test.csv")
        if alt.exists():
            print(f"Input not found at {input_csv}. Falling back to {alt}.")
            input_csv = alt
        else:
            raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    test_df = pd.read_csv(input_csv)
    test_df.columns = test_df.columns.str.strip()
    id_col = _detect_id_column(test_df)

    X = _prepare_features(test_df, feature_cols)

    # Predict probabilities if supported, else decision function
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X)[:, 1]
    elif hasattr(pipe, "decision_function"):
        scores = pipe.decision_function(X)
        # Min-max to [0,1] as a fallback probability-like score
        smin, smax = float(np.min(scores)), float(np.max(scores))
        y_proba = (scores - smin) / (smax - smin + 1e-9)
    else:
        # Fall back to discrete predictions
        preds = pipe.predict(X)
        y_proba = preds.astype(float)

    # Binary class: threshold at 0.5
    y_pred = (y_proba >= 0.5).astype(int)

    # Build submission
    if sample_sub_path and Path(sample_sub_path).exists():
        sub = pd.read_csv(sample_sub_path)
        sub.columns = sub.columns.str.strip()
        sub_id_col = _detect_id_column(sub)
        if sub_id_col is None:
            raise ValueError("Sample submission is missing an id column ('id' or 'loan_id').")
        # Detect target column name from sample
        target_col = [c for c in sub.columns if c != sub_id_col][0]

        if id_col is not None:
            # Map predictions to sample ids via merge on id
            out = sub[[sub_id_col]].merge(
                test_df[[id_col]].assign(**{target_col: y_pred}),
                left_on=sub_id_col,
                right_on=id_col,
                how="left",
            )[[sub_id_col, target_col]]
            # If any missing (ids mismatch), fall back to order alignment when lengths match
            if out[target_col].isna().any() and len(sub) == len(y_pred):
                out = pd.DataFrame({sub_id_col: sub[sub_id_col], target_col: y_pred})
        else:
            # No id in test: assume same order as sample submission
            if len(sub) != len(test_df):
                raise ValueError(
                    f"Test has no id column and its length ({len(test_df)}) does not match sample submission length ({len(sub)})."
                )
            out = pd.DataFrame({sub_id_col: sub[sub_id_col], target_col: y_pred})
    else:
        # No sample submission provided. Use available id if present; else create a row index id.
        target_col = "loan_status"
        if id_col is not None:
            out = pd.DataFrame({id_col: test_df[id_col], target_col: y_pred})
        else:
            out = pd.DataFrame({"row_id": np.arange(len(test_df)), target_col: y_pred})

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv, index=False)
    print(f"Saved predictions to: {output_csv.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="models/xgb.joblib")
    p.add_argument("--input", type=str, default="data/test_cleaned.csv")
    p.add_argument("--sample-sub", type=str, default="data/sample_submission.csv")
    p.add_argument("--output", type=str, default="submissions/submission.csv")
    args = p.parse_args()

    main(
        model_path=Path(args.model),
        input_csv=Path(args.input),
        sample_sub_path=Path(args.sample_sub) if args.sample_sub else None,
        output_csv=Path(args.output),
    )

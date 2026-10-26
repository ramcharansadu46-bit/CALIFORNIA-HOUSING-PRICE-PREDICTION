"""Predict median house value for new districts with the saved pipeline.

Usage (from the project root):
    python -m src.predict data/processed/test.csv            # prints first 10 predictions
Input CSV must contain the 9 predictor columns (the target column is ignored if present).
"""
import sys
import joblib
import pandas as pd
from .config import MODEL_DIR, TARGET


def main(path):
    model = joblib.load(MODEL_DIR / "best_model.joblib")
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET], errors="ignore")
    out = df.copy()
    out["predicted_median_house_value"] = model.predict(X)
    print(out.head(10).to_string())
    return out


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/processed/test.csv")

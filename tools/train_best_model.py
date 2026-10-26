"""Fit the final best XGBoost pipeline on training data and serialize it for production."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from src.config import MODEL_DIR, RESULT_DIR, TARGET
from src.evaluate import regression_metrics
from src.features import build_preprocessor


def train_and_save():
    print("Loading data...")
    train = pd.read_csv("data/processed/train.csv")
    test = pd.read_csv("data/processed/test.csv")

    X_train = train.drop(columns=[TARGET])
    y_train = train[TARGET]
    X_test = test.drop(columns=[TARGET])
    y_test = test[TARGET]

    print("Building and fitting pipeline (KNN imputer + feature engineering + XGBoost)...")
    pipe = Pipeline([
        ("prep", build_preprocessor(impute="knn", ratios=True, spatial=True)),
        ("model", XGBRegressor(
            n_estimators=800,
            max_depth=8,
            learning_rate=0.03,
            subsample=0.9,
            colsample_bytree=0.7,
            min_child_weight=1,
            tree_method="hist",
            n_jobs=1,
            random_state=42
        ))
    ])

    pipe.fit(X_train, y_train)

    print("Saving model to models/best_model.joblib...")
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipe, MODEL_DIR / "best_model.joblib", compress=3)

    print("Evaluating on test set...")
    pred = pipe.predict(X_test)
    metrics = regression_metrics(y_test, pred)

    pd.DataFrame({"actual": y_test.values, "predicted": pred, "residual": y_test.values - pred}).to_csv(
        RESULT_DIR / "test_predictions.csv", index=False
    )

    card = {
        "model": "XGBoost",
        "hyperparameters": {
            "subsample": 0.9,
            "n_estimators": 800,
            "min_child_weight": 1,
            "max_depth": 8,
            "learning_rate": 0.03,
            "colsample_bytree": 0.7
        },
        "imputer": "knn",
        "feature_set": "F3 +ratios+spatial",
        "trained_on_rows": len(X_train),
        "test_metrics": metrics,
        "target": "median_house_value",
        "input_columns": list(X_train.columns),
        "environment": {
            "python": "3.12.13",
            "pandas": "3.0.2",
            "numpy": "2.4.4",
            "scikit-learn": "1.8.0",
            "xgboost": "3.4.1",
            "random_state": 42,
            "quick_mode": False
        },
        "load_example": "import joblib, pandas as pd; m = joblib.load('models/best_model.joblib'); m.predict(pd.read_csv('data/processed/test.csv').drop(columns='median_house_value').head())",
        "note": "Pipeline pickles classes from src/ -> run from the project root so `src` is importable."
    }
    with open(MODEL_DIR / "model_card.json", "w") as f:
        json.dump(card, f, indent=2)

    print(f"Test Metrics: {json.dumps(metrics, indent=2)}")
    print("Done!")


if __name__ == "__main__":
    train_and_save()

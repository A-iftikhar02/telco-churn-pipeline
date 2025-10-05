
import json
from pathlib import Path

import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
DEFAULT_DATASET = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SAMPLE_DATASET = DATA_DIR / "sample_telco_churn.csv"


def load_any_dataset() -> pd.DataFrame:
    if DEFAULT_DATASET.exists():
        return pd.read_csv(DEFAULT_DATASET)
    if SAMPLE_DATASET.exists():
        return pd.read_csv(SAMPLE_DATASET)
    raise FileNotFoundError("No dataset to construct a sample input")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Align with training: coerce TotalCharges to numeric and drop identifier
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])  # identifier not used
    return df


def get_expected_columns(model) -> list:
    cols = getattr(model, "raw_feature_names_", None)
    if cols:
        return list(cols)
    # Fallback: infer from a local dataset if available
    for p in [DEFAULT_DATASET, SAMPLE_DATASET]:
        if p.exists():
            df = pd.read_csv(p)
            df = preprocess_dataframe(df)
            if "Churn" in df.columns:
                df = df.drop(columns=["Churn"])  # target
            return list(df.columns)
    return []


def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found at {MODEL_PATH}. Train the model first (run 'python train_pipeline.py').")

    model = joblib.load(MODEL_PATH)

    df = load_any_dataset()
    df = preprocess_dataframe(df)
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])  # target not used for inference

    # Build a single-row sample aligned with model's expected columns
    expected_cols = get_expected_columns(model)
    X = df.head(1)
    if expected_cols:
        # Reindex ensures presence/order; values filled will be imputed by pipeline later if needed
        X = X.reindex(columns=expected_cols)

    pred = model.predict(X)[0]
    out = {"prediction": int(pred)}

    # Try predict_proba if available
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(X)[:, 1][0])
        except Exception:
            prob = None
    out["probability_positive"] = prob

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

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


def main():
    model = joblib.load(MODEL_PATH)
    df = load_any_dataset()
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])  # identifier not used
    if "Churn" in df.columns:
        X = df.drop(columns=["Churn"]).head(1)
    else:
        X = df.head(1)

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

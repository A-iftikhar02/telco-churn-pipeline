import argparse
import json
import os
from pathlib import Path
from typing import List, Optional

import joblib
import numpy as np
import pandas as pd
import requests
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

DATA_DIR = Path(__file__).parent / "data"
MODELS_DIR = Path(__file__).parent / "models"
DEFAULT_DATASET = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SAMPLE_DATASET = DATA_DIR / "sample_telco_churn.csv"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
METRICS_PATH = MODELS_DIR / "metrics.json"

DOWNLOAD_URLS = [
    # Blastchar mirror (may be flaky)
    "https://raw.githubusercontent.com/blastchar/telco-customer-churn/master/WA_Fn-UseC_-Telco-Customer-Churn.csv",
    # IBM sample repo mirror
    "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/WA_Fn-UseC_-Telco-Customer-Churn.csv",
]


def ensure_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def try_download_dataset(dest: Path) -> bool:
    for url in DOWNLOAD_URLS:
        try:
            resp = requests.get(url, timeout=20)
            if resp.status_code == 200 and len(resp.content) > 0:
                dest.write_bytes(resp.content)
                return True
        except Exception:
            pass
    return False


def load_dataset(use_sample: bool = False) -> pd.DataFrame:
    ensure_dirs()

    if use_sample and SAMPLE_DATASET.exists():
        return pd.read_csv(SAMPLE_DATASET)

    if DEFAULT_DATASET.exists():
        return pd.read_csv(DEFAULT_DATASET)

    # Attempt download
    ok = try_download_dataset(DEFAULT_DATASET)
    if ok:
        return pd.read_csv(DEFAULT_DATASET)

    # Fallback to bundled sample
    if SAMPLE_DATASET.exists():
        return pd.read_csv(SAMPLE_DATASET)

    raise FileNotFoundError("No dataset available and download failed. Ensure sample exists.")


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    # Standard fixes for Telco dataset
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    # Drop customerID if present (identifier)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])
    return df


def identify_feature_columns(df: pd.DataFrame, target_col: str) -> (List[str], List[str]):
    feature_df = df.drop(columns=[target_col])
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_df.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def build_pipeline(numeric_cols: List[str], categorical_cols: List[str]) -> Pipeline:
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    base_clf = LogisticRegression(max_iter=1000, solver="lbfgs")

    pipe = Pipeline(steps=[
        ("preprocess", preprocessor),
        ("clf", base_clf)
    ])

    # Reasonable, small grids to keep training snappy
    param_grid = [
        {
            "clf": [LogisticRegression(max_iter=1000, solver="lbfgs")],
            "clf__C": [0.5, 1.0, 2.0]
        },
        {
            "clf": [RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)],
            "clf__max_depth": [None, 8, 16],
            "clf__min_samples_split": [2, 10]
        }
    ]

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        n_jobs=-1,
        cv=3,
        refit=True,
        verbose=0,
    )

    return grid


def compute_metrics(model, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
    metrics = {}
    y_pred = model.predict(X_test)
    metrics["accuracy"] = float(accuracy_score(y_test, y_pred))

    auc: Optional[float] = None
    try:
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(X_test)
            auc = roc_auc_score(y_test, y_score)
    except Exception:
        auc = None
    if auc is not None:
        metrics["roc_auc"] = float(auc)
    else:
        metrics["roc_auc"] = None
    return metrics


def make_json_safe(obj):
    if obj is None:
        return None
    if isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): make_json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [make_json_safe(v) for v in obj]
    return repr(obj)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use_sample", type=int, default=0, help="Use bundled sample dataset if set to 1")
    args = parser.parse_args()

    df_raw = load_dataset(use_sample=bool(args.use_sample))
    df = preprocess_dataframe(df_raw.copy())

    if "Churn" not in df.columns:
        raise ValueError("Expected target column 'Churn' in dataset")

    # Target to binary
    y = df["Churn"].astype(str).str.strip().map({"Yes": 1, "No": 0})
    if y.isna().any():
        # If sample uses 1/0 already
        y = pd.to_numeric(df["Churn"], errors="coerce")
    if y.isna().any():
        raise ValueError("Could not parse target column 'Churn'")

    X = df.drop(columns=["Churn"])

    numeric_cols, categorical_cols = identify_feature_columns(df, target_col="Churn")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = build_pipeline(numeric_cols, categorical_cols)
    model.fit(X_train, y_train)

    best_estimator = model.best_estimator_ if hasattr(model, "best_estimator_") else model

    # Attach raw feature names for serving apps
    try:
        setattr(best_estimator, "raw_feature_names_", list(X.columns))
    except Exception:
        pass

    metrics = compute_metrics(best_estimator, X_test, y_test)

    ensure_dirs()
    joblib.dump(best_estimator, MODEL_PATH)

    out = {
        "best_params": make_json_safe(getattr(model, "best_params_", None)),
        "best_score_cv": float(getattr(model, "best_score_", np.nan)) if hasattr(model, "best_score_") else None,
        "metrics": metrics,
        "model_path": str(MODEL_PATH),
        "dataset": str(DEFAULT_DATASET if DEFAULT_DATASET.exists() and not args.use_sample else SAMPLE_DATASET),
    }
    METRICS_PATH.write_text(json.dumps(out, indent=2))

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

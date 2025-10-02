import json
import socket
from pathlib import Path
from typing import Dict, List

import gradio as gr
import joblib
import pandas as pd

MODELS_DIR = Path(__file__).parent / "models"
DATA_DIR = Path(__file__).parent / "data"
MODEL_PATH = MODELS_DIR / "churn_pipeline.joblib"
DEFAULT_DATASET = DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
SAMPLE_DATASET = DATA_DIR / "sample_telco_churn.csv"


def get_expected_columns(model) -> List[str]:
    cols = getattr(model, "raw_feature_names_", None)
    if cols:
        return list(cols)
    # Fallback: try to infer from an available dataset
    for p in [DEFAULT_DATASET, SAMPLE_DATASET]:
        if p.exists():
            df = pd.read_csv(p)
            if "customerID" in df.columns:
                df = df.drop(columns=["customerID"])  # identifier not used
            if "Churn" in df.columns:
                df = df.drop(columns=["Churn"])  # target
            return list(df.columns)
    return []


def predict_from_form(*values):
    model = joblib.load(MODEL_PATH)
    expected = get_expected_columns(model)
    # values come in same order as components list we construct
    data = {col: val for col, val in zip(expected, values)}
    df = pd.DataFrame([data])
    pred = int(model.predict(df)[0])
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(df)[:, 1][0])
        except Exception:
            prob = None
    return json.dumps({"prediction": pred, "probability_positive": prob}, indent=2)


def predict_from_csv(file: gr.File | None):
    if file is None:
        return "Please upload a CSV file."
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(file.name)
    if "customerID" in df.columns:
        df = df.drop(columns=["customerID"])  # identifier not used
    if "Churn" in df.columns:
        df = df.drop(columns=["Churn"])  # target
    pred = model.predict(df)
    out = []
    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = model.predict_proba(df)[:, 1]
        except Exception:
            prob = None
    for i, p in enumerate(pred):
        item = {"row": int(i), "prediction": int(p)}
        if prob is not None:
            item["probability_positive"] = float(prob[i])
        out.append(item)
    return json.dumps(out, indent=2)


def find_open_port(start_port: int = 7860, max_trials: int = 10) -> int:
    port = start_port
    for _ in range(max_trials):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return port
            except OSError:
                port += 1
    return port


def build_demo():
    model = joblib.load(MODEL_PATH)
    expected = get_expected_columns(model)

    with gr.Blocks(title="Telco Churn Prediction") as demo:
        gr.Markdown("""
        ### Telco Churn Prediction
        - Upload a CSV with the raw columns (excluding `Churn`) or
        - Fill the form (columns detected from the model).
        """)
        with gr.Tab("Form"):
            form_inputs = []
            for col in expected:
                form_inputs.append(gr.Textbox(label=col))
            form_btn = gr.Button("Predict")
            form_output = gr.JSON(label="Prediction")
            form_btn.click(predict_from_form, inputs=form_inputs, outputs=form_output)
        with gr.Tab("CSV Upload"):
            file_in = gr.File(label="CSV with raw features")
            csv_btn = gr.Button("Predict CSV")
            csv_out = gr.JSON(label="Predictions")
            csv_btn.click(predict_from_csv, inputs=[file_in], outputs=csv_out)
    return demo


if __name__ == "__main__":
    port = find_open_port(7860)
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=port)

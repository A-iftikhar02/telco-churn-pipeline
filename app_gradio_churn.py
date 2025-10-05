# app_gradio_churn.py — FULL FEATURE VERSION (FIXED)
import json, os
import gradio as gr
import pandas as pd
from joblib import load

MODEL_PATH = "./models/churn_pipeline.joblib"
if not os.path.exists(MODEL_PATH):
    raise SystemExit("Model not found. Train it first:  python train_pipeline.py --use_sample 1")

pipe = load(MODEL_PATH)

# Spec for all features used during training (everything except customerID & Churn)
INPUT_SPECS = [
    ("gender", ["Male","Female"], "Male"),
    ("SeniorCitizen", [0,1], 0),
    ("Partner", ["Yes","No"], "No"),
    ("Dependents", ["Yes","No"], "No"),
    ("tenure", "number", 12),
    ("PhoneService", ["Yes","No"], "Yes"),
    ("MultipleLines", ["Yes","No","No phone service"], "No"),
    ("InternetService", ["DSL","Fiber optic","No"], "DSL"),
    ("OnlineSecurity", ["Yes","No","No internet service"], "No"),
    ("OnlineBackup", ["Yes","No","No internet service"], "No"),
    ("DeviceProtection", ["Yes","No","No internet service"], "No"),
    ("TechSupport", ["Yes","No","No internet service"], "No"),
    ("StreamingTV", ["Yes","No","No internet service"], "No"),
    ("StreamingMovies", ["Yes","No","No internet service"], "No"),
    ("Contract", ["Month-to-month","One year","Two year"], "Month-to-month"),
    ("PaperlessBilling", ["Yes","No"], "Yes"),
    ("PaymentMethod", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"], "Electronic check"),
    ("MonthlyCharges", "number", 70.0),
    ("TotalCharges", "number", 1200.0),
]

NUMERIC_COLS = {"SeniorCitizen","tenure","MonthlyCharges","TotalCharges"}

def predict(*values):
    try:
        names = [n for n, *_ in INPUT_SPECS]
        record = dict(zip(names, values))
        df = pd.DataFrame([record])

        # Ensure numeric columns are numeric
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Predict
        try:
            proba = float(pipe.predict_proba(df)[0, 1])
        except Exception:
            proba = 0.0
        y = pipe.predict(df)[0]
        label = "Yes" if (y == 1 or y == "Yes") else "No"

        return (label, proba, json.dumps(record, indent=2))
    except Exception as e:
        return ("Error", 0.0, json.dumps({"error": str(e)}, indent=2))

def _apply_preset(preset):
    """Return a list of values (in INPUT_SPECS order) for the preset."""
    # Defaults (start from current defaults)
    base = {n: d for n, opts, d in INPUT_SPECS}
    if preset == "low":
        base.update(dict(
            gender="Female", SeniorCitizen=0, Partner="Yes", Dependents="Yes",
            tenure=48, PhoneService="Yes", MultipleLines="No",
            InternetService="DSL",
            OnlineSecurity="Yes", OnlineBackup="Yes", DeviceProtection="Yes", TechSupport="Yes",
            StreamingTV="Yes", StreamingMovies="Yes",
            Contract="Two year", PaperlessBilling="No", PaymentMethod="Bank transfer (automatic)",
            MonthlyCharges=45.0, TotalCharges=2100.0,
        ))
    elif preset == "med":
        base.update(dict(
            gender="Male", SeniorCitizen=0, Partner="No", Dependents="No",
            tenure=10, PhoneService="Yes", MultipleLines="Yes",
            InternetService="DSL",
            OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
            StreamingTV="No", StreamingMovies="No",
            Contract="One year", PaperlessBilling="Yes", PaymentMethod="Credit card (automatic)",
            MonthlyCharges=70.0, TotalCharges=700.0,
        ))
    elif preset == "high":
        base.update(dict(
            gender="Male", SeniorCitizen=1, Partner="No", Dependents="No",
            tenure=2, PhoneService="Yes", MultipleLines="No",
            InternetService="Fiber optic",
            OnlineSecurity="No", OnlineBackup="No", DeviceProtection="No", TechSupport="No",
            StreamingTV="No", StreamingMovies="No",
            Contract="Month-to-month", PaperlessBilling="Yes", PaymentMethod="Electronic check",
            MonthlyCharges=105.0, TotalCharges=210.0,
        ))
    return [base[n] for n, *_ in INPUT_SPECS]

with gr.Blocks(title="Telco Churn Predictor") as demo:
    gr.Markdown("# Telco Churn Predictor\nProvide customer attributes to get a churn prediction.")

    # Build inputs INSIDE the Blocks context so they render
    inputs = []
    with gr.Row():
        for i, (name, opts, default) in enumerate(INPUT_SPECS):
            if opts == "number":
                comp = gr.Number(value=default, label=name)
            else:
                comp = gr.Dropdown(opts, value=default, label=name)
            inputs.append(comp)

    with gr.Row():
        btn_low = gr.Button("Prefill: Low Risk")
        btn_med = gr.Button("Prefill: Medium Risk")
        btn_high = gr.Button("Prefill: High Risk")
        btn_pred = gr.Button("Predict", variant="primary")

    out1 = gr.Label(label="Churn? (Yes/No)")
    out2 = gr.Number(label="Churn Probability")
    out3 = gr.Textbox(label="Submitted Record (JSON)", lines=10)

    # Prefill handlers
    btn_low.click(lambda: _apply_preset("low"), outputs=inputs)
    btn_med.click(lambda: _apply_preset("med"), outputs=inputs)
    btn_high.click(lambda: _apply_preset("high"), outputs=inputs)

    # Predict handler
    btn_pred.click(fn=predict, inputs=inputs, outputs=[out1, out2, out3])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860, show_error=True, debug=True)

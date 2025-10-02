# Telco Churn Pipeline

End-to-end ML pipeline to train and serve a churn model on the Telco Customer Churn dataset. Includes automatic dataset download, training with preprocessing + GridSearchCV, model export, quick smoke test, and a Gradio app for interactive predictions.

## Project Layout
```
telco-churn-pipeline/
 ├─ train_pipeline.py
 ├─ app_gradio_churn.py
 ├─ quickcheck.py
 ├─ requirements.txt
 ├─ README.md
 ├─ .gitignore
 ├─ data/
 │   ├─ WA_Fn-UseC_-Telco-Customer-Churn.csv   # (auto-download if missing)
 │   └─ sample_telco_churn.csv                 # (bundled synthetic)
 └─ models/
     ├─ churn_pipeline.joblib                  # (output)
     └─ metrics.json                           # (output)
```

## Setup
```bash
python -V
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Train
Try full dataset; if it fails, use sample:

**Windows PowerShell:**
```powershell
try { python train_pipeline.py } catch { python train_pipeline.py --use_sample 1 }
```

**Linux/Mac Bash:**
```bash
python train_pipeline.py || python train_pipeline.py --use_sample 1
```

## Verify
```bash
python quickcheck.py
```

## Run App
```bash
python app_gradio_churn.py
```

Notes:
- If the real dataset is missing, the script auto-downloads from known URLs. If that fails, it falls back to the bundled `sample_telco_churn.csv`.
- The script coerces `TotalCharges` to numeric with `errors="coerce"` and imputes missing values.
- If AUC cannot be computed (no `predict_proba`), it's skipped gracefully.
- If Gradio port is busy, the app increments the port.

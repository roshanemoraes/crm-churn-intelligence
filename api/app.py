"""
CRM Churn Intelligence — FastAPI Prediction Endpoint
-----------------------------------------------------
Deployment simulation: loads the trained pipeline and exposes POST /predict.

Model loading strategy (checked at startup):
  1. If S3_BUCKET_NAME env var is set → download artifacts from S3
  2. Otherwise                         → load from local models/ directory

Run locally:
    uvicorn api.app:app --reload

On EC2 (with S3_BUCKET_NAME set):
    S3_BUCKET_NAME=crm-churn-intelligence uvicorn api.app:app --host 0.0.0.0 --port 8000
"""

import json
import os
import tempfile
from typing import Optional

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from src.data_processing.features import apply_feature_engineering
from src.data_processing.preprocess import apply_manual_encoding


# ── Startup: load model artifacts (S3 or local)

BASE_DIR          = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_MODEL_PATH  = os.path.join(BASE_DIR, "models", "final_model.pkl")
LOCAL_COLS_PATH   = os.path.join(BASE_DIR, "models", "feature_columns.json")
STATIC_DIR        = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

S3_BUCKET         = os.getenv("S3_BUCKET_NAME", "")
MODEL_S3_KEY      = os.getenv("MODEL_S3_KEY",        "models/final_model.pkl")
FEATURE_COLS_S3_KEY = os.getenv("FEATURE_COLS_S3_KEY", "models/feature_columns.json")
AWS_REGION        = os.getenv("AWS_REGION", "eu-west-1")


def _load_from_s3(bucket: str, key: str, suffix: str):
    """Download a single S3 object to a temp file and return the local path."""
    import boto3
    s3 = boto3.client("s3", region_name=AWS_REGION)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    print(f"Downloading s3://{bucket}/{key} ...")
    s3.download_file(bucket, key, tmp.name)
    return tmp.name


if S3_BUCKET:
    # ── Production path: download from S3
    _model_file = _load_from_s3(S3_BUCKET, MODEL_S3_KEY, ".pkl")
    _cols_file  = _load_from_s3(S3_BUCKET, FEATURE_COLS_S3_KEY, ".json")
    print(f"Loaded model artifacts from s3://{S3_BUCKET}")
else:
    # ── Local dev path: read from models/ directory
    if not os.path.exists(LOCAL_MODEL_PATH):
        raise RuntimeError(
            f"Trained model not found at {LOCAL_MODEL_PATH}. "
            "Run `python -m src.scripts.run_training` first."
        )
    _model_file = LOCAL_MODEL_PATH
    _cols_file  = LOCAL_COLS_PATH
    print(f"Loaded model artifacts from local disk.")

pipeline = joblib.load(_model_file)

with open(_cols_file) as f:
    FEATURE_COLUMNS: list[str] = json.load(f)


# ── Pydantic schema — raw customer features (matches CSV columns)

class CustomerFeatures(BaseModel):
    customerID: Optional[str] = Field(default=None, description="Customer identifier (optional, not used in prediction)")
    gender: str = Field(..., example="Female")
    SeniorCitizen: int = Field(..., example=0, description="0 = No, 1 = Yes")
    Partner: str = Field(..., example="Yes")
    Dependents: str = Field(..., example="No")
    tenure: int = Field(..., example=12, description="Months with the company")
    PhoneService: str = Field(..., example="Yes")
    MultipleLines: str = Field(..., example="No", description="Yes / No / No phone service")
    InternetService: str = Field(..., example="Fiber optic", description="DSL / Fiber optic / No")
    OnlineSecurity: str = Field(..., example="No", description="Yes / No / No internet service")
    OnlineBackup: str = Field(..., example="Yes", description="Yes / No / No internet service")
    DeviceProtection: str = Field(..., example="No", description="Yes / No / No internet service")
    TechSupport: str = Field(..., example="No", description="Yes / No / No internet service")
    StreamingTV: str = Field(..., example="No", description="Yes / No / No internet service")
    StreamingMovies: str = Field(..., example="No", description="Yes / No / No internet service")
    Contract: str = Field(..., example="Month-to-month", description="Month-to-month / One year / Two year")
    PaperlessBilling: str = Field(..., example="Yes")
    PaymentMethod: str = Field(..., example="Electronic check",
                               description="Electronic check / Mailed check / Bank transfer (automatic) / Credit card (automatic)")
    MonthlyCharges: float = Field(..., example=70.35)
    TotalCharges: float = Field(..., example=845.5, description="Set to 0.0 for brand-new customers (tenure=0)")


class PredictionResponse(BaseModel):
    customerID: Optional[str]
    churn_prediction: str        # "Yes" or "No"
    probability: float           # churn probability (0–1)
    risk_level: str              # "High" / "Medium" / "Low"


# ── App

app = FastAPI(
    title="CRM Churn Intelligence API",
    description="Predict customer churn probability using a trained ML pipeline.",
    version="1.0.0",
)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


def _risk_label(prob: float) -> str:
    if prob >= 0.7:
        return "High"
    elif prob >= 0.4:
        return "Medium"
    return "Low"


def _preprocess(customer: CustomerFeatures) -> pd.DataFrame:
    """
    Replicate the training preprocessing pipeline on a single customer record:
      1. Build a one-row DataFrame from the request body
      2. Feature engineering (clv_proxy, total_services, tenure_bucket)
      3. Drop customerID
      4. Manual encoding (binary / ordinal / OHE)
      5. Reindex to training column order, fill any missing dummies with 0
    """
    data = customer.model_dump()
    df = pd.DataFrame([data])

    # Feature engineering (must happen before encoding while columns are still strings)
    df = apply_feature_engineering(df)

    # Drop non-feature columns
    df = df.drop(columns=["customerID"], errors="ignore")

    # Encoding — matches training exactly
    df = apply_manual_encoding(df)

    # Align to training column schema (pd.get_dummies may omit unseen dummy cols)
    df = df.reindex(columns=FEATURE_COLUMNS, fill_value=0)

    return df


@app.get("/", response_class=FileResponse)
def serve_ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
def health_check():
    source = f"s3://{S3_BUCKET}/{MODEL_S3_KEY}" if S3_BUCKET else _model_file
    return {"status": "ok", "model_source": source}


@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerFeatures):
    try:
        df = _preprocess(customer)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Preprocessing failed: {e}")

    try:
        prob = float(pipeline.predict_proba(df)[0, 1])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model inference failed: {e}")

    label = "Yes" if prob >= 0.5 else "No"

    return PredictionResponse(
        customerID=customer.customerID,
        churn_prediction=label,
        probability=round(prob, 4),
        risk_level=_risk_label(prob),
    )

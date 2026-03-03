# CRM Churn Intelligence

A machine learning system that predicts customer churn for a telecom business. It covers
the full pipeline from exploratory analysis and feature engineering through multi-model
training, hyperparameter tuning, and a production-ready REST API deployed on AWS EC2 with
model artifacts stored in S3.

---

## Table of Contents

1. [Problem Statement](#problem-statement)
2. [Solution Overview](#solution-overview)
3. [Project Structure](#project-structure)
4. [Data & Feature Engineering](#data--feature-engineering)
5. [Model Training & Results](#model-training--results)
6. [API](#api)
7. [Local Setup](#local-setup)
8. [EC2 Deployment](#ec2-deployment)
9. [Scaling, Retraining & Monitoring](#scaling-retraining--monitoring)

---

## Problem Statement

Customer churn is one of the most expensive problems in subscription businesses. Acquiring a
new customer costs 5–25× more than retaining an existing one. This project builds a binary
classifier that flags customers likely to cancel their service, giving the business time to
intervene with targeted retention offers before churn happens.

**Dataset:** IBM Telco Customer Churn — 7,043 customers, 20 features, ~26% churn rate.

---

## Solution Overview

```
Raw CSV  ──▶  Preprocessing  ──▶  Feature Engineering  ──▶  Encoding
                                                                │
                                                                ▼
                                              6-Model Training + Threshold Tuning
                                                                │
                                                                ▼
                                                     Best Model  ──▶  models/final_model.pkl
                                                                              │
                                                                              ▼
                                                              FastAPI POST /predict  ◀──  EC2 / Local
                                                                              │
                                                                              ▼
                                                                     JSON Response
```

Key design decisions are documented in [reports/design_decisions.md](reports/design_decisions.md).

---

## Project Structure

```
crm-churn-intelligence/
├── api/
│   ├── app.py                   # FastAPI application (prediction endpoint + UI)
│   └── static/
│       └── index.html           # Browser-based prediction form
├── data/
│   ├── raw/                     # Source CSV (gitignored)
│   └── samples/
│       └── sample_customer.json # Example request body for curl testing
├── models/
│   ├── final_model.pkl          # Trained sklearn Pipeline (gitignored, stored in S3)
│   └── feature_columns.json     # Column schema saved at training time
├── reports/
│   ├── model_comparison.csv     # Per-model metrics table
│   ├── best_model_metrics.json  # Best model metrics snapshot
│   ├── design_decisions.md      # Encoding & modeling rationale
│   └── *.png                    # EDA and evaluation plots (gitignored)
├── src/
│   ├── data_processing/
│   │   ├── load_data.py         # CSV loading + validation
│   │   ├── features.py          # Feature engineering (CLV proxy, service count, tenure bucket)
│   │   └── preprocess.py        # Encoding, imputation, train/test split
│   ├── modeling/
│   │   ├── train.py             # Model builders + Pipeline assembly
│   │   ├── evaluate.py          # Metrics, threshold tuning, confusion matrices
│   │   └── tune.py              # RandomizedSearchCV for LightGBM & XGBoost
│   └── scripts/
│       └── run_training.py      # End-to-end training entry point
├── churn_analysis.ipynb         # Exploratory analysis notebook
├── .env.example                 # Environment variable template
├── requirements.txt
└── README.md
```

---

## Data & Feature Engineering

### Hidden NaN Fix

The raw CSV has 11 rows where `TotalCharges` is a whitespace string `' '` instead of a
number. `pandas.isnull()` returns `False` for these, so they are invisible to standard null
checks. All 11 affected rows have `tenure = 0` (brand-new customers with no billing cycle
yet). Rather than imputing the median (~$600), these are correctly set to `0` before
numeric conversion — see `src/data_processing/preprocess.py: fix_total_charges()`.

### Engineered Features

| Feature | Formula | Rationale |
|---|---|---|
| `clv_proxy` | `tenure × MonthlyCharges` | Approximates customer lifetime value |
| `total_services` | Count of `"Yes"` across 6 add-on service columns | Engagement depth |
| `tenure_bucket` | Binned into `0-12 / 12-24 / 24-48 / 48+` months | Non-linear tenure effect |

### Encoding Strategy

Encoding is done semantically rather than with blanket one-hot encoding:

| Column type | Strategy |
|---|---|
| Binary Yes/No columns (`Partner`, `PhoneService`, etc.) | `== "Yes"` → `0/1` |
| Service columns with "No internet/phone service" | Treat all non-Yes as `0` |
| `gender` | `== "Male"` → `0/1` |
| `Contract` | Ordinal `0/1/2` (month-to-month = higher risk) |
| `tenure_bucket` | Ordinal `0/1/2/3` |
| `InternetService`, `PaymentMethod` | `pd.get_dummies` (nominal, no natural order) |

Column alignment between training and inference is handled by saving `feature_columns.json`
at training time and reindexing each inference row to match it exactly.

---

## Model Training & Results

Six classifiers are trained and compared. All use `class_weight="balanced"` (or equivalent)
to handle the ~26% churn minority class.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.738 | 0.504 | 0.786 | 0.614 | **0.841** |
| Random Forest | 0.793 | 0.645 | 0.487 | 0.555 | 0.827 |
| **LightGBM** ✓ | 0.768 | 0.547 | **0.733** | **0.626** | 0.834 |
| XGBoost | 0.757 | 0.529 | 0.746 | 0.619 | 0.834 |
| SVM | 0.744 | 0.512 | 0.775 | 0.616 | 0.819 |
| MLP | 0.784 | 0.599 | 0.567 | 0.582 | 0.841 |

**Selected model:** LightGBM — best F1 score (0.626). Threshold tuned to 0.57 for an
improved F1 of 0.634.

### Threshold Tuning

The default 0.5 prediction threshold is optimised per-model by scanning the full probability
range and picking the threshold that maximises F1 on the test set. This is especially
important for the churn use case where recall (catching actual churners) matters more than
precision.

### Hyperparameter Tuning

`--tune` runs `RandomizedSearchCV` over LightGBM and XGBoost (`n_jobs=1` on Windows to
avoid spawn overhead; `n_iter=10` by default, configurable via `--n_iter`).

---

## API

The FastAPI app at `api/app.py` exposes three endpoints:

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Serves the browser prediction form |
| `GET` | `/health` | Returns model status and source path |
| `POST` | `/predict` | Returns churn prediction + probability + risk level |

### Request body (`POST /predict`)

```json
{
  "customerID": "7590-VHVEG",
  "gender": "Female",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "No",
  "MultipleLines": "No phone service",
  "InternetService": "DSL",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 29.85,
  "TotalCharges": 358.20
}
```

### Response

```json
{
  "customerID": "7590-VHVEG",
  "churn_prediction": "Yes",
  "probability": 0.7812,
  "risk_level": "High"
}
```

Risk levels: `High` (≥ 0.7) · `Medium` (≥ 0.4) · `Low` (< 0.4)

### Model loading strategy

| Environment | How the model is loaded |
|---|---|
| Local dev (`S3_BUCKET_NAME` not set) | Loaded from `models/final_model.pkl` |
| EC2 / production (`S3_BUCKET_NAME` set) | Downloaded from S3 at startup |

---

## Local Setup

### 1. Clone and create a virtual environment

```bash
git clone <repo-url>
cd crm-churn-intelligence
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the raw data

Download the [Telco Customer Churn dataset](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
and place the CSV at:

```
data/raw/Telco-Customer-Churn.csv
```

### 4. Run training

```bash
# Baseline (6 models, picks best by F1)
python -m src.scripts.run_training

# With hyperparameter tuning on LightGBM & XGBoost
python -m src.scripts.run_training --tune

# With tuning + more iterations
python -m src.scripts.run_training --tune --n_iter 20
```

Outputs written to `models/` and `reports/`.

### 5. Start the API

```bash
uvicorn api.app:app --reload
```

Open http://localhost:8000 for the browser form, or test with curl:

```bash
curl -s -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/samples/sample_customer.json | python -m json.tool
```

---

## EC2 Deployment

This section describes deploying the API on an Ubuntu EC2 instance with model artifacts
stored in S3.

### Prerequisites

- EC2 instance (Ubuntu 22.04, `t3.micro` or larger)
- S3 bucket (e.g. `crm-churn-intelligence`)
- IAM role attached to the instance with `s3:GetObject` permission on that bucket

### Step 1 — Train locally and upload artifacts to S3

```bash
# Copy .env.example and fill in your S3 bucket name and AWS region
cp .env.example .env
# Edit .env: set S3_BUCKET_NAME, AWS_REGION (access keys not needed if using IAM role)

# Train and push artifacts to S3 in one step
python -m src.scripts.run_training --upload
```

This saves `models/final_model.pkl` and `models/feature_columns.json` to S3.

### Step 2 — Set up the EC2 instance

SSH into the instance:

```bash
ssh -i your-key.pem ubuntu@<EC2-PUBLIC-IP>
```

Install system packages and clone the repo:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv git tmux
git clone <repo-url>
cd crm-churn-intelligence
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Step 3 — Configure environment variables

The IAM role attached to the instance provides S3 credentials automatically — no access
keys are needed on the server. Only two variables are required:

```bash
export S3_BUCKET_NAME=crm-churn-intelligence
export AWS_REGION=ap-south-1
```

Or create a `.env` file on the instance (with only these two variables):

```
S3_BUCKET_NAME=crm-churn-intelligence
AWS_REGION=ap-south-1
```

### Step 4 — Start the server in a persistent tmux session

```bash
tmux new-session -s api
source .venv/bin/activate
export S3_BUCKET_NAME=crm-churn-intelligence
export AWS_REGION=ap-south-1
uvicorn api.app:app --host 0.0.0.0 --port 8000
```

Detach from the session with `Ctrl-B D`. The server keeps running after you disconnect.

**Useful tmux commands:**

| Command | Action |
|---|---|
| `tmux attach -t api` | Reattach to the running session |
| `Ctrl-B D` | Detach (leave server running) |
| `Ctrl-B [` | Scroll mode (use arrow keys, `Q` to exit) |
| `tmux kill-session -t api` | Stop the server |

### Step 5 — Test

From your local machine (EC2 security group must allow inbound TCP on port 8000):

```bash
curl http://<EC2-PUBLIC-IP>:8000/health
curl -X POST http://<EC2-PUBLIC-IP>:8000/predict \
  -H "Content-Type: application/json" \
  -d @data/samples/sample_customer.json
```

---

## Scaling, Retraining & Monitoring

### Scaling to 100k+ records

The current single-instance deployment handles individual predictions efficiently (each
request is one row through a trained pipeline, sub-millisecond inference). For bulk scoring:

- **Batch scoring:** Run `pipeline.predict_proba(df)` directly on a full DataFrame — the
  sklearn pipeline handles tens of thousands of rows in seconds.
- **Horizontal scaling:** Place the FastAPI app behind an Application Load Balancer and run
  multiple EC2 instances (or AWS ECS containers). S3 model storage means every instance
  loads the same artifact without coordination.
- **Serverless option:** Package the inference code as an AWS Lambda function + API Gateway
  for zero-idle-cost scaling.

### Retraining

The training pipeline is fully reproducible via `python -m src.scripts.run_training`. A
retraining workflow would:

1. Trigger on a schedule (monthly) or when data drift is detected.
2. Run the training script on fresh labelled data.
3. Compare new model metrics against the production baseline in `reports/best_model_metrics.json`.
4. Upload the new artifact to S3 with `--upload` only if metrics improve.
5. Restart the API (or use a versioned S3 key + rolling deployment) to load the new model.

This can be automated with AWS EventBridge + CodeBuild or a simple cron job.

### Monitoring

Key signals to watch in production:

| Signal | Tool | Alert if... |
|---|---|---|
| Prediction distribution drift | CloudWatch custom metrics | Mean churn probability shifts > 5% week-over-week |
| Feature distribution drift | Log input features per request; compare with training stats | KS-test p-value < 0.05 for key features |
| API latency / error rate | CloudWatch + ALB access logs | p99 latency > 500ms or error rate > 1% |
| Model accuracy decay | Periodically join predictions with actual churn outcomes | F1 drops > 3 points from baseline |

### Cost considerations

| Resource | Free Tier / Low-cost option |
|---|---|
| Model storage | S3 Standard — first 5 GB free, then ~$0.023/GB/month |
| Inference server | `t3.micro` EC2 — 750 hours/month free for 12 months |
| Batch retraining | Spot instances reduce training cost by ~70% |
| Monitoring | CloudWatch — first 10 custom metrics free |

For a small-to-medium CRM (< 1M predictions/month) the total AWS bill stays well under $10/month.

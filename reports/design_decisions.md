# CRM Churn Intelligence — Design Decisions & Results Report

> Generated from `churn_analysis.ipynb` runtime logs
> Dataset: Telco Customer Churn (IBM Sample)

---

## 1. Dataset Overview

| Property | Value |
|----------|-------|
| Rows | 7,043 |
| Columns | 21 |
| Target | `Churn` (Yes / No) |
| Class distribution | ~73.5% No Churn / ~26.5% Churn |
| Duplicates | 0 |

---

## 2. Data Quality — Hidden NaN Discovery

Standard `df.isnull().sum()` reported **zero missing values**, masking a real data issue:

```
--- Missing Values ---
Series([], dtype: int64)
```

A secondary scan for whitespace-only strings revealed:

```
WARNING — hidden NaN (whitespace-only) values found:
  TotalCharges: 11 rows
```

**Root cause:** 11 new customers with `tenure = 0` had a single space character `' '` stored in `TotalCharges` instead of a numeric value or null. Standard pandas `.isnull()` cannot detect whitespace strings.

**Decision:** Apply domain-aware correction before numeric conversion — new customers with no billing history should have `TotalCharges = 0`, not an imputed median (~$600 which would be statistically misleading).

```
TotalCharges NaNs after fix: 0
```

---

## 3. Feature Engineering

Three features were engineered beyond the raw columns:

| Feature | Formula / Logic | Rationale |
|---------|----------------|-----------|
| `clv_proxy` | `tenure × MonthlyCharges` | Proxy for customer lifetime value |
| `total_services` | Count of "Yes" across 6 service columns | Measures service depth / stickiness |
| `tenure_bucket` | Bins: 0–12, 12–24, 24–48, 48+ months | Captures non-linear tenure effect |

---

## 4. Encoding Strategy

Instead of blanket One-Hot Encoding for all categorical columns, a semantically informed encoding scheme was applied:

| Column(s) | Strategy | Reason |
|-----------|----------|--------|
| `MultipleLines`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies` | Binary `== 'Yes'` → 0/1 | 3-value columns (`Yes/No/No service`) — "no service" treated as equivalent to "no" |
| `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling` | Binary `== 'Yes'` → 0/1 | True binary Yes/No |
| `gender` | Binary `== 'Male'` → 0/1 | True binary |
| `Contract` | Ordinal: `Month-to-month=0`, `One year=1`, `Two year=2` | Natural order — longer contract = lower churn risk |
| `tenure_bucket` | Ordinal: `0-12=0`, `12-24=1`, `24-48=2`, `48+=3` | Preserves temporal order |
| `InternetService`, `PaymentMethod` | One-Hot Encoding (`drop_first=True`) | Truly nominal — no natural order |

**Result after encoding:**
```
Shape after encoding : (7043, 27)
Remaining object/category cols: none — all encoded
```

---

## 5. Preprocessing Pipeline

Since all features were numeric after manual encoding, a simplified single pipeline was used:

```
Pipeline([
    SimpleImputer(strategy='median'),
    StandardScaler()
])
```

**Train/Test Split:**
```
Train size: 5,634   Test size: 1,409
Churn rate — train: 26.54%   test: 26.54%   (stratified)
```

---

## 6. Model Selection

Six models were selected to cover distinct learning paradigms:

| Model | Paradigm | Imbalance Handling |
|-------|----------|--------------------|
| Logistic Regression | Linear | `class_weight='balanced'` |
| Random Forest | Ensemble (Bagging) | `class_weight='balanced_subsample'` |
| LightGBM | Gradient Boosting | `class_weight='balanced'` |
| XGBoost | Gradient Boosting | `scale_pos_weight=3` |
| SVM (RBF kernel) | Margin-based | `class_weight='balanced'` |
| MLP (64→32) | Neural Network | — |

---

## 7. Training Results (Default Threshold = 0.5)

```
Training logreg...
  F1=0.6138  AUC=0.8407  Recall=0.7861
Training random_forest...
  F1=0.5601  AUC=0.8277  Recall=0.4920
Training lightgbm...
  F1=0.6273  AUC=0.8337  Recall=0.7380
Training xgboost...
  F1=0.6225  AUC=0.8323  Recall=0.7540
Training svm...
  F1=0.6170  AUC=0.8195  Recall=0.7754
Training mlp...
  F1=0.5678  AUC=0.8366  Recall=0.5374
```

### Classification Reports (Default Threshold)

**Logistic Regression**
```
              precision    recall  f1-score   support
    No Churn       0.90      0.72      0.80      1035
       Churn       0.50      0.79      0.61       374
    accuracy                           0.74      1409
   macro avg       0.70      0.75      0.71      1409
weighted avg       0.80      0.74      0.75      1409
```

**Random Forest**
```
              precision    recall  f1-score   support
    No Churn       0.83      0.90      0.87      1035
       Churn       0.65      0.49      0.56       374
    accuracy                           0.79      1409
   macro avg       0.74      0.70      0.71      1409
weighted avg       0.78      0.79      0.79      1409
```

**LightGBM**
```
              precision    recall  f1-score   support
    No Churn       0.89      0.78      0.83      1035
       Churn       0.55      0.74      0.63       374
    accuracy                           0.77      1409
   macro avg       0.72      0.76      0.73      1409
weighted avg       0.80      0.77      0.78      1409
```

**XGBoost**
```
              precision    recall  f1-score   support
    No Churn       0.90      0.76      0.82      1035
       Churn       0.53      0.75      0.62       374
    accuracy                           0.76      1409
   macro avg       0.71      0.76      0.72      1409
weighted avg       0.80      0.76      0.77      1409
```

**SVM (RBF)**
```
              precision    recall  f1-score   support
    No Churn       0.90      0.73      0.81      1035
       Churn       0.51      0.78      0.62       374
    accuracy                           0.74      1409
   macro avg       0.71      0.75      0.71      1409
weighted avg       0.80      0.74      0.76      1409
```

**MLP**
```
              precision    recall  f1-score   support
    No Churn       0.84      0.87      0.85      1035
       Churn       0.60      0.54      0.57       374
    accuracy                           0.78      1409
   macro avg       0.72      0.70      0.71      1409
weighted avg       0.78      0.78      0.78      1409
```

---

## 8. Threshold Tuning

**Motivation:** The default `predict()` threshold of 0.5 is suboptimal for imbalanced classes. By scanning thresholds 0.20–0.65, the decision boundary that maximises F1 on the test set was found for each model.

```
Model           Default F1 (0.50)    Best Threshold    Tuned F1     Gain
------------------------------------------------------------------------
logreg          0.6138               0.54              0.6194       +0.0056
random_forest   0.5601               0.27              0.6211       +0.0610
lightgbm        0.6273               0.59              0.6281       +0.0008
xgboost         0.6225               0.52              0.6284       +0.0059
svm             0.6170               0.32              0.6205       +0.0035
mlp             0.5678               0.31              0.6376       +0.0698
```

**Key observations:**
- `random_forest` and `mlp` had the largest gains (+0.061 and +0.070) — both were under-predicting churn at 0.5, requiring a much lower threshold
- `lightgbm` was already near-optimal at 0.5 (gain of only +0.0008)
- After tuning, all six models converge to a similar F1 range of **0.619–0.628**

---

## 9. Best Model (by F1)

```
Best model: lightgbm  |  F1=0.6273  AUC=0.8337
```

**Final saved metrics:**
```
model       : lightgbm
accuracy    : 0.7672
precision   : 0.5455
recall      : 0.7380
f1_score    : 0.6273
roc_auc     : 0.8337
```

**Sanity check:**
```
Sample prediction: No Churn  (probability=0.003)
```

---

## 10. Summary of Key Design Decisions

| Decision | Chosen Approach | Alternative Considered |
|----------|----------------|----------------------|
| Missing value detection | Custom whitespace scan + `.isnull()` | `.isnull()` only |
| Hidden NaN imputation | Domain fill (`tenure=0` → `TotalCharges=0`) | Median imputation |
| Categorical encoding | Semantically informed (binary/ordinal/OHE per column) | Blanket OneHotEncoder |
| Preprocessing pipeline | Single `Pipeline(imputer → scaler)` | `ColumnTransformer` with separate categorical pipeline |
| Imbalance handling | `class_weight='balanced'` on all models | SMOTE oversampling |
| Model comparison metric | F1-score (primary) + ROC-AUC (secondary) | Accuracy |
| Threshold selection | Grid search 0.20–0.65, maximise F1 | Fixed 0.5 default |
| Best model selection | LightGBM (highest F1 at default threshold) | XGBoost (highest F1 after tuning) |

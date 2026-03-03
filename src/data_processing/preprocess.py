import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from .features import apply_feature_engineering


def fix_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix TotalCharges hidden NaNs: strip whitespace, fill empty strings with '0'
    (customers with tenure=0 have no charges yet), then convert to numeric.
    """
    df["TotalCharges"] = df["TotalCharges"].str.strip()
    df.loc[df["TotalCharges"] == "", "TotalCharges"] = "0"
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    return df


def apply_manual_encoding(df: pd.DataFrame) -> pd.DataFrame:
    """
    Semantically informed encoding — replaces blanket OneHotEncoding.

    Strategy per column type:
      - Binary-ish service cols  → (== 'Yes') → 0/1
      - Simple Yes/No cols       → (== 'Yes') → 0/1
      - Gender                   → (== 'Male') → 0/1
      - Contract                 → ordinal 0/1/2
      - tenure_bucket            → ordinal 0/1/2/3
      - InternetService,
        PaymentMethod            → pd.get_dummies (drop_first=True)
    """
    # Binary-ish: Yes/No/No internet service or No phone service → 1/0
    binary_service_cols = [
        "MultipleLines", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
    ]
    for c in binary_service_cols:
        df[c] = (df[c] == "Yes").astype(int)

    # Simple Yes/No binary columns
    for c in ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]:
        df[c] = (df[c] == "Yes").astype(int)

    # Gender
    df["gender"] = (df["gender"] == "Male").astype(int)

    # Ordinal: Contract (shorter term = higher churn risk)
    df["Contract"] = df["Contract"].map({"Month-to-month": 0, "One year": 1, "Two year": 2})

    # Ordinal: tenure_bucket
    df["tenure_bucket"] = df["tenure_bucket"].map(
        {"0-12": 0, "12-24": 1, "24-48": 2, "48+": 3}
    ).astype(int)

    # Nominal OHE: truly nominal columns with no natural order
    df = pd.get_dummies(df, columns=["InternetService", "PaymentMethod"], drop_first=True, dtype=int)

    return df


def drop_correlated_features(df: pd.DataFrame, threshold: float = 0.90) -> pd.DataFrame:
    """
    Drop one column from each pair whose absolute Pearson correlation exceeds
    the threshold. Upper triangle only to avoid duplicate pairs.
    """
    corr = df.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    if to_drop:
        print(f"Dropping {len(to_drop)} correlated feature(s): {to_drop}")
    return df.drop(columns=to_drop)


def map_target(df: pd.DataFrame) -> pd.DataFrame:
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})
    return df


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def build_preprocessor() -> Pipeline:
    """
    Simple imputer + scaler pipeline.
    All features are numeric after manual encoding — no ColumnTransformer needed.
    """
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
    ])


def prepare_training_data(df: pd.DataFrame):
    """
    Full preprocessing pipeline:
      1. Fix TotalCharges hidden NaNs (domain-aware)
      2. Feature engineering (clv_proxy, total_services, tenure_bucket)
      3. Map target (Yes/No → 1/0)
      4. Drop customerID
      5. Manual encoding (binary / ordinal / OHE)
      6. Drop highly correlated features (threshold=0.90)
      7. Stratified train/test split
      8. Build simple preprocessor
    """
    df = fix_total_charges(df)
    df = apply_feature_engineering(df)
    df = map_target(df)
    df = df.drop(columns=["customerID"])
    df = apply_manual_encoding(df)

    X = df.drop("Churn", axis=1)
    X = drop_correlated_features(X)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = split_data(X, y)
    preprocessor = build_preprocessor()

    return X_train, X_test, y_train, y_test, preprocessor

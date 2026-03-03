import os
import pandas as pd


REQUIRED_COLUMNS = [
    "customerID", "gender", "SeniorCitizen", "Partner", "Dependents",
    "tenure", "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection", "TechSupport",
    "StreamingTV", "StreamingMovies", "Contract", "PaperlessBilling",
    "PaymentMethod", "MonthlyCharges", "TotalCharges", "Churn"
]


def load_raw_data(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File not found at: {path}")
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError("Loaded dataframe is empty.")
    return df


def detect_hidden_nans(df: pd.DataFrame) -> dict:
    """
    Detect whitespace-only string values that .isnull() cannot catch.
    Returns {column: count} for any affected columns.
    """
    hidden = {}
    for col in df.select_dtypes(include="object").columns:
        count = (df[col].str.strip() == "").sum()
        if count > 0:
            hidden[col] = int(count)
    return hidden


def validate_raw_data(df: pd.DataFrame) -> None:
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    if df.duplicated().sum() > 0:
        raise ValueError("Dataset contains duplicate rows.")

    hidden = detect_hidden_nans(df)
    if hidden:
        print("WARNING — hidden NaN (whitespace-only) values found:")
        for col, count in hidden.items():
            print(f"  {col}: {count} rows")
    else:
        print("No hidden NaN values detected.")


def summarize_data(df: pd.DataFrame) -> dict:
    return {
        "shape": list(df.shape),
        "missing_values": df.isnull().sum().to_dict(),
        "churn_distribution": df["Churn"].value_counts(normalize=True).to_dict(),
    }

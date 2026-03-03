import pandas as pd

SERVICE_COLUMNS = [
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies"
]

def add_clv_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Customer Lifetime Value proxy.
    """
    df["clv_proxy"] = df["tenure"] * df["MonthlyCharges"]
    return df

def add_total_services(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count number of subscribed add-on services.
    """
    df["total_services"] = df[SERVICE_COLUMNS].apply(
        lambda row: sum(row == "Yes"), axis=1
    )
    return df

def add_tenure_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """
    Bucket tenure into categorical segments.
    """
    bins = [0, 12, 24, 48, 100]
    labels = ["0-12", "12-24", "24-48", "48+"]

    df["tenure_bucket"] = pd.cut(
        df["tenure"],
        bins=bins,
        labels=labels,
        include_lowest=True
    )

    return df

def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master feature engineering function.
    """
    df = add_clv_proxy(df)
    df = add_total_services(df)
    df = add_tenure_bucket(df)
    return df

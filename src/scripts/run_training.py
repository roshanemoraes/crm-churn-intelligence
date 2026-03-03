import os
import json
import argparse
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

from src.data_processing.load_data import load_raw_data, validate_raw_data, summarize_data
from src.data_processing.preprocess import prepare_training_data
from src.modeling.train import train_and_evaluate, save_model
from src.modeling.evaluate import save_metrics, tune_threshold, save_confusion_matrix
from src.modeling.tune import tune_model


DEFAULT_DATA_PATH = "data/raw/Telco-Customer-Churn.csv"
DEFAULT_REPORTS_DIR = "reports"
DEFAULT_MODELS_DIR = "models"

MODEL_NAMES = ["logreg", "random_forest", "lightgbm", "xgboost", "svm", "mlp"]
TUNABLE_MODELS = ["lightgbm", "xgboost"]


def ensure_dirs():
    os.makedirs(DEFAULT_REPORTS_DIR, exist_ok=True)
    os.makedirs(DEFAULT_MODELS_DIR, exist_ok=True)


def save_model_comparison(results: list, path: str = "reports/model_comparison.csv"):
    df = pd.DataFrame(results)
    df.to_csv(path, index=False)
    return df


def main(args):
    ensure_dirs()

    df = load_raw_data(args.data_path)
    validate_raw_data(df)
    summary = summarize_data(df)
    with open(os.path.join(DEFAULT_REPORTS_DIR, "data_summary.json"), "w") as f:
        json.dump(summary, f, indent=4)

    X_train, X_test, y_train, y_test, preprocessor = prepare_training_data(df)

    all_results = []
    best_model_name = None
    best_f1 = -1
    best_pipeline = None
    best_metrics = None

    for name in MODEL_NAMES:
        print(f"\nTraining {name}...")
        pipeline, metrics = train_and_evaluate(name, preprocessor, X_train, y_train, X_test, y_test)

        # Threshold tuning
        y_proba = pipeline.predict_proba(X_test)[:, 1]
        best_t, f1_tuned, _ = tune_threshold(y_test, y_proba)
        metrics["best_threshold"] = best_t
        metrics["f1_tuned"] = f1_tuned

        print(f"  Default F1={metrics['f1_score']:.4f}  |  "
              f"Tuned F1={f1_tuned:.4f} @ threshold={best_t:.2f}")

        save_confusion_matrix(
            y_test, pipeline.predict(X_test),
            model_name=name,
            reports_dir=DEFAULT_REPORTS_DIR,
        )

        all_results.append({"model": name, **metrics})

        if metrics["f1_score"] > best_f1:
            best_f1 = metrics["f1_score"]
            best_model_name = name
            best_pipeline = pipeline
            best_metrics = metrics

    if args.tune:
        print("\n" + "=" * 55)
        print("  HYPERPARAMETER TUNING (RandomizedSearchCV)")
        print(f"  n_iter={args.n_iter}  cv=5  scoring=f1  n_jobs=1")
        print("=" * 55)

        for name in TUNABLE_MODELS:
            print(f"\nTuning {name}...")
            tuned_pipeline, tuned_metrics = tune_model(
                name, preprocessor, X_train, y_train, X_test, y_test,
                n_iter=args.n_iter,
            )
            tuned_name = f"{name}_tuned"

            y_proba = tuned_pipeline.predict_proba(
                X_test.values if hasattr(X_test, "values") else X_test
            )[:, 1]
            best_t, f1_tuned, _ = tune_threshold(y_test, y_proba)
            tuned_metrics["best_threshold"] = best_t
            tuned_metrics["f1_tuned"] = f1_tuned

            print(f"  Tuned F1={tuned_metrics['f1_score']:.4f}  |  "
                  f"Threshold-tuned F1={f1_tuned:.4f} @ threshold={best_t:.2f}")

            save_confusion_matrix(
                y_test, tuned_pipeline.predict(
                    X_test.values if hasattr(X_test, "values") else X_test
                ),
                model_name=tuned_name,
                reports_dir=DEFAULT_REPORTS_DIR,
            )

            all_results.append({"model": tuned_name, **tuned_metrics})

            if tuned_metrics["f1_score"] > best_f1:
                best_f1 = tuned_metrics["f1_score"]
                best_model_name = tuned_name
                best_pipeline = tuned_pipeline
                best_metrics = tuned_metrics

    comparison_df = save_model_comparison(all_results)
    print("\nModel Comparison:\n")
    print(comparison_df.to_string(index=False))

    best_model_path = os.path.join(DEFAULT_MODELS_DIR, "final_model.pkl")
    save_model(best_pipeline, best_model_path)

    feature_cols_path = os.path.join(DEFAULT_MODELS_DIR, "feature_columns.json")
    with open(feature_cols_path, "w") as f:
        json.dump(X_train.columns.tolist(), f, indent=2)

    metrics_path = os.path.join(DEFAULT_REPORTS_DIR, "best_model_metrics.json")
    save_metrics(best_metrics, metrics_path)

    print(f"\nTraining complete!")
    print(f"Best model : {best_model_name}  (F1={best_f1:.4f})")
    print(f"Saved model to        : {best_model_path}")
    print(f"Saved feature schema  : {feature_cols_path}")
    print(f"Saved best metrics to : {metrics_path}")
    print(f"Saved comparison to   : reports/model_comparison.csv")

    if args.upload:
        _upload_to_s3(best_model_path, feature_cols_path)


def _upload_to_s3(model_path: str, feature_cols_path: str):
    import boto3

    bucket = os.getenv("S3_BUCKET_NAME", "")
    region = os.getenv("AWS_REGION", "eu-west-1")
    model_key = os.getenv("MODEL_S3_KEY", "models/final_model.pkl")
    cols_key  = os.getenv("FEATURE_COLS_S3_KEY", "models/feature_columns.json")

    if not bucket:
        print("\nS3_BUCKET_NAME is not set — skipping upload.")
        print("Set it in your .env file or environment and re-run with --upload.")
        return

    s3 = boto3.client("s3", region_name=region)

    for local_path, s3_key in [(model_path, model_key), (feature_cols_path, cols_key)]:
        print(f"Uploading {local_path} → s3://{bucket}/{s3_key} ...")
        s3.upload_file(local_path, bucket, s3_key)

    print(f"Artifacts uploaded to s3://{bucket}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run churn model training and evaluation")
    parser.add_argument(
        "--data_path", type=str, default=DEFAULT_DATA_PATH,
        help="Path to Telco churn CSV"
    )
    parser.add_argument(
        "--tune", action="store_true",
        help="Run RandomizedSearchCV on LightGBM and XGBoost after baseline training"
    )
    parser.add_argument(
        "--n_iter", type=int, default=10,
        help="Number of RandomizedSearchCV iterations per model (default: 10)"
    )
    parser.add_argument(
        "--upload", action="store_true",
        help="Upload final_model.pkl and feature_columns.json to S3 after training (requires S3_BUCKET_NAME env var)"
    )
    args = parser.parse_args()
    main(args)

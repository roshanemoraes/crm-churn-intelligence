import joblib
import os
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier

from .evaluate import evaluate_model


def build_model(model_name: str):
    models = {
        "logreg": LogisticRegression(
            max_iter=10000, class_weight="balanced", random_state=42
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, class_weight="balanced_subsample", random_state=42
        ),
        "lightgbm": LGBMClassifier(
            n_estimators=300, learning_rate=0.05, class_weight="balanced",
            random_state=42, verbose=-1
        ),
        "xgboost": XGBClassifier(
            n_estimators=300, learning_rate=0.05, scale_pos_weight=3,
            random_state=42, verbosity=0, eval_metric="logloss"
        ),
        "svm": SVC(
            kernel="rbf", class_weight="balanced", probability=True, random_state=42
        ),
        "mlp": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=500, early_stopping=True, random_state=42
        ),
    }
    if model_name not in models:
        raise ValueError(f"Unsupported model: '{model_name}'. Choose from: {list(models)}")
    return models[model_name]


def train_and_evaluate(model_name, preprocessor, X_train, y_train, X_test, y_test):
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", build_model(model_name)),
    ])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_test, y_test, model_name)
    return pipeline, metrics


def save_model(pipeline, path="models/final_model.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(pipeline, path)

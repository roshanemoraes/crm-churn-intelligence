import warnings
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier


warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names",
    category=UserWarning,
)


LGBM_PARAM_GRID = {
    "model__num_leaves":        [31, 63, 127],
    "model__min_child_samples": [10, 20, 50],
    "model__learning_rate":     [0.01, 0.05, 0.1],
    "model__n_estimators":      [200, 300, 500],
    "model__subsample":         [0.7, 0.8, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 1.0],
}

XGB_PARAM_GRID = {
    "model__max_depth":         [3, 4, 5, 6],
    "model__min_child_weight":  [1, 3, 5],
    "model__learning_rate":     [0.01, 0.05, 0.1],
    "model__n_estimators":      [200, 300, 500],
    "model__subsample":         [0.7, 0.8, 1.0],
    "model__colsample_bytree":  [0.7, 0.8, 1.0],
    "model__gamma":             [0, 0.1, 0.2],
}

_PARAM_GRIDS = {
    "lightgbm": LGBM_PARAM_GRID,
    "xgboost":  XGB_PARAM_GRID,
}

_BASE_ESTIMATORS = {
    "lightgbm": LGBMClassifier(class_weight="balanced", random_state=42, verbose=-1),
    "xgboost":  XGBClassifier(scale_pos_weight=3, random_state=42, verbosity=0, eval_metric="logloss"),
}


def tune_model(model_name: str, preprocessor, X_train, y_train, X_test, y_test, n_iter: int = 10):
    """
    Run RandomizedSearchCV for lightgbm or xgboost.
    Returns (best_pipeline, metrics_dict).
    Only lightgbm and xgboost are supported — these are the most tunable models.
    """
    if model_name not in _PARAM_GRIDS:
        raise ValueError(
            f"tune_model only supports {list(_PARAM_GRIDS)}. "
            f"Got '{model_name}'."
        )

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", _BASE_ESTIMATORS[model_name]),
    ])

    import numpy as np
    X_tr = X_train.values if hasattr(X_train, "values") else X_train
    X_te = X_test.values  if hasattr(X_test,  "values") else X_test

    search = RandomizedSearchCV(
        pipeline,
        _PARAM_GRIDS[model_name],
        n_iter=n_iter,
        scoring="f1",
        cv=5,
        random_state=42,
        n_jobs=1,
        verbose=1,
    )
    search.fit(X_tr, y_train)

    best_pipeline = search.best_estimator_
    print(f"\nBest params for {model_name}:")
    for k, v in search.best_params_.items():
        print(f"  {k}: {v}")

    y_pred  = best_pipeline.predict(X_te)
    y_proba = best_pipeline.predict_proba(X_te)[:, 1]

    metrics = {
        "model":     f"{model_name}_tuned",
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    return best_pipeline, metrics

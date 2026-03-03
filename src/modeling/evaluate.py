import json
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)


def evaluate_model(pipeline, X_test, y_test, model_name: str = "model") -> dict:
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    metrics = {
        "model":     model_name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall":    round(recall_score(y_test, y_pred), 4),
        "f1_score":  round(f1_score(y_test, y_pred), 4),
        "roc_auc":   round(roc_auc_score(y_test, y_proba), 4),
    }

    print(f"\n{'='*50}\n  {model_name.upper()}\n{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


def tune_threshold(y_test, y_proba, thresholds=None):
    """
    Scan classification thresholds to find the one that maximises F1.
    Returns (best_threshold, best_f1, y_pred_tuned).
    """
    if thresholds is None:
        thresholds = np.arange(0.20, 0.65, 0.01)

    best_t, best_f1 = 0.5, 0.0
    for t in thresholds:
        score = f1_score(y_test, (y_proba >= t).astype(int))
        if score > best_f1:
            best_f1, best_t = score, t

    y_pred_tuned = (y_proba >= best_t).astype(int)
    return round(float(best_t), 2), round(float(best_f1), 4), y_pred_tuned


def save_confusion_matrix(y_true, y_pred, model_name: str = "model", reports_dir: str = "reports"):
    os.makedirs(reports_dir, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(model_name, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(reports_dir, f"confusion_matrix_{model_name}.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()


def save_metrics(metrics: dict, path: str = "reports/best_model_metrics.json"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

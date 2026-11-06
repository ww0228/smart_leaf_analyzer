"""Classification metrics used by the model-development and validation workflow.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

FINAL_MODEL_NAME = "LogisticRegression"


def metric_summary(y_true: np.ndarray,
                   y_pred: np.ndarray,
                   y_prob: np.ndarray,
                   model_name: str,
                   split_name: str) -> dict[str, Any]:
    auc = np.nan
    if len(np.unique(y_true)) == 2:
        auc = roc_auc_score(y_true, y_prob)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    recall_healthy = recall_score(y_true, y_pred, pos_label=0, zero_division=0)
    recall_tipburn = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    return {
        "Split": split_name,
        "Model": model_name,
        "n": len(y_true),
        "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "ROC_AUC": auc,
        "Precision_Healthy": precision_score(y_true, y_pred, pos_label=0, zero_division=0),
        "Recall_Healthy": recall_healthy,
        "Precision_Tipburn": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "Recall_Tipburn": recall_tipburn,
        "Balanced_Accuracy": (recall_healthy + recall_tipburn) / 2.0,
        "TN": int(cm[0, 0]),
        "FP": int(cm[0, 1]),
        "FN": int(cm[1, 0]),
        "TP": int(cm[1, 1]),
    }


def binary_classification_metrics(y_true: np.ndarray,
                                  y_pred: np.ndarray,
                                  y_prob: np.ndarray,
                                  split_name: str,
                                  model_name: str = FINAL_MODEL_NAME) -> dict[str, Any]:
    metrics = metric_summary(y_true, y_pred, y_prob, model_name, split_name)
    metrics["Accuracy"] = (metrics["TN"] + metrics["TP"]) / max(1, len(y_true))
    metrics["Macro_F1"] = metrics["F1_macro"]
    return metrics

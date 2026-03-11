"""Метрики. Основная — F1 токсичного класса (метка 1)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def classification_metrics(y_true, y_pred, scores=None) -> dict:
    """scores — любые монотонные оценки токсичности (вероятность или decision_function)."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    result = {
        "f1_toxic": f1_score(y_true, y_pred, pos_label=1, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "precision_toxic": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "recall_toxic": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
    }
    if scores is not None and len(np.unique(scores)) > 1:
        result["roc_auc"] = roc_auc_score(y_true, scores)
        result["average_precision"] = average_precision_score(y_true, scores)
    result = {k: round(float(v), 4) for k, v in result.items()}
    result["confusion_matrix"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
    return result

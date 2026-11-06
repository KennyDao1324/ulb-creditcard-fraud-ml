from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def best_f1_threshold(target: np.ndarray, probability: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(target, probability)
    if thresholds.size == 0:
        return 0.5
    scores = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    return float(thresholds[int(np.nanargmax(scores))])


def binary_metrics(target, probability, threshold: float) -> dict[str, float | int]:
    target_array = np.asarray(target, dtype=int)
    probability_array = np.asarray(probability, dtype=float)
    prediction = (probability_array >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(target_array, prediction, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(target_array, prediction)),
        "precision": float(precision_score(target_array, prediction, zero_division=0)),
        "recall": float(recall_score(target_array, prediction, zero_division=0)),
        "f1": float(f1_score(target_array, prediction, zero_division=0)),
        "roc_auc": float(roc_auc_score(target_array, probability_array)),
        "pr_auc": float(average_precision_score(target_array, probability_array)),
        "false_positive_rate": float(fp / max(fp + tn, 1)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

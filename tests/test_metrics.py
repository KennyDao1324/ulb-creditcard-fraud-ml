import numpy as np

from fraudbench.metrics import best_f1_threshold, binary_metrics


def test_binary_metrics_are_valid() -> None:
    target = np.array([0, 0, 1, 1])
    score = np.array([0.1, 0.2, 0.8, 0.9])
    threshold = best_f1_threshold(target, score)
    metrics = binary_metrics(target, score, threshold)
    assert metrics["f1"] == 1.0
    assert metrics["pr_auc"] == 1.0

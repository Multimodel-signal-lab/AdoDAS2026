from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    roc_auc_score,
)


def binary_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> float:
    preds = (probs >= threshold).astype(int)
    scores = []
    for c in range(probs.shape[1]):
        scores.append(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
    return float(np.mean(scores))


def per_class_f1(probs: np.ndarray, labels: np.ndarray, threshold: float = 0.5) -> list[float]:
    preds = (probs >= threshold).astype(int)
    return [
        float(f1_score(labels[:, c], preds[:, c], zero_division=0.0))
        for c in range(probs.shape[1])
    ]


def macro_auroc(probs: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(probs.shape[1]):
        unique = np.unique(labels[:, c])
        if len(unique) < 2:
            scores.append(0.0)
        else:
            scores.append(float(roc_auc_score(labels[:, c], probs[:, c])))
    return float(np.mean(scores))


def _quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 4) -> float:
    N = num_classes
    w = np.zeros((N, N), dtype=np.float64)
    for i in range(N):
        for j in range(N):
            w[i, j] = (i - j) ** 2 / ((N - 1) ** 2)

    hist_true = np.bincount(y_true, minlength=N).astype(np.float64)
    hist_pred = np.bincount(y_pred, minlength=N).astype(np.float64)
    n = len(y_true)

    O = np.zeros((N, N), dtype=np.float64)
    for t, p in zip(y_true, y_pred):
        O[t, p] += 1

    E = np.outer(hist_true, hist_pred) / n

    num = np.sum(w * O)
    den = np.sum(w * E)
    if den == 0:
        return 1.0
    return 1.0 - num / den


def mean_qwk(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(preds.shape[1]):
        scores.append(_quadratic_weighted_kappa(labels[:, c], preds[:, c]))
    return float(np.mean(scores))


def per_item_qwk(preds: np.ndarray, labels: np.ndarray) -> list[float]:
    return [
        _quadratic_weighted_kappa(labels[:, c], preds[:, c])
        for c in range(preds.shape[1])
    ]


def mean_mae(preds: np.ndarray, labels: np.ndarray) -> float:
    scores = []
    for c in range(preds.shape[1]):
        scores.append(float(mean_absolute_error(labels[:, c], preds[:, c])))
    return float(np.mean(scores))

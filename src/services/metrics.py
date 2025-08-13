"""
Binary classification metrics and simple threshold tuning.

This module implements:
- `compute_metrics_from_probs`: accuracy/precision/recall/MCC computed
  from positive-class probabilities at a given threshold.
- `_binary_clf_curve`: helper that builds monotonic FP/TP counts vs thresholds
  (scores sorted descending), similar in spirit to scikit-learn's utility.
- `roc_auc`: trapezoidal ROC-AUC.
- `pr_auc`: trapezoidal Precision-Recall AUC.
- `tune_threshold`: grid search for the best decision threshold by either
  F1 score or Youden's J statistic.

Notes
-----
- All operations assume **binary** targets encoded as {0, 1}.
- Small epsilons are used to avoid division-by-zero.
"""

import math
import numpy as np
from typing import Dict, Tuple


def compute_metrics_from_probs(proba: np.ndarray, y: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    """Compute accuracy, precision, recall, and MCC at a fixed threshold.

    Args
    ----
    proba:
        Positive-class probabilities, shape (N,) or (N, 1).
    y:
        Ground-truth binary labels (0/1), shape (N,) or (N, 1).
    thr:
        Decision threshold applied to `proba` (default: 0.5).

    Returns
    -------
    dict
        Dictionary with keys: ``accuracy``, ``precision``, ``recall``, ``mcc``.

    Notes
    -----
    - Adds a small epsilon in denominators for numerical stability.
    - MCC formula: (TP*TN - FP*FN) / sqrt((TP+FP)(TP+FN)(TN+FP)(TN+FN)).
    """
    proba = proba.ravel()
    y = y.ravel().astype(int)
    yhat = (proba >= thr).astype(int)

    # Confusion matrix counts
    tp = int(np.sum((yhat == 1) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))

    # Metrics with epsilon for stability
    eps = 1e-8
    acc  = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)

    # Matthews Correlation Coefficient (MCC)
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = num / den

    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "mcc": float(mcc)}


def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Construct FP/TP counts and thresholds for binary classifier scores.

    The arrays are computed by sorting scores in **descending** order and
    stepping through distinct score values.

    Args
    ----
    y_true:
        Ground-truth labels in {0,1}, shape (N,).
    y_score:
        Prediction scores/probabilities for the positive class, shape (N,).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        (fps, tps, thresholds) where arrays are aligned and monotonic with
        decreasing thresholds.

    Notes
    -----
    - `fps` here is cumulative FP count; `tps` is cumulative TP count.
    - This mirrors the internal logic typically used to build ROC/PR curves.
    """
    order = np.argsort(-y_score)
    y_true = y_true.astype(int).ravel()[order]
    y_score = y_score.ravel()[order]

    # Indices where the sorted score value changes
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # Cumulative positives among sorted labels at those thresholds
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    return fps, tps, thresholds


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal integration.

    Args
    ----
    y_true:
        Binary labels {0,1}.
    y_score:
        Positive-class scores/probabilities.

    Returns
    -------
    float
        Area under the ROC curve.

    Notes
    -----
    - FPR = FP / (FP + TN), TPR = TP / (TP + FN).
    - Uses small denominators via `np.maximum(..., 1e-16)` for stability.
    """
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fns = tps[-1] - tps
    tns = (y_true.size - tps - fps)

    fpr = fps / np.maximum(fps + tns, 1e-16)
    tpr = tps / np.maximum(tps + fns, 1e-16)

    # Ensure strict ascending in the integration axis
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute Precision-Recall AUC via trapezoidal integration.

    Args
    ----
    y_true:
        Binary labels {0,1}.
    y_score:
        Positive-class scores/probabilities.

    Returns
    -------
    float
        Area under the Precision-Recall curve.

    Notes
    -----
    - precision = TP / (TP + FP), recall = TP / P, where P = total positives.
    - Uses `np.maximum(..., 1e-16)` to stabilize divisions.
    """
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    precision = tps / np.maximum(tps + fps, 1e-16)
    recall    = tps / np.maximum(tps[-1], 1e-16)

    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))


def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str = "f1") -> Tuple[float, float]:
    """Grid-search the decision threshold to maximize a chosen criterion.

    Args
    ----
    y_true:
        Binary labels {0,1}.
    y_score:
        Positive-class scores/probabilities.
    strategy:
        Optimization target: ``"f1"`` (default) or ``"youden"`` (Youden's J).

    Returns
    -------
    tuple[float, float]
        (best_threshold, best_value) for the selected strategy.

    Notes
    -----
    - Searches thresholds uniformly in [0.01, 0.99] (99 points).
    - F1 = 2PR / (P + R); Youden's J = TPR - FPR.
    """
    def _f1(thr: float) -> float:
        yhat = (y_score >= thr).astype(int)
        tp = np.sum((yhat == 1) & (y_true == 1))
        fp = np.sum((yhat == 1) & (y_true == 0))
        fn = np.sum((yhat == 0) & (y_true == 1))
        prec = tp / max(tp + fp, 1e-16)
        rec  = tp / max(tp + fn, 1e-16)
        return float(2 * prec * rec / max(prec + rec, 1e-16))

    def _youden(thr: float) -> float:
        yhat = (y_score >= thr).astype(int)
        tp = np.sum((yhat == 1) & (y_true == 1))
        tn = np.sum((yhat == 0) & (y_true == 0))
        fp = np.sum((yhat == 1) & (y_true == 0))
        fn = np.sum((yhat == 0) & (y_true == 1))
        tpr = tp / max(tp + fn, 1e-16)
        fpr = fp / max(fp + tn, 1e-16)
        return float(tpr - fpr)

    best_thr, best_val = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        val = _f1(thr) if strategy == "f1" else _youden(thr)
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val

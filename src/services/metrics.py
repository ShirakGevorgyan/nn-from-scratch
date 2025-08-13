# src/services/metrics.py
import math
import numpy as np
from typing import Dict, Tuple

def compute_metrics_from_probs(proba: np.ndarray, y: np.ndarray, thr: float = 0.5) -> Dict[str, float]:
    proba = proba.ravel()
    y = y.ravel().astype(int)
    yhat = (proba >= thr).astype(int)
    tp = int(np.sum((yhat == 1) & (y == 1)))
    tn = int(np.sum((yhat == 0) & (y == 0)))
    fp = int(np.sum((yhat == 1) & (y == 0)))
    fn = int(np.sum((yhat == 0) & (y == 1)))
    eps = 1e-8
    acc = (tp + tn) / (tp + tn + fp + fn + eps)
    prec = tp / (tp + fp + eps)
    rec  = tp / (tp + fn + eps)
    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = num / den
    return {"accuracy": float(acc), "precision": float(prec), "recall": float(rec), "mcc": float(mcc)}

def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray):
    order = np.argsort(-y_score)
    y_true = y_true.astype(int).ravel()[order]
    y_score = y_score.ravel()[order]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    return fps, tps, thresholds

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fns = tps[-1] - tps
    tns = (y_true.size - tps - fps)
    fpr = fps / np.maximum(fps + tns, 1e-16)
    tpr = tps / np.maximum(tps + fns, 1e-16)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))

def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    precision = tps / np.maximum(tps + fps, 1e-16)
    recall    = tps / np.maximum(tps[-1], 1e-16)
    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))

def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str = "f1") -> Tuple[float, float]:
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

"""
From-scratch neural network components, preprocessing, training loop,
metrics, and lightweight explainability helpers.

This module intentionally avoids external ML frameworks to keep every step
transparent for learning and auditing. It includes:

- Data utilities:
    * `train_val_test_split`: random split with fixed ratios.
    * Minimal preprocessors: `TargetEncoder`, `StandardScaler`, `MinMaxScaler`.

- Core NN building blocks:
    * `DenseLayer`, `ReLU`, `Sigmoid`, `Dropout`.

- Loss & metrics:
    * Binary cross-entropy (BCE) and its gradient.
    * Accuracy/Precision/Recall/MCC, ROC-AUC, PR-AUC, F1/Youden utilities.

- Optimizers & training:
    * `AdamOptimizer`, `AdamWOptimizer`, gradient clipping.
    * `ReduceLROnPlateau`, `EarlyStopping`, and `train_model`.

- Explainability (NumPy-based):
    * Permutation importance (MCC drop).
    * Partial dependence (PDP) for 1D feature sweeps.
    * Numerical `gradient_check` for backprop validation.
"""

import math
import numpy as np
import pandas as pd
from typing import Tuple, Dict, List, Any

# ---------------------------
# Data split / preprocessors
# ---------------------------

def train_val_test_split(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Shuffle and split a DataFrame into train/val/test partitions.

    Args:
        df: Input dataset.
        train_ratio: Fraction for the training split.
        val_ratio: Fraction for the validation split.
        seed: Random seed for reproducibility.

    Returns:
        (train_df, val_df, test_df) as three DataFrames whose row counts
        respect the requested ratios (remaining rows go to the test set).

    Notes:
        - The function shuffles the rows before slicing.
        - Ratios are applied sequentially on the shuffled data.
    """
    np.random.seed(seed)
    shuffled_df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_total = len(shuffled_df)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    train_df = shuffled_df.iloc[:n_train]
    val_df = shuffled_df.iloc[n_train : n_train + n_val]
    test_df = shuffled_df.iloc[n_train + n_val :]

    return train_df, val_df, test_df


class TargetEncoder:
    """Mean target encoder for categorical columns.

    Learns a mapping `category -> mean(target)` per categorical column and
    replaces categories with their learned means.

    Attributes:
        category_map: Dict[column_name, Dict[category_value, mean_target]]
    """
    def __init__(self) -> None:
        self.category_map: Dict[str, Dict[Any, float]] = {}

    def fit(self, df: pd.DataFrame, cat_cols: List[str], target_col: str) -> None:
        """Learn per-category mean of the target for each categorical column."""
        for col in cat_cols:
            cat2mean: Dict[Any, float] = {}
            grouped = df.groupby(col)[target_col].mean()
            for category_val, avg_target in grouped.items():
                cat2mean[category_val] = float(avg_target)
            self.category_map[col] = cat2mean

    def transform(self, df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
        """Replace categories with learned means; unseen categories -> global mean."""
        df_enc = df.copy()
        for col in cat_cols:
            cat2mean = self.category_map[col]
            global_mean = np.mean(list(cat2mean.values())) if len(cat2mean) else 0.0
            df_enc[col] = df_enc[col].apply(lambda x: cat2mean.get(x, global_mean))
        return df_enc

    def fit_transform(self, df: pd.DataFrame, cat_cols: List[str], target_col: str) -> pd.DataFrame:
        """Convenience: fit on `df` and return transformed DataFrame."""
        self.fit(df, cat_cols, target_col)
        return self.transform(df, cat_cols)


class StandardScaler:
    """Column-wise standardization: (x - mean) / std.

    Attributes:
        means: Dict[column_name, float]
        stds: Dict[column_name, float]  (replaced by 1e-8 if std == 0)
    """
    def __init__(self) -> None:
        self.means: Dict[str, float] = {}
        self.stds: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> None:
        """Compute per-column mean and std (with zero-std fallback)."""
        for col in cols:
            self.means[col] = float(df[col].mean())
            std_val = float(df[col].std())
            self.stds[col] = std_val if std_val != 0 else 1e-8

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply standardization to a copy of the DataFrame."""
        df_scaled = df.copy()
        for col in cols:
            df_scaled[col] = (df_scaled[col] - self.means[col]) / self.stds[col]
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Fit and then transform."""
        self.fit(df, cols)
        return self.transform(df, cols)


class MinMaxScaler:
    """Column-wise min-max scaling to [0, 1].

    Attributes:
        mins: Dict[column_name, float]
        maxs: Dict[column_name, float]
    """
    def __init__(self) -> None:
        self.mins: Dict[str, float] = {}
        self.maxs: Dict[str, float] = {}

    def fit(self, df: pd.DataFrame, cols: List[str]) -> None:
        """Compute per-column min and max."""
        for col in cols:
            self.mins[col] = float(df[col].min())
            self.maxs[col] = float(df[col].max())

    def transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Apply min-max scaling; uses 1e-8 denominator for zero-range columns."""
        df_scaled = df.copy()
        for col in cols:
            mn = self.mins[col]
            mx = self.maxs[col]
            denom = (mx - mn) if (mx - mn) != 0 else 1e-8
            df_scaled[col] = (df_scaled[col] - mn) / denom
        return df_scaled

    def fit_transform(self, df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
        """Fit and then transform."""
        self.fit(df, cols)
        return self.transform(df, cols)

# -------------
# Core layers
# -------------

class DenseLayer:
    """Fully-connected (affine) layer with simple moment buffers.

    Uses a Glorot/Xavier-like uniform initialization with limit 1/sqrt(input_dim).

    Attributes:
        W, b: parameters
        mW, vW, mb, vb: moment buffers (used by Adam/AdamW)
        dW, db: gradients computed during backprop
    """
    def __init__(self, input_dim: int, output_dim: int) -> None:
        # Xavier/Glorot uniform init (մոտ քո տարբերակին)
        limit = 1.0 / math.sqrt(input_dim)
        self.W = np.random.uniform(-limit, limit, (input_dim, output_dim))
        self.b = np.zeros((1, output_dim))

        # moments for optimizers
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

        self.dW = None
        self.db = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Compute affine transform: x @ W + b."""
        return x @ self.W + self.b

    def backward(self, x: np.ndarray, grad_output: np.ndarray) -> np.ndarray:
        """Backprop through the affine transform; returns dL/dx."""
        batch_size = x.shape[0]
        self.dW = (x.T @ grad_output) / max(batch_size, 1)
        self.db = grad_output.mean(axis=0, keepdims=True)
        grad_input = grad_output @ self.W.T
        return grad_input


class ReLU:
    """Rectified Linear Unit activation with saved input for backward."""
    def __init__(self) -> None:
        self.x = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply element-wise max(0, x)."""
        self.x = x
        return np.maximum(0, x)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Mask gradients where the cached input was negative."""
        grad = grad_output.copy()
        grad[self.x < 0] = 0
        return grad


class Sigmoid:
    """Sigmoid activation σ(x) = 1 / (1 + exp(-x))."""
    def __init__(self) -> None:
        self.out = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Cache σ(x) during forward to reuse in backward."""
        self.out = 1.0 / (1.0 + np.exp(-x))
        return self.out

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Return grad_output * σ(x) * (1 - σ(x))."""
        return grad_output * (self.out * (1 - self.out))

# ----------------------
# Regularization layer
# ----------------------

class Dropout:
    """Train-time dropout; disabled at evaluation.

    Args:
        p: Drop probability in [0, 1).
        seed: RNG seed for reproducibility.

    Notes:
        - During training, scales activations by 1/(1-p) to keep expectations.
    """
    def __init__(self, p: float = 0.5, seed: int = 42) -> None:
        assert 0.0 <= p < 1.0
        self.p = p
        self.rng = np.random.default_rng(seed)
        self.mask = None

    def forward(self, x: np.ndarray, train: bool = True) -> np.ndarray:
        """Apply dropout mask at train-time; return x unchanged at eval-time."""
        if not train or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (self.rng.random(x.shape) >= self.p).astype(x.dtype) / (1.0 - self.p)
        return x * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Propagate gradients through the same mask."""
        if self.mask is None:
            return grad_output
        return grad_output * self.mask

# -------------
# Loss & metrics
# -------------

def bce_loss(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> float:
    """Binary cross-entropy loss with epsilon clamping for stability."""
    pred_clamped = np.clip(pred, eps, 1 - eps)
    return -np.mean(
        target * np.log(pred_clamped) + (1 - target) * np.log(1 - pred_clamped)
    )

def bce_loss_grad(pred: np.ndarray, target: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Analytical gradient of BCE w.r.t. predictions (sigmoid outputs)."""
    pred_clamped = np.clip(pred, eps, 1 - eps)
    return (pred_clamped - target) / (pred_clamped * (1 - pred_clamped) + eps)

def compute_metrics(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    """Compute accuracy/precision/recall/MCC at a fixed probability threshold."""
    pred_binary = (pred >= threshold).astype(int)
    tp = np.sum((pred_binary == 1) & (target == 1))
    tn = np.sum((pred_binary == 0) & (target == 0))
    fp = np.sum((pred_binary == 1) & (target == 0))
    fn = np.sum((pred_binary == 0) & (target == 1))

    eps = 1e-8
    accuracy  = (tp + tn) / (tp + tn + fp + fn + eps)
    precision = tp / (tp + fp + eps)
    recall    = tp / (tp + fn + eps)
    numerator = (tp * tn) - (fp * fn)
    denominator = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + eps
    mcc = numerator / denominator

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "mcc": float(mcc),
    }

# ----------------------
# Extra curves & tuning
# ----------------------

def _binary_clf_curve(y_true: np.ndarray, y_score: np.ndarray):
    """Construct cumulative FP/TP vs descending thresholds (internal helper)."""
    order = np.argsort(-y_score)  # descending
    y_true = y_true.astype(int).ravel()[order]
    y_score = y_score.ravel()[order]
    distinct_value_indices = np.where(np.diff(y_score))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps
    thresholds = y_score[threshold_idxs]
    return fps, tps, thresholds

def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute ROC-AUC via trapezoidal rule (FPR vs TPR)."""
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    fns = tps[-1] - tps
    tns = (y_true.size - tps - fps)
    fpr = fps / np.maximum(fps + tns, 1e-16)
    tpr = tps / np.maximum(tps + fns, 1e-16)
    order = np.argsort(fpr)
    return float(np.trapz(tpr[order], fpr[order]))

def precision_recall_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Compute area under the Precision-Recall curve via trapezoidal rule."""
    fps, tps, _ = _binary_clf_curve(y_true, y_score)
    precision = tps / np.maximum(tps + fps, 1e-16)
    recall = tps / np.maximum(tps[-1], 1e-16)
    order = np.argsort(recall)
    return float(np.trapz(precision[order], recall[order]))

def f1_score_from_proba(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    """Compute F1 score after binarizing `y_score` at threshold `thr`."""
    y_pred = (y_score >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    precision = tp / max(tp + fp, 1e-16)
    recall = tp / max(tp + fn, 1e-16)
    return float(2 * precision * recall / max(precision + recall, 1e-16))

def youden_j(y_true: np.ndarray, y_score: np.ndarray, thr: float) -> float:
    """Compute Youden's J statistic (TPR - FPR) at threshold `thr`."""
    y_pred = (y_score >= thr).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    tpr = tp / max(tp + fn, 1e-16)
    fpr = fp / max(fp + tn, 1e-16)
    return float(tpr - fpr)

def tune_threshold(y_true: np.ndarray, y_score: np.ndarray, strategy: str = "f1") -> Tuple[float, float]:
    """Grid-search threshold in [0.01, 0.99] to maximize F1 or Youden's J.

    Returns:
        (best_threshold, best_value)
    """
    best_thr, best_val = 0.5, -1.0
    for thr in np.linspace(0.01, 0.99, 99):
        val = f1_score_from_proba(y_true, y_score, float(thr)) if strategy == "f1" else youden_j(y_true, y_score, float(thr))
        if val > best_val:
            best_val, best_thr = val, float(thr)
    return best_thr, best_val

# -----------------
# Model definition
# -----------------

class NeuralNetwork:
    """A simple 4-layer MLP for binary classification.

    Architecture:
        [Input] -> Dense(32) -> ReLU -> Dropout
                -> Dense(64) -> ReLU -> Dropout
                -> Dense(32) -> ReLU -> Dropout
                -> Dense(1)  -> Sigmoid

    Notes:
        - `self.training` toggles dropout behavior (True in `forward_and_backward`).
    """
    def __init__(self, input_dim: int, dropout_p: float = 0.15) -> None:
        self.layer1 = DenseLayer(input_dim, 32)
        self.act1 = ReLU()
        self.do1  = Dropout(dropout_p)

        self.layer2 = DenseLayer(32, 64)
        self.act2 = ReLU()
        self.do2  = Dropout(dropout_p)

        self.layer3 = DenseLayer(64, 32)
        self.act3 = ReLU()
        self.do3  = Dropout(dropout_p)

        self.layer4 = DenseLayer(32, 1)
        self.act4 = Sigmoid()

        # training flag for dropout behavior
        self.training = False

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Inference pass; uses current `self.training` for dropout."""
        # use self.training to toggle dropout
        out1 = self.layer1.forward(x)
        out1a = self.act1.forward(out1)
        out1a = self.do1.forward(out1a, train=self.training)

        out2 = self.layer2.forward(out1a)
        out2a = self.act2.forward(out2)
        out2a = self.do2.forward(out2a, train=self.training)

        out3 = self.layer3.forward(out2a)
        out3a = self.act3.forward(out3)
        out3a = self.do3.forward(out3a, train=self.training)

        out4 = self.layer4.forward(out3a)
        out4a = self.act4.forward(out4)
        return out4a

    def forward_and_backward(self, x: np.ndarray, y: np.ndarray) -> float:
        """Training step: forward pass with dropout ON, then backprop.

        Returns:
            Scalar BCE loss value for the given batch.
        """
        # Force training behavior (dropout ON here)
        prev_flag = self.training
        self.training = True

        out1 = self.layer1.forward(x)
        out1a = self.act1.forward(out1)
        out1a = self.do1.forward(out1a, train=True)

        out2 = self.layer2.forward(out1a)
        out2a = self.act2.forward(out2)
        out2a = self.do2.forward(out2a, train=True)

        out3 = self.layer3.forward(out2a)
        out3a = self.act3.forward(out3)
        out3a = self.do3.forward(out3a, train=True)

        out4 = self.layer4.forward(out3a)
        pred = self.act4.forward(out4)

        loss_val = bce_loss(pred, y)
        grad_pred = bce_loss_grad(pred, y)

        grad_out4 = self.act4.backward(grad_pred)
        grad_out3a = self.layer4.backward(out3a, grad_out4)
        grad_out3 = self.do3.backward(self.act3.backward(grad_out3a))
        grad_out2a = self.layer3.backward(out2a, grad_out3)
        grad_out2 = self.do2.backward(self.act2.backward(grad_out2a))
        grad_out1a = self.layer2.backward(out1a, grad_out2)
        _ = self.do1.backward(self.act1.backward(grad_out1a))
        _ = self.layer1.backward(x, _)

        self.training = prev_flag
        return loss_val

    def parameters(self) -> List[DenseLayer]:
        """Return the list of trainable layers (for optimizers)."""
        return [self.layer1, self.layer2, self.layer3, self.layer4]

# ---------------
# Optimizers
# ---------------

class AdamOptimizer:
    """Classic Adam optimizer for a list of `DenseLayer` parameters.

    Args:
        params: Layers to update (must expose `W`, `b`, `dW`, `db`, `mW`, `mb`, `vW`, `vb`).
        lr: Learning rate.
        beta1: Exponential decay for first moment.
        beta2: Exponential decay for second moment.
        eps: Numerical stability term.
    """
    def __init__(self, params: List[DenseLayer], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0

    def step(self) -> None:
        """Apply one Adam update step to all layers."""
        self.t += 1
        for layer in self.params:
            if layer.dW is None or layer.db is None:
                continue
            layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db

            layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)

            mW_hat = layer.mW / (1 - self.beta1 ** self.t)
            mb_hat = layer.mb / (1 - self.beta1 ** self.t)
            vW_hat = layer.vW / (1 - self.beta2 ** self.t)
            vb_hat = layer.vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps))
            layer.b -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))

class AdamWOptimizer:
    """Adam with decoupled weight decay (AdamW)."""
    def __init__(self, params: List[DenseLayer], lr: float = 1e-3, beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8, weight_decay: float = 1e-4) -> None:
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.t = 0

    def step(self) -> None:
        """Apply one AdamW update step to all layers (decoupled L2)."""
        self.t += 1
        for layer in self.params:
            if layer.dW is None or layer.db is None:
                continue
            layer.mW = self.beta1 * layer.mW + (1 - self.beta1) * layer.dW
            layer.mb = self.beta1 * layer.mb + (1 - self.beta1) * layer.db
            layer.vW = self.beta2 * layer.vW + (1 - self.beta2) * (layer.dW ** 2)
            layer.vb = self.beta2 * layer.vb + (1 - self.beta2) * (layer.db ** 2)

            mW_hat = layer.mW / (1 - self.beta1 ** self.t)
            mb_hat = layer.mb / (1 - self.beta1 ** self.t)
            vW_hat = layer.vW / (1 - self.beta2 ** self.t)
            vb_hat = layer.vb / (1 - self.beta2 ** self.t)

            layer.W -= self.lr * (mW_hat / (np.sqrt(vW_hat) + self.eps) + self.weight_decay * layer.W)
            layer.b -= self.lr * (mb_hat / (np.sqrt(vb_hat) + self.eps))

def clip_gradients(params: List[DenseLayer], max_norm: float = 1.0) -> None:
    """Global norm gradient clipping across all layers."""
    total = 0.0
    for layer in params:
        if layer.dW is not None and layer.db is not None:
            total += float(np.sum(layer.dW ** 2) + np.sum(layer.db ** 2))
    total = math.sqrt(max(total, 1e-16))
    scale = min(1.0, max_norm / total)
    if scale < 1.0:
        for layer in params:
            if layer.dW is not None and layer.db is not None:
                layer.dW *= scale
                layer.db *= scale

# -----------------------
# Scheduler / Early stop
# -----------------------

class ReduceLROnPlateau:
    """Reduce learning rate when a monitored metric has stopped improving.

    Args:
        optimizer: Optimizer whose `lr` will be adjusted.
        factor: Multiplicative factor for lr reduction.
        patience: Number of epochs with no improvement before reducing lr.
        min_lr: Lower bound for lr.
        verbose: Print changes when True.
    """
    def __init__(self, optimizer, factor: float = 0.5, patience: int = 3, min_lr: float = 1e-6, verbose: bool = True):
        self.optimizer = optimizer
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.verbose = verbose
        self.best = float("inf")
        self.num_bad = 0

    def step(self, current: float):
        """Update lr based on the current monitored value (e.g., val_loss)."""
        if current < self.best - 1e-12:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.optimizer.lr = max(self.optimizer.lr * self.factor, self.min_lr)
                if self.verbose:
                    print(f"[ReduceLROnPlateau] lr -> {self.optimizer.lr:.6f}")
                self.num_bad = 0

class EarlyStopping:
    """Stop training early when the monitored metric stops improving.

    Args:
        patience: Epochs to wait for improvement.
        min_delta: Minimum decrease to qualify as an improvement.
    """
    def __init__(self, patience: int = 5, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.num_bad = 0
        self.stop = False

    def step(self, current: float):
        """Update internal counters and toggle `stop` when patience exceeded."""
        if current < self.best - self.min_delta:
            self.best = current
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.stop = True

# ----------------
# Training loop
# ----------------

def train_model(
    model: NeuralNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    use_adamw: bool = True,
    weight_decay: float = 1e-4,
    grad_clip: float = 1.0,
    es_patience: int = 5,
    sch_patience: int = 2,
    sch_factor: float = 0.5,
) -> None:
    """Train the network with mini-batch optimization and callbacks.

    Workflow per epoch:
        1) Shuffle training data and iterate mini-batches.
        2) Forward+backward with dropout ON; clip gradients; optimizer.step().
        3) Evaluate on validation set (loss, ROC-AUC, PR-AUC, F1-tuned threshold).
        4) Print metrics; then scheduler/early-stopping callbacks.

    Args:
        model: Neural network instance.
        X_train, y_train: Training arrays.
        X_val, y_val: Validation arrays.
        epochs: Max epochs to train.
        batch_size: Mini-batch size.
        lr: Base learning rate.
        use_adamw: Whether to use AdamW instead of Adam.
        weight_decay: L2 decay factor for AdamW.
        grad_clip: Global norm clipping threshold (None to disable).
        es_patience: EarlyStopping patience (in epochs).
        sch_patience: ReduceLROnPlateau patience.
        sch_factor: LR reduction factor.

    Returns:
        None. Progress is printed; the model is updated in-place.
    """
    optimizer = AdamWOptimizer(model.parameters(), lr=lr, weight_decay=weight_decay) if use_adamw \
                else AdamOptimizer(model.parameters(), lr=lr)
    scheduler = ReduceLROnPlateau(optimizer, factor=sch_factor, patience=sch_patience, min_lr=1e-6, verbose=True)
    earlystop = EarlyStopping(patience=es_patience, min_delta=1e-4)

    n_samples = X_train.shape[0]
    n_batches = int(math.ceil(n_samples / batch_size))

    for epoch in range(epochs):
        # train
        perm = np.random.permutation(n_samples)
        X_train_shuffled = X_train[perm]
        y_train_shuffled = y_train[perm]

        epoch_loss = 0.0
        model.training = True
        for b in range(n_batches):
            start = b * batch_size
            end = start + batch_size
            x_batch = X_train_shuffled[start:end]
            y_batch = y_train_shuffled[start:end]

            loss = model.forward_and_backward(x_batch, y_batch)

            if grad_clip is not None:
                clip_gradients(model.parameters(), max_norm=float(grad_clip))

            optimizer.step()
            epoch_loss += loss

        avg_loss = epoch_loss / max(n_batches, 1)

        # validate
        model.training = False
        val_proba = model.forward(X_val).ravel()
        val_loss = bce_loss(val_proba.reshape(-1, 1), y_val.reshape(-1, 1))
        auc_roc = roc_auc_score(y_val, val_proba)
        auc_pr  = precision_recall_auc(y_val, val_proba)
        best_thr_f1, best_f1 = tune_threshold(y_val, val_proba, strategy="f1")

        metrics_default_thr = compute_metrics(val_proba.reshape(-1,1), y_val.reshape(-1,1), threshold=0.5)
        metrics_best_thr    = compute_metrics(val_proba.reshape(-1,1), y_val.reshape(-1,1), threshold=best_thr_f1)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"AUC-ROC: {auc_roc:.3f} | AUC-PR: {auc_pr:.3f} | "
            f"F1*thr={best_thr_f1:.2f} (F1={best_f1:.3f}) | "
            f"Acc@0.5: {metrics_default_thr['accuracy']:.3f} | "
            f"MCC@0.5: {metrics_default_thr['mcc']:.3f} | "
            f"Acc@* : {metrics_best_thr['accuracy']:.3f} | "
            f"MCC@* : {metrics_best_thr['mcc']:.3f}"
        )

        # callbacks
        scheduler.step(val_loss)
        earlystop.step(val_loss)
        if earlystop.stop:
            print("[EarlyStopping] Training stopped early.")
            break


# ============================================
# Explainability & Validation helpers (from scratch)
# ============================================

def _score_mcc(y_true: np.ndarray, y_score: np.ndarray, thr: float = 0.5) -> float:
    """Compute MCC after binarizing probabilities at threshold `thr`."""
    y_true = y_true.astype(int).ravel()
    y_pred = (y_score.ravel() >= thr).astype(int)

    tp = np.sum((y_pred == 1) & (y_true == 1))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))

    num = (tp * tn) - (fp * fn)
    den = math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) + 1e-16
    return float(num / den)


def permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    baseline_thr: float = 0.5,
    n_repeats: int = 5,
    random_seed: int = 42
) -> np.ndarray:
    """Permutation importance via mean MCC drop over `n_repeats` shuffles.

    Args:
        model: Network exposing `forward(X) -> proba`.
        X: 2D feature matrix.
        y: Binary labels aligned to rows of `X`.
        baseline_thr: Threshold for MCC computation.
        n_repeats: Number of independent permutations per feature.
        random_seed: RNG seed.

    Returns:
        Importance array of shape (n_features,), larger means more important.
    """
    rng = np.random.default_rng(random_seed)

    # Baseline on intact validation data
    base_proba = model.forward(X).ravel()
    base_score = _score_mcc(y, base_proba, thr=baseline_thr)

    importances = np.zeros(X.shape[1], dtype=float)

    for j in range(X.shape[1]):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])  # permute one feature
            proba = model.forward(Xp).ravel()
            score = _score_mcc(y, proba, thr=baseline_thr)
            drops.append(base_score - score)
        importances[j] = float(np.mean(drops))

    return importances


def partial_dependence(
    model,
    X: np.ndarray,
    feature_idx: int,
    grid: np.ndarray | None = None,
    grid_size: int = 20
) -> tuple[np.ndarray, np.ndarray]:
    """Compute 1D PDP for a feature index by sweeping a value grid.

    Args:
        model: Network exposing `forward(X) -> proba`.
        X: Input matrix used as the background.
        feature_idx: Column index to vary.
        grid: Optional explicit grid; if None, uses 1st..99th percentiles.
        grid_size: Number of points if `grid` is None.

    Returns:
        (grid_values, mean_predictions) where predictions are averaged
        over rows at each grid value.
    """
    if grid is None:
        vmin, vmax = np.percentile(X[:, feature_idx], [1, 99])
        grid = np.linspace(vmin, vmax, grid_size)

    means = []
    Xc = X.copy()
    for g in grid:
        Xc[:, feature_idx] = g
        p = model.forward(Xc).ravel()
        means.append(float(np.mean(p)))

    return grid, np.array(means)


def gradient_check(
    model,
    X: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-5,
    num_checks: int = 10,
    seed: int = 0
) -> dict[str, float]:
    """Numerical gradient checking via central differences on random params.

    For randomly selected entries in each parameter tensor (W and b),
    perturb by ±eps and compare the numerical gradient with backprop.

    Args:
        model: Network exposing `forward` and `forward_and_backward`.
        X, y: Mini-batch and labels.
        eps: Small step for finite differences.
        num_checks: How many entries per tensor to test.
        seed: RNG seed for reproducibility.

    Returns:
        {'max_rel_error': float, 'mean_rel_error': float}
    """
    rng = np.random.default_rng(seed)

    # Make sure shapes are (N,1) for loss fn
    y_col = y.reshape(-1, 1)

    # Run one backward pass to populate dW/db
    _ = model.forward_and_backward(X, y_col)

    rel_errors: list[float] = []

    for layer in model.parameters():
        for pname in ("W", "b"):
            P = getattr(layer, pname)      # weights or bias
            dP = getattr(layer, "d" + pname)  # gradient from backprop

            # Choose random indices
            if P.ndim == 2:
                idx_i = rng.integers(0, P.shape[0], size=num_checks)
                idx_j = rng.integers(0, P.shape[1], size=num_checks)
                indices = list(zip(idx_i, idx_j))
            else:
                # bias has shape (1, D)
                idx_j = rng.integers(0, P.shape[1], size=num_checks)
                indices = [(0, j) for j in idx_j]

            for i, j in indices:
                old = P[i, j] if P.ndim == 2 else P[0, j]

                # f(theta + eps)
                if P.ndim == 2:
                    P[i, j] = old + eps
                else:
                    P[0, j] = old + eps
                pred_plus = model.forward(X)
                loss_plus = bce_loss(pred_plus, y_col)

                # f(theta - eps)
                if P.ndim == 2:
                    P[i, j] = old - eps
                else:
                    P[0, j] = old - eps
                pred_minus = model.forward(X)
                loss_minus = bce_loss(pred_minus, y_col)

                # restore
                if P.ndim == 2:
                    P[i, j] = old
                    grad_backprop = dP[i, j]
                else:
                    P[0, j] = old
                    grad_backprop = dP[0, j]

                grad_num = (loss_plus - loss_minus) / (2.0 * eps)
                rel = abs(grad_num - grad_backprop) / max(1e-8, abs(grad_num) + abs(grad_backprop))
                rel_errors.append(float(rel))

    return {
        "max_rel_error": float(np.max(rel_errors)),
        "mean_rel_error": float(np.mean(rel_errors)),
    }

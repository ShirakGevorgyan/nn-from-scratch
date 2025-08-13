# src/services/explain.py
from typing import List, Dict, Optional, Tuple
import numpy as np
import pandas as pd

from .artifacts import ModelBundle
from .metrics import compute_metrics_from_probs


def permutation_importance(
    bundle: ModelBundle,
    df: pd.DataFrame,
    y: np.ndarray,
    baseline_thr: float = 0.5,
    n_repeats: int = 5,
    random_seed: int = 42,
) -> Tuple[List[str], np.ndarray, float]:
    """
    Simple permutation importance using MCC drop.
    Returns (columns, importances, base_mcc).
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame.")
    if y is None or len(y) == 0:
        raise ValueError("Empty labels array.")
    if not (0.0 <= baseline_thr <= 1.0):
        raise ValueError("baseline_thr must be in [0, 1].")
    if n_repeats <= 0:
        n_repeats = 1

    rng = np.random.default_rng(random_seed)

    X = bundle.align_columns(df)
    y = np.asarray(y).astype(int).ravel()

    base_proba = bundle.predict_proba(X)
    base_mcc = compute_metrics_from_probs(base_proba, y, thr=baseline_thr)["mcc"]

    cols = list(X.columns)
    importances = np.zeros(len(cols), dtype=float)

    for j, col in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            Xp[col] = Xp[col].values[rng.permutation(len(Xp))]
            proba = bundle.predict_proba(Xp)
            mcc = compute_metrics_from_probs(proba, y, thr=baseline_thr)["mcc"]
            drops.append(base_mcc - mcc)
        importances[j] = float(np.mean(drops))

    return cols, importances, float(base_mcc)


def partial_dependence(
    bundle: ModelBundle,
    df: pd.DataFrame,
    feature: str,
    grid: Optional[List[float]] = None,
    grid_size: int = 20,
    ice: bool = False,
    ice_count: int = 10,
    seed: int = 42,
) -> Dict[str, object]:
    """
    Basic PDP (and optional ICE) for a single feature.
    Returns dict with keys: feature, grid, pdp, (optional) ice.
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame.")

    X = bundle.align_columns(df)
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(X.columns)}")

    col = X[feature].astype(float).values

    if grid is None or len(grid) == 0:
        vmin, vmax = np.percentile(col, [1, 99])
        if vmin == vmax:
            # degenerate case: make a tiny grid around value
            vmin, vmax = float(vmin) - 1e-6, float(vmax) + 1e-6
        grid = list(np.linspace(float(vmin), float(vmax), int(max(grid_size, 2))))

    # PDP
    pdp_vals: List[float] = []
    X_tmp = X.copy()
    for g in grid:
        X_tmp[feature] = g
        proba = bundle.predict_proba(X_tmp)
        pdp_vals.append(float(np.mean(proba)))

    result: Dict[str, object] = {
        "feature": feature,
        "grid": [float(v) for v in grid],
        "pdp": pdp_vals,
    }

    # ICE (simple, take up to ice_count rows)
    if ice:
        rng = np.random.default_rng(seed)
        n = len(X)
        take = min(int(ice_count), n)
        idxs = rng.choice(n, size=take, replace=False)
        curves = []
        for i in idxs:
            row = X.iloc[[int(i)]].copy()
            curve = []
            for g in grid:
                row_mod = row.copy()
                row_mod[feature] = g
                proba = bundle.predict_proba(row_mod)
                curve.append(float(np.mean(proba)))
            curves.append({"row_index": int(i), "curve": curve})
        result["ice"] = curves

    return result

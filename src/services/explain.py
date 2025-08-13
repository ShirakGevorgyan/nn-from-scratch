# src/services/explain.py
"""
Explainability utilities for the Heart Disease Detector.

This module provides two model-agnostic explainability primitives:

1) Permutation Feature Importance (PFI)
   - Measures the expected performance drop (here: MCC) when shuffling a
     single feature, breaking its relationship with the target.
   - Returns the baseline MCC and a per-feature importance vector
     (mean MCC drop across shuffles).

2) Partial Dependence (PDP) and optional ICE
   - PDP estimates the marginal effect of a single feature on the model's
     output by averaging predictions over a grid, while holding other
     features fixed (at their observed values).
   - ICE (Individual Conditional Expectation) draws per-instance curves
     over the same grid to visualize heterogeneity across samples.

Notes
-----
- Both functions assume the provided `ModelBundle` exposes:
    * `align_columns(df: pd.DataFrame) -> pd.DataFrame`
    * `predict_proba(df: pd.DataFrame) -> np.ndarray`
- Metrics are computed via the service layer (see `.metrics`).
- No randomness affects baseline scores; RNG is used only for permutations
  (PFI) and sampling ICE rows.
"""

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
    Compute permutation feature importance (PFI) using MCC drop.

    The importance of a feature is defined as the expected decrease in
    Matthews Correlation Coefficient (MCC) after randomly shuffling that
    feature and recomputing predictions. Repeating the shuffle multiple
    times stabilizes the estimate.

    Args
    ----
    bundle:
        Model + preprocessing wrapper exposing `align_columns` and `predict_proba`.
    df:
        Input features as a DataFrame. Must contain all columns expected by
        the bundle.
    y:
        Ground-truth binary labels (0/1), aligned with `df` rows.
    baseline_thr:
        Threshold used to compute the baseline MCC (default: 0.5).
    n_repeats:
        Number of independent shuffles per feature (default: 5).
    random_seed:
        Seed for the permutation RNG to ensure reproducibility.

    Returns
    -------
    tuple[list[str], np.ndarray, float]
        A triple `(columns, importances, base_mcc)` where:
        - `columns` is the ordered list of feature names,
        - `importances` is an array of mean MCC drops per feature,
        - `base_mcc` is the MCC on the unshuffled data at `baseline_thr`.

    Raises
    ------
    ValueError
        If `df`/`y` are empty or `baseline_thr` is outside [0, 1].

    Notes
    -----
    - If `n_repeats <= 0`, the function coerces it to 1 (single shuffle).
    - MCC is computed via `compute_metrics_from_probs` in the metrics service.
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame.")
    if y is None or len(y) == 0:
        raise ValueError("Empty labels array.")
    if not (0.0 <= baseline_thr <= 1.0):
        raise ValueError("baseline_thr must be in [0, 1].")
    if n_repeats <= 0:
        n_repeats = 1

    # Deterministic RNG for reproducible permutations
    rng = np.random.default_rng(random_seed)

    # Ensure column order/typing matches what the bundle expects
    X = bundle.align_columns(df)
    y = np.asarray(y).astype(int).ravel()

    # Baseline performance (no shuffling)
    base_proba = bundle.predict_proba(X)
    base_mcc = compute_metrics_from_probs(base_proba, y, thr=baseline_thr)["mcc"]

    cols = list(X.columns)
    importances = np.zeros(len(cols), dtype=float)

    # For each feature, shuffle its values `n_repeats` times and
    # record the mean MCC drop relative to the baseline
    for j, col in enumerate(cols):
        drops = []
        for _ in range(n_repeats):
            Xp = X.copy()
            # Permute the column values, breaking dependency with target
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
    Compute 1D Partial Dependence (PDP) and optional ICE for a given feature.

    PDP is computed by sweeping the feature across a grid of values while
    keeping all other features fixed at their observed values; predictions
    are then averaged across rows at each grid point. ICE curves depict the
    same sweep for individual rows to highlight heterogeneity.

    Args
    ----
    bundle:
        Model + preprocessing wrapper exposing `align_columns` and `predict_proba`.
    df:
        Background data used to average out other features (must contain `feature`).
    feature:
        Feature name for which to compute PDP/ICE.
    grid:
        Optional explicit grid values for the feature. If not provided, a grid
        is generated between the 1st and 99th percentiles of the feature.
    grid_size:
        Number of grid points when `grid` is not provided (min 2).
    ice:
        If True, compute ICE curves for a subsample of rows.
    ice_count:
        Number of rows to sample for ICE (capped at dataset size).
    seed:
        RNG seed for ICE row sampling.

    Returns
    -------
    dict
        A dictionary with keys:
        - "feature": str, the feature name,
        - "grid": list[float], the grid values,
        - "pdp": list[float], mean predicted probabilities at each grid value,
        - "ice": optional list[{"row_index": int, "curve": list[float]}].

    Raises
    ------
    ValueError
        If `df` is empty or `feature` is not present in the columns.

    Notes
    -----
    - Predictions are averaged directly over the model's positive-class
      probabilities (i.e., not thresholded).
    - The percentile-based grid avoids extreme outliers; for constant
      features, a tiny neighborhood grid is created around the value.
    """
    if df is None or len(df) == 0:
        raise ValueError("Empty DataFrame.")

    # Align and validate columns
    X = bundle.align_columns(df)
    if feature not in X.columns:
        raise ValueError(f"Feature '{feature}' not found. Available: {list(X.columns)}")

    col = X[feature].astype(float).values

    # Build grid if not provided: use robust percentiles to avoid outliers
    if grid is None or len(grid) == 0:
        vmin, vmax = np.percentile(col, [1, 99])
        if vmin == vmax:
            # Degenerate case (constant feature): expand a tiny epsilon band
            vmin, vmax = float(vmin) - 1e-6, float(vmax) + 1e-6
        grid = list(np.linspace(float(vmin), float(vmax), int(max(grid_size, 2))))

    # PDP: for each grid value, overwrite the column and average predictions
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

    # ICE: sample up to `ice_count` rows and record their individual curves
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
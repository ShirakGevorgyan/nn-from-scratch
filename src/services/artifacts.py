"""
Artifact loading and runtime compatibility helpers.

This module is responsible for:
- Locating the serialized artifacts on disk (``model/`` directory).
- Loading the trained model and preprocessing components from **pickle** files.
- Providing a lightweight ``ModelBundle`` wrapper that exposes a consistent
  interface to the API/service layers:
    * ``align_columns(df) -> pd.DataFrame``
    * ``predict_proba(df) -> np.ndarray`` (positive-class probabilities)

Design notes
------------
- Only the pickle path is supported in this project (no NPZ/JSON variants).
- We insert the project ``SRC`` directory into ``sys.path`` and normalize
  the ``nn_from_scratch`` import so that older notebooks / pickles that
  referenced ``src.nn_from_scratch`` continue to work.
- ``_ensure_runtime_compat`` makes older pickles forward-compatible by
  attaching missing attributes (e.g., ``training``, dropout layers).

Directory layout
----------------
BASE_DIR/
  ├─ src/
  │   └─ services/
  │       └─ artifacts.py   (this file)
  └─ model/
      ├─ model.pkl
      └─ preprocessing.pkl
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nn_from_scratch import Dropout

# Resolve project directories and ensure imports work across environments.
SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Backward-compat import shim:
# Some serialized artifacts may refer to `src.nn_from_scratch`. We alias both
# names to the same loaded module so unpickling and runtime imports succeed.
try:
    import nn_from_scratch as nn_mod
    sys.modules["nn_from_scratch"] = nn_mod
    sys.modules["src.nn_from_scratch"] = nn_mod
except Exception:
    # Best-effort shim; if import fails here, loading will error later anyway.
    pass


BASE_DIR = SRC_DIR.parent            
MODEL_DIR = BASE_DIR / "model"
MODEL_PKL = MODEL_DIR / "model.pkl"
PREP_PKL  = MODEL_DIR / "preprocessing.pkl"


class ModelBundle:
    """Container for the trained model and its preprocessing steps.

    Attributes
    ----------
    model : object | None
        Trained model that exposes a `forward(x: np.ndarray) -> np.ndarray`.
    encoder : object | None
        Optional categorical encoder with `transform(df, cols)` method.
    std_scaler : object | None
        Optional standard scaler with `transform(df, cols)` method.
    mm_scaler : object | None
        Optional min-max scaler with `transform(df, cols)` method.
    cat_cols : list[str]
        Names of categorical input columns expected by `encoder`.
    num_cols : list[str]
        Names of numerical columns for standard scaling.
    float_cols : list[str]
        Names of float columns for min-max scaling.
    feature_order : list[str]
        Final feature ordering expected by the model; falls back to
        `[cat_cols + num_cols + float_cols]` if empty.
    source : str
        A short tag identifying the artifact backend, e.g., "pickle".
    """
    def __init__(self) -> None:
        self.model = None
        self.encoder = None
        self.std_scaler = None
        self.mm_scaler = None
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.float_cols: list[str] = []
        self.feature_order: list[str] = [] 
        self.source = "pickle"

    def align_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Ensure the DataFrame has all expected columns in the right order.

        - Missing columns are created and filled with zeros.
        - All values are coerced to numeric and NaNs are filled with 0.0.

        Parameters
        ----------
        df : pd.DataFrame
            Input features.

        Returns
        -------
        pd.DataFrame
            A new DataFrame that matches the expected model input layout.
        """
        order = self.feature_order or (self.cat_cols + self.num_cols + self.float_cols)
        aligned = df.copy()

        for c in order:
            if c not in aligned.columns:
                aligned[c] = 0
                
        aligned = aligned[order]
        aligned = aligned.apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return aligned

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        """
        Applies (encoder -> std -> minmax) if available, then runs model.forward().
        Returns probabilities as (N,) numpy array of float.

        Pipeline
        --------
        1) `align_columns` to fix order and backfill missing columns.
        2) If present:
           - `encoder.transform(df, cat_cols)`
           - `std_scaler.transform(df, num_cols)`
           - `mm_scaler.transform(df, float_cols)`
        3) Extract numpy array and call `model.forward(x)`, then ravel to (N,).

        Notes
        -----
        - The returned values are positive-class probabilities suitable for
          thresholding or averaging.
        """
        df = self.align_columns(df)

        if self.encoder is not None and self.cat_cols:
            df = self.encoder.transform(df, self.cat_cols)
        if self.std_scaler is not None and self.num_cols:
            df = self.std_scaler.transform(df, self.num_cols)
        if self.mm_scaler is not None and self.float_cols:
            df = self.mm_scaler.transform(df, self.float_cols)

        order = self.feature_order or (self.cat_cols + self.num_cols + self.float_cols)
        df = df[order]

        x = df.values.astype(float)
        proba = self.model.forward(x).ravel()
        return proba


def _exists(p: Path) -> bool:
    """Safely check whether a path exists, guarding against OS/errors."""
    try:
        return p.exists()
    except Exception:
        return False


def _ensure_runtime_compat(model) -> None:
    """Compat for old pickles that lack dropout/training attrs.

    Older serialized models might miss certain runtime attributes expected by
    the current code. This helper injects:
    - `training` flag defaulting to `False`
    - no-op `Dropout` layers for attributes `do1`, `do2`, `do3`
    """
    if not hasattr(model, "training"):
        model.training = False
    for name in ("do1", "do2", "do3"):
        if not hasattr(model, name):
            setattr(model, name, Dropout(0.0))  # no-op


def _load_from_pickle() -> ModelBundle:
    """Load model + preprocessing artifacts from pickle files.

    Expects two files under `model/`:
    - `model.pkl`           : trained model object
    - `preprocessing.pkl`   : dict with encoders/scalers/column lists

    Returns
    -------
    ModelBundle
        A populated bundle with model, preprocessors, and metadata.

    Raises
    ------
    FileNotFoundError
        If either of the required pickle files is missing.
    """
    import pickle

    if not (_exists(MODEL_PKL) and _exists(PREP_PKL)):
        raise FileNotFoundError("Pickle artifacts not found (model.pkl / preprocessing.pkl).")

    with open(MODEL_PKL, "rb") as f:
        mdl = pickle.load(f)
    _ensure_runtime_compat(mdl)

    with open(PREP_PKL, "rb") as f:
        artifacts = pickle.load(f)

    b = ModelBundle()
    b.model       = mdl
    b.encoder     = artifacts.get("encoder")     or artifacts.get("final_encoder")
    b.std_scaler  = artifacts.get("std_scaler")  or artifacts.get("final_std_scaler")
    b.mm_scaler   = artifacts.get("mm_scaler")   or artifacts.get("final_mm_scaler")
    b.cat_cols    = artifacts.get("cat_cols", [])
    b.num_cols    = artifacts.get("num_cols", [])
    b.float_cols  = artifacts.get("float_cols", [])
    b.feature_order = artifacts.get("feature_order", b.cat_cols + b.num_cols + b.float_cols)
    b.source = "pickle"
    return b


def load_bundle() -> ModelBundle:
    """Always load pickle artifacts (no NPZ/JSON).

    Returns
    -------
    ModelBundle
        A bundle ready for inference/explainability pipelines.
    """
    return _load_from_pickle()

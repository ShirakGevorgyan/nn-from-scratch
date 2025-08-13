# src/services/artifacts.py
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from nn_from_scratch import Dropout

SRC_DIR = Path(__file__).resolve().parents[1]
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    import nn_from_scratch as nn_mod
    sys.modules["nn_from_scratch"] = nn_mod
    sys.modules["src.nn_from_scratch"] = nn_mod
except Exception:
    pass


BASE_DIR = SRC_DIR.parent            
MODEL_DIR = BASE_DIR / "model"
MODEL_PKL = MODEL_DIR / "model.pkl"
PREP_PKL  = MODEL_DIR / "preprocessing.pkl"


class ModelBundle:
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
    try:
        return p.exists()
    except Exception:
        return False


def _ensure_runtime_compat(model) -> None:
    """Compat for old pickles that lack dropout/training attrs."""
    if not hasattr(model, "training"):
        model.training = False
    for name in ("do1", "do2", "do3"):
        if not hasattr(model, name):
            setattr(model, name, Dropout(0.0))  # no-op


def _load_from_pickle() -> ModelBundle:
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
    """Always load pickle artifacts (no NPZ/JSON)."""
    return _load_from_pickle()


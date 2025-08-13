"""
Heart Disease Detector API.

This module exposes a FastAPI application that serves a from-scratch neural
network model for heart-disease risk prediction. It loads a serialized
"bundle" of artifacts (preprocessing + model) and provides:

Endpoints
---------
- GET  `/`                  : Liveness/health check.
- GET  `/version`           : App + artifact version info.
- GET  `/feature-map`       : Feature mapping exposed by the bundle.
- POST `/predict`           : Predict probabilities (and optional labels).
- POST `/metrics`           : Compute evaluation metrics on labeled data.
- POST `/explain/permutation` : Permutation feature importance (global).
- POST `/explain/pdp`         : Partial dependence (and optional ICE).

Notes
-----
- Input validation is handled by Pydantic models defined here and in
  ``.schemas``.
- The artifacts are loaded via ``services.artifacts.load_bundle`` and must
  implement a ``predict_proba(pd.DataFrame) -> np.ndarray`` interface.
- No business logic is implemented in the API layer; the API delegates to
  the service layer for metrics and explainability routines.
"""

from fastapi import FastAPI, HTTPException, Query
from typing import List, Union, Optional, Dict, Any, Annotated

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

from .schemas import HeartDiseaseRequest

# Services
from .services.artifacts import load_bundle
from .services.metrics import compute_metrics_from_probs, roc_auc, pr_auc, tune_threshold
from .services.explain import (
    permutation_importance as svc_perm,
    partial_dependence as svc_pdp,
)

APP_VERSION = "0.3"

# Instantiate the FastAPI app with descriptive metadata for the OpenAPI schema.
app = FastAPI(
    title="Heart Disease Detector API",
    version=APP_VERSION,
    description="API for predicting heart disease risk using a from-scratch neural network",
)

# -----------------------------
# Pydantic models (API schemas)
# -----------------------------

class LabeledHeartDiseaseRequest(HeartDiseaseRequest):
    """Input row for supervised evaluation endpoints.

    Inherits all feature fields from ``HeartDiseaseRequest`` and adds the
    binary target label.

    Attributes
    ----------
    target:
        Ground-truth label of the positive class. Must be 0 or 1.
    """
    target: Annotated[int, Field(ge=0, le=1)]


class TopItem(BaseModel):
    """Container for a single (feature, importance) pair.

    Attributes
    ----------
    feature:
        Column name as used by the model bundle.
    importance:
        Importance score (e.g., mean MCC drop) for the feature.
    """
    feature: str
    importance: float


class PermutationRequest(BaseModel):
    """Request payload for permutation feature importance.

    Attributes
    ----------
    data:
        Labeled rows used to compute baseline performance and shuffled
        counterfactuals.
    baseline_thr:
        Classification threshold to compute the baseline MCC.
    n_repeats:
        Number of shuffles per feature to estimate importance stability.
    top_k:
        (Optional) If provided, the response will include the top-K features
        by importance as a convenience summary.
    random_seed:
        Seed used to make the shuffling deterministic/reproducible.
    """
    data: List[LabeledHeartDiseaseRequest]
    baseline_thr: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    n_repeats: int = 5
    top_k: Optional[int] = Field(default=None, ge=1)
    random_seed: int = 42


class PermutationResponse(BaseModel):
    """Response schema for permutation importance results.

    Attributes
    ----------
    base_mcc:
        Baseline Matthews correlation coefficient at ``baseline_thr``.
    columns:
        List of feature names in the order used to compute importances.
    importances:
        List of importance scores aligned with ``columns``.
    top:
        (Optional) A compact top-K list (feature + score) if requested.
    """
    base_mcc: float
    columns: List[str]
    importances: List[float]
    top: Optional[List[TopItem]] = None


class PDPRequest(BaseModel):
    """Request payload for Partial Dependence (and optional ICE) computation.

    Attributes
    ----------
    data:
        Unlabeled rows used as background to average out other features.
    feature:
        The target feature for which to compute PDP/ICE.
    grid:
        (Optional) Explicit grid values for the target feature. If omitted,
        a grid is generated automatically.
    grid_size:
        Number of grid points to generate if ``grid`` is not provided.
    ice:
        Whether to include ICE (individual conditional expectation) curves.
    ice_count:
        Number of individual instances to sample for ICE curves (if enabled).
    seed:
        Random seed for reproducible sampling.
    """
    data: List[HeartDiseaseRequest]
    feature: str
    grid: Optional[List[float]] = None
    grid_size: int = 20
    ice: bool = False
    ice_count: int = 10
    seed: int = 42


class PDPResponse(BaseModel):
    """Response schema for PDP (and optional ICE) results.

    Attributes
    ----------
    feature:
        Name of the feature analyzed.
    grid:
        Grid values used to evaluate the model on the feature.
    pdp:
        Partial dependence values corresponding to ``grid``.
    ice:
        (Optional) List of per-instance curves, each stored as a small dict
        with metadata and values (shape and keys are determined in the
        service layer).
    """
    feature: str
    grid: List[float]
    pdp: List[float]
    ice: Optional[List[Dict[str, Any]]] = None


# -----------------
# Load model bundle
# -----------------
try:
    # The bundle encapsulates preprocessing + model and exposes predict_proba.
    BUNDLE = load_bundle()  # pickle-only (configured in services.artifacts)
except Exception as e:
    # Fail fast if artifacts are unavailable/misaligned with the codebase.
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# -----------
# Endpoints
# -----------

@app.get("/")
async def health_check():
    """Liveness probe and minimal environment info.

    Returns
    -------
    dict
        A small payload with app version, status and artifact source.
    """
    return {"version": APP_VERSION, "status": "OK", "artifact_source": BUNDLE.source}


@app.get("/version")
async def version():
    """Return application version and artifact source.

    Returns
    -------
    dict
        Keys: ``app_version`` and ``artifact_source``.
    """
    return {"app_version": APP_VERSION, "artifact_source": BUNDLE.source}


@app.get("/feature-map")
async def feature_map():
    """Expose the feature mapping used by the underlying model bundle.

    Returns
    -------
    dict
        Keys: ``cat_cols``, ``num_cols``, ``float_cols``.
    """
    return {
        "cat_cols": BUNDLE.cat_cols,
        "num_cols": BUNDLE.num_cols,
        "float_cols": BUNDLE.float_cols,
    }


@app.post("/predict")
async def predict(
    data: Union[HeartDiseaseRequest, List[HeartDiseaseRequest]],
    threshold: Annotated[Optional[float], Query(ge=0.0, le=1.0)] = None,
):
    """Predict positive-class probabilities (and optional hard labels).

    Parameters
    ----------
    data:
        A single instance or a list of instances following
        ``HeartDiseaseRequest``.
    threshold:
        If provided, probabilities are binarized into labels using this
        threshold and both are returned.

    Returns
    -------
    dict
        Always includes ``predictions`` (list of floats). If ``threshold`` is
        provided, also returns ``labels`` and echoes back ``threshold``.

    Raises
    ------
    HTTPException
        With status 400 if validation/inference fails.
    """
    try:
        records: List[Dict[str, Any]] = (
            [d.dict() for d in data] if isinstance(data, list) else [data.dict()]
        )
        df = pd.DataFrame(records)
        proba = BUNDLE.predict_proba(df).astype(float).tolist()

        if threshold is None:
            return {"predictions": proba}

        labels = (np.array(proba) >= float(threshold)).astype(int).tolist()
        return {"predictions": proba, "labels": labels, "threshold": float(threshold)}

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during prediction: {str(e)}")


@app.post("/metrics")
async def metrics(payload: List[LabeledHeartDiseaseRequest]):
    """Compute evaluation metrics given labeled examples.

    The routine computes metrics at a default threshold (0.5) and at an
    automatically tuned threshold (by F1), and reports ROC-AUC and PR-AUC.

    Parameters
    ----------
    payload:
        List of labeled rows (features + binary target).

    Returns
    -------
    dict
        Contains the tuned threshold, metrics at default and optimal
        thresholds, and scalar AUC scores.

    Raises
    ------
    HTTPException
        With status 400 if the payload is empty or computation fails.
    """
    try:
        if not payload:
            raise ValueError("Empty payload.")
        records = [d.dict() for d in payload]
        y = np.array([r.pop("target") for r in records], dtype=int)
        df = pd.DataFrame(records)

        proba = BUNDLE.predict_proba(df).astype(float)
        res_default = compute_metrics_from_probs(proba, y, thr=0.5)
        thr_opt, _ = tune_threshold(y, proba, strategy="f1")
        res_opt = compute_metrics_from_probs(proba, y, thr=thr_opt)

        return {
            "threshold_used": float(thr_opt),
            "metrics_at_default": res_default,
            "metrics_at_optimal": res_opt,
            "auc_roc": float(roc_auc(y, proba)),
            "auc_pr": float(pr_auc(y, proba)),
        }

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error during metrics: {str(e)}")


@app.post("/explain/permutation", response_model=PermutationResponse)
async def explain_permutation(payload: PermutationRequest):
    """Compute permutation feature importance over labeled data.

    The importance is derived from the expected drop in a baseline metric
    (e.g., MCC) when shuffling a given feature.

    Parameters
    ----------
    payload:
        See ``PermutationRequest`` for fields and constraints.

    Returns
    -------
    PermutationResponse
        Base MCC, full vectors of (columns, importances), and an optional top-K.

    Raises
    ------
    HTTPException
        With status 400 if input is empty or computation fails.
    """
    try:
        if not payload.data:
            raise ValueError("Empty 'data'.")

        recs = [d.dict() for d in payload.data]
        y = np.array([r.pop("target") for r in recs], dtype=int)
        df = pd.DataFrame(recs)

        cols, imps, base_mcc = svc_perm(
            BUNDLE,
            df,
            y,
            baseline_thr=float(payload.baseline_thr),
            n_repeats=int(payload.n_repeats),
            random_seed=int(payload.random_seed),
        )

        top_out: Optional[List[TopItem]] = None
        if payload.top_k:
            k = int(min(payload.top_k, len(cols)))
            order = np.argsort(-imps)[:k]
            top_out = [TopItem(feature=cols[i], importance=float(imps[i])) for i in order]

        return PermutationResponse(
            base_mcc=float(base_mcc),
            columns=cols,
            importances=[float(v) for v in imps],
            top=top_out,
        )

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in permutation importance: {e}")


@app.post("/explain/pdp", response_model=PDPResponse)
async def explain_pdp(payload: PDPRequest):
    """Compute Partial Dependence (and optional ICE) for a single feature.

    Parameters
    ----------
    payload:
        See ``PDPRequest`` for parameters controlling grid generation, ICE,
        and sampling.

    Returns
    -------
    PDPResponse
        The evaluated grid, PDP values, and (optionally) ICE curves.

    Raises
    ------
    HTTPException
        With status 400 if input is empty or computation fails.
    """
    try:
        if not payload.data:
            raise ValueError("Empty 'data'.")

        df = pd.DataFrame([d.dict() for d in payload.data])

        res = svc_pdp(
            BUNDLE,
            df,
            feature=payload.feature,
            grid=payload.grid,
            grid_size=int(payload.grid_size),
            ice=bool(payload.ice),
            ice_count=int(payload.ice_count),
            seed=int(payload.seed),
        )
        return PDPResponse(**res)

    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error in PDP/ICE: {e}")

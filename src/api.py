# src/api.py
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

app = FastAPI(
    title="Heart Disease Detector API",
    version=APP_VERSION,
    description="API for predicting heart disease risk using a from-scratch neural network",
)

# -----------------------------
# Pydantic models (API schemas)
# -----------------------------

class LabeledHeartDiseaseRequest(HeartDiseaseRequest):
    target: Annotated[int, Field(ge=0, le=1)]

class TopItem(BaseModel):
    feature: str
    importance: float

class PermutationRequest(BaseModel):
    data: List[LabeledHeartDiseaseRequest]
    baseline_thr: Annotated[float, Field(ge=0.0, le=1.0)] = 0.5
    n_repeats: int = 5
    top_k: Optional[int] = Field(default=None, ge=1)
    random_seed: int = 42

class PermutationResponse(BaseModel):
    base_mcc: float
    columns: List[str]
    importances: List[float]
    top: Optional[List[TopItem]] = None

class PDPRequest(BaseModel):
    data: List[HeartDiseaseRequest]
    feature: str
    grid: Optional[List[float]] = None
    grid_size: int = 20
    ice: bool = False
    ice_count: int = 10
    seed: int = 42

class PDPResponse(BaseModel):
    feature: str
    grid: List[float]
    pdp: List[float]
    ice: Optional[List[Dict[str, Any]]] = None

# -----------------
# Load model bundle
# -----------------
try:
    BUNDLE = load_bundle()  # pickle-only (configured in services.artifacts)
except Exception as e:
    raise RuntimeError(f"Failed to load model artifacts: {e}")

# -----------
# Endpoints
# -----------

@app.get("/")
async def health_check():
    return {"version": APP_VERSION, "status": "OK", "artifact_source": BUNDLE.source}

@app.get("/version")
async def version():
    return {"app_version": APP_VERSION, "artifact_source": BUNDLE.source}

@app.get("/feature-map")
async def feature_map():
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

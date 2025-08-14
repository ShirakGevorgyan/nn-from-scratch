# Heart Disease Detector API

A production-ready **FastAPI** service that predicts heart disease risk using a **from‑scratch NumPy neural network**, with built‑in metrics and explainability (Permutation Importance & PDP/ICE). The model and preprocessing live in portable pickle artifacts, so you can train elsewhere and serve here.

> **Disclaimer:** This project is for educational/demo purposes only and **not** for clinical use.

---

## ✨ Features

- **REST API** (FastAPI) for predictions and evaluation
- **From‑scratch NN** (NumPy): Dense layers, ReLU/Sigmoid, Dropout, Adam/AdamW
- **Metrics**: Accuracy, Precision, Recall, MCC, ROC‑AUC, PR‑AUC, threshold tuning (F1 / Youden)
- **Explainability**:
  - **Permutation Importance** (MCC drop)
  - **Partial Dependence** + optional **ICE** curves
- **Portable artifacts**: `model/model.pkl` + `model/preprocessing.pkl`
- **Dockerized**: one simple image for quick sharing and deployment

---

## 🗂 Repository Structure

```
.
├─ data/heart_disease.csv                 # (sample data; not required at runtime)
├─ notebook/heart_disease_analysis.ipynb  # notebooks (not required at runtime)
├─ model/                                 # <mount or bake> model artifacts here
│  ├─ model.pkl                           # trained model object
│  └─ preprocessing.pkl                   # dict(encoders/scalers/cols...)
├─ src/
│  ├─ api.py                              # FastAPI app + endpoints
│  ├─ nn_from_scratch.py                  # NN, training loop, preprocessors
│  ├─ schemas.py                          # Pydantic request schemas
│  └─ services/
│     ├─ artifacts.py                     # ModelBundle loader (pickle-only)
│     ├─ explain.py                       # Permutation importance, PDP/ICE
│     └─ metrics.py                       # Metrics + threshold tuning
├─ requirements.txt
├─ pyproject.toml
├─ Dockerfile
├─ README.md
└─ .gitignore
```

---

## 🚀 Quickstart

### Option A — Docker (recommended for sharing)

1) Put your artifacts in `./model/`:
   - `model/model.pkl`
   - `model/preprocessing.pkl`

2) Build and run:
```bash
docker build -t heart-api .
docker run --rm -p 8000:8000 -v "$PWD/model:/app/model:ro" heart-api
```

3) Open the docs: http://localhost:8000/docs  
   (or: `curl http://localhost:8000/version`)

> **Docker Compose**
>
> ```yaml
> services:
>   api:
>     build: .
>     image: heart-api
>     ports:
>       - "8000:8000"
>     volumes:
>       - ./model:/app/model:ro
> ```
> Run with: `docker compose up --build`

---

### Option B — Local (Python)

Requirements are pinned for **Pydantic v1** compatibility (project uses `.dict()`):

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src:$PYTHONPATH                # Windows (PowerShell): $env:PYTHONPATH="$PWD/src;$env:PYTHONPATH"
uvicorn src.api:app --reload --port 8000
```

Place artifacts at `./model/` before hitting the API.

---

## ⚙️ Model Artifacts

Artifacts are loaded by `src/services/artifacts.py` (pickle only). Expected keys in `preprocessing.pkl`:

- `encoder` (or `final_encoder`) – optional categorical encoder with `.transform(df, cols)`
- `std_scaler` (or `final_std_scaler`) – optional standard scaler for numeric cols
- `mm_scaler` (or `final_mm_scaler`) – optional min‑max scaler for float cols
- `cat_cols`, `num_cols`, `float_cols` – column lists
- `feature_order` (optional) – final feature order expected by the model

> The **ModelBundle** aligns columns and calls `model.forward(x)` to get positive‑class probabilities.

---

## 🔌 API Reference

Base URL: `http://localhost:8000`  
Interactive docs: `/docs` (Swagger), `/redoc` (ReDoc)

### `GET /` — Health
Returns `{ "version": "...", "status": "OK", "artifact_source": "pickle" }`

### `GET /version`
```json
{ "app_version": "0.3", "artifact_source": "pickle" }
```

### `GET /feature-map`
Lists feature groups used by the bundle:
```json
{
  "cat_cols": ["sex","cp","fbs","restecg","exang","slope"],
  "num_cols": ["age","trestbps","chol","thalach"],
  "float_cols": ["oldpeak"]
}
```

### `POST /predict`
Predict probabilities (and optional labels if `threshold` provided).

- **Query**: `threshold` ∈ [0,1] (optional)
- **Body**: `HeartDiseaseRequest` or `List[HeartDiseaseRequest]`

`HeartDiseaseRequest` fields:
```json
{
  "age": 57, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250,
  "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1,
  "oldpeak": 1.2, "slope": 2
}
```

**Example**
```bash
curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" -d '{"age":57,"sex":1,"cp":0,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":140,"exang":1,"oldpeak":1.2,"slope":2}'
```
**Response**
```json
{ "predictions":[0.74], "labels":[1], "threshold":0.5 }
```

### `POST /metrics`
Compute metrics on labeled rows.

- **Body**: `List[LabeledHeartDiseaseRequest]` where each item = features + `"target": 0|1`

**Returns**
```json
{
  "threshold_used": 0.37,
  "metrics_at_default": { "accuracy": 0.91, "precision": 0.88, "recall": 0.85, "mcc": 0.73 },
  "metrics_at_optimal": { "accuracy": 0.92, "precision": 0.90, "recall": 0.86, "mcc": 0.75 },
  "auc_roc": 0.91,
  "auc_pr": 0.88
}

```

### `POST /explain/permutation`
Permutation Feature Importance (global).

**Body** (`PermutationRequest`):
```json
{
  "baseline_thr": 0.5,
  "n_repeats": 5,
  "top_k": 10,
  "random_seed": 42,
  "data": [ { "<features>", "target": 0|1 }, ... ]
}
```

**Response**
```json
{
  "base_mcc": 0.42,
  "columns": ["sex", "cp", "fbs", "restecg", "exang", "slope", "age", "trestbps", "chol", "thalach", "oldpeak"],
  "importances": [0.02, 0.00, 0.01, 0.00, 0.00, 0.05, 0.03, 0.00, 0.00, 0.00, 0.00],
  "top": [
    { "feature": "slope", "importance": 0.05 },
    { "feature": "age", "importance": 0.03 },
    { "feature": "sex", "importance": 0.02 }
  ]
}
```

### `POST /explain/pdp`
Partial Dependence for a single feature (optionally with ICE curves).

**Body** (`PDPRequest`):
```json
{
  "feature": "age",
  "grid": null,
  "grid_size": 20,
  "ice": false,
  "ice_count": 10,
  "seed": 42,
  "data": [
    { "age": 57, "sex": 1, "cp": 0, "trestbps": 130, "chol": 250, "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1, "oldpeak": 1.2, "slope": 2 },
    { "age": 52, "sex": 0, "cp": 2, "trestbps": 120, "chol": 210, "fbs": 0, "restecg": 0, "thalach": 160, "exang": 0, "oldpeak": 0.4, "slope": 1 }
  ]
}
```
**Response**
```json
{
  "feature": "age",
  "grid": [52.0, 52.4, 52.8, 53.2],
  "pdp": [0.489, 0.491, 0.494, 0.496],
  "ice": null
}
```

---

## 🧠 From‑Scratch Model (overview)

The model is defined in `src/nn_from_scratch.py`:

- **Architecture**
  - Dense(32) → ReLU → Dropout
  - Dense(64) → ReLU → Dropout
  - Dense(32) → ReLU → Dropout
  - Dense(1)  → Sigmoid

- **Training utilities**
  - `train_val_test_split`, `TargetEncoder`, `StandardScaler`, `MinMaxScaler`
  - `bce_loss` (+ gradient), `Adam` / `AdamW`, gradient clipping
  - `ReduceLROnPlateau`, `EarlyStopping`, `train_model(...)`

> The API itself does **not** train. Train offline, export artifacts, and serve.

---

## 🔧 Configuration

- **Port / workers (Dockerfile defaults)**: port `8000`, workers `2`
- **Change port** with Docker: map host port, e.g. `-p 8001:8000`
- **Artifacts**: mount `./model` at `/app/model` (or bake them into the image)

---

## 🧪 Smoke Tests

```bash
curl http://localhost:8000/version
curl http://localhost:8000/feature-map

curl -X POST "http://localhost:8000/predict?threshold=0.5" \
  -H "Content-Type: application/json" \
  -d '{"age":57,"sex":1,"cp":0,"trestbps":130,"chol":250,"fbs":0,"restecg":1,"thalach":140,"exang":1,"oldpeak":1.2,"slope":2}'
```

---

## 🐞 Troubleshooting

- **`address already in use` on port 8000**  
  Another process/container is using it. Stop it or map a different host port, e.g. `-p 8001:8000` (compose: `"8001:8000"`).

- **`FileNotFoundError: Pickle artifacts not found`**  
  Place `model.pkl` & `preprocessing.pkl` in `./model/` (mounted to `/app/model`).

- **`ImportError: nn_from_scratch`**  
  The image sets `PYTHONPATH=/app/src:/app`. For local dev, export `PYTHONPATH=$PWD/src:$PYTHONPATH`.

- **Odd AUC/metrics with tiny payloads**  
  Curves need enough points. Use larger evaluation data.

---

## 📄 License & Attribution

This project is released under the MIT License (see `LICENSE`).

### Dataset Attribution
This project uses (or derives from) the **UCI Machine Learning Repository – Heart Disease** dataset (commonly the *Cleveland* subset).

Please cite:
- Dua, D. & Graff, C. (2019). UCI Machine Learning Repository. University of California, Irvine, School of Information and Computer Sciences. https://archive.ics.uci.edu
- Original dataset contributors include R. Detrano and collaborators.

> Data is for research/education only. This software is **not** for clinical use.


---

## 🙌 Acknowledgements

Thanks to everyone who likes learning by building from scratch 🤝
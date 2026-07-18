"""API de serving do modelo — FastAPI + Pydantic v2.

O modelo (Pipeline sklearn: scaler + RandomForest) é carregado uma única vez
no startup via lifespan. Em runtime a API é independente do MLflow server:
consome o artefato exportado em models/model.joblib (ver ADR e ARCHITECTURE.md).
"""

from __future__ import annotations

import os
import time
import warnings
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException

from api.schemas import FEATURE_ORDER, PredictionRequest, PredictionResponse

MODEL_PATH = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
VERSION_PATH = MODEL_PATH.parent / "MODEL_VERSION"

# O Pipeline foi treinado com um DataFrame nomeado; sem pandas em runtime,
# a predicao usa um array numpy na mesma ordem (FEATURE_ORDER) e o sklearn
# avisa que os nomes de coluna nao foram passados — aviso esperado, sem
# impacto no resultado.
warnings.filterwarnings(
    "ignore", message="X does not have valid feature names", category=UserWarning
)

state: dict[str, Any] = {"model": None, "version": "unknown"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not MODEL_PATH.exists():
        raise RuntimeError(
            f"Modelo não encontrado em {MODEL_PATH}. Rode `make tune` antes do build."
        )
    state["model"] = joblib.load(MODEL_PATH)
    if VERSION_PATH.exists():
        state["version"] = VERSION_PATH.read_text().strip()
    yield
    state["model"] = None


app = FastAPI(
    title="Breast Cancer Classifier API",
    version="1.0.0",
    description="Serving do pipeline MLOps (Breast Cancer Wisconsin).",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict[str, str]:
    """Usado pelas probes de liveness/readiness do Kubernetes."""
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")
    return {"status": "ok", "model_version": state["version"]}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: PredictionRequest) -> PredictionResponse:
    if state["model"] is None:
        raise HTTPException(status_code=503, detail="modelo não carregado")

    t0 = time.perf_counter()
    row = [getattr(payload, name.replace(" ", "_")) for name in FEATURE_ORDER]
    features = np.array([row], dtype=float)
    proba_maligno = float(state["model"].predict_proba(features)[0, 1])
    label = int(proba_maligno >= 0.5)
    latency_ms = (time.perf_counter() - t0) * 1000

    return PredictionResponse(
        prediction=label,
        diagnosis="maligno" if label == 1 else "benigno",
        probability_malignant=round(proba_maligno, 6),
        model_version=state["version"],
        latency_ms=round(latency_ms, 3),
    )

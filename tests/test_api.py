"""Testes da API com TestClient — usa um modelo mínimo treinado on-the-fly."""

import joblib
import pytest
from fastapi.testclient import TestClient
from sklearn.linear_model import LogisticRegression

from api.schemas import FEATURE_ORDER
from mlops_bc.data import make_splits
from mlops_bc.train import build_pipeline


@pytest.fixture(scope="module")
def client(tmp_path_factory, monkeypatch_module=None):
    # Treina um modelo rápido e aponta MODEL_PATH para ele antes de importar o app
    model_dir = tmp_path_factory.mktemp("models")
    x_train, _, y_train, _ = make_splits()
    pipe = build_pipeline(LogisticRegression(max_iter=5000, random_state=42))
    pipe.fit(x_train, y_train)
    joblib.dump(pipe, model_dir / "model.joblib")
    (model_dir / "MODEL_VERSION").write_text("test:v0\n")

    import os

    os.environ["MODEL_PATH"] = str(model_dir / "model.joblib")
    # Reimporta o módulo para reavaliar MODEL_PATH
    import importlib

    import api.main as main

    importlib.reload(main)
    with TestClient(main.app) as c:
        yield c


def _benign_payload() -> dict:
    """Valores próximos à mediana de um caso benigno do dataset."""
    base = {
        "mean radius": 12.0, "mean texture": 18.0, "mean perimeter": 78.0,
        "mean area": 450.0, "mean smoothness": 0.09, "mean compactness": 0.08,
        "mean concavity": 0.04, "mean concave points": 0.025, "mean symmetry": 0.17,
        "mean fractal dimension": 0.062, "radius error": 0.3, "texture error": 1.1,
        "perimeter error": 2.0, "area error": 23.0, "smoothness error": 0.006,
        "compactness error": 0.02, "concavity error": 0.025,
        "concave points error": 0.01, "symmetry error": 0.02,
        "fractal dimension error": 0.003, "worst radius": 13.5,
        "worst texture": 24.0, "worst perimeter": 87.0, "worst area": 550.0,
        "worst smoothness": 0.12, "worst compactness": 0.2, "worst concavity": 0.16,
        "worst concave points": 0.08, "worst symmetry": 0.28,
        "worst fractal dimension": 0.08,
    }
    assert set(base) == set(FEATURE_ORDER)
    return base


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_predict_benign_case(client):
    r = client.post("/predict", json=_benign_payload())
    assert r.status_code == 200
    body = r.json()
    assert body["prediction"] in (0, 1)
    assert 0.0 <= body["probability_malignant"] <= 1.0
    assert body["diagnosis"] in ("benigno", "maligno")
    assert body["latency_ms"] >= 0


def test_predict_missing_feature_is_422(client):
    payload = _benign_payload()
    payload.pop("mean radius")
    r = client.post("/predict", json=payload)
    assert r.status_code == 422


def test_predict_negative_value_is_422(client):
    payload = _benign_payload()
    payload["mean radius"] = -1.0
    r = client.post("/predict", json=payload)
    assert r.status_code == 422

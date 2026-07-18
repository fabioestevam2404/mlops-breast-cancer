"""Testes de modelagem: pipeline, métricas e contrato do artefato."""

import numpy as np
from sklearn.linear_model import LogisticRegression

from mlops_bc.data import make_splits
from mlops_bc.train import build_pipeline, compute_metrics


def test_pipeline_has_scaler_and_model():
    pipe = build_pipeline(LogisticRegression(max_iter=5000))
    assert list(pipe.named_steps) == ["scaler", "model"]


def test_pipeline_trains_and_beats_chance():
    x_train, x_test, y_train, y_test = make_splits()
    pipe = build_pipeline(LogisticRegression(max_iter=5000, random_state=42))
    pipe.fit(x_train, y_train)
    metrics = compute_metrics(
        y_test.to_numpy(), pipe.predict(x_test), pipe.predict_proba(x_test)[:, 1]
    )
    assert set(metrics) == {"accuracy", "precision", "recall", "f1", "roc_auc"}
    assert metrics["recall"] > 0.9
    assert metrics["roc_auc"] > 0.95


def test_metrics_perfect_prediction():
    y = np.array([0, 1, 0, 1])
    m = compute_metrics(y, y, y.astype(float))
    assert all(v == 1.0 for v in m.values())


def test_train_and_log_records_run(tmp_path):
    """Integração leve: train_and_log deve criar run com métricas no MLflow."""
    import mlflow

    from mlops_bc.train import train_and_log

    mlflow.set_tracking_uri(f"sqlite:///{tmp_path}/test.db")
    mlflow.set_experiment("test-exp")
    x_train, x_test, y_train, y_test = make_splits()
    metrics = train_and_log(
        "logreg-test",
        LogisticRegression(max_iter=5000, random_state=42),
        x_train, y_train.to_numpy(), x_test, y_test.to_numpy(),
    )
    assert metrics["roc_auc"] > 0.95
    runs = mlflow.search_runs(experiment_names=["test-exp"])
    assert len(runs) == 1
    assert runs.iloc[0]["metrics.recall"] == metrics["recall"]

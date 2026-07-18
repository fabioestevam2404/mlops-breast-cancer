"""Treino: Logistic Regression (baseline) vs Random Forest, tudo logado no MLflow."""

from __future__ import annotations

from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from mlops_bc.config import (
    EXPERIMENT_BASELINE,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    REPORTS_DIR,
)
from mlops_bc.data import load_processed


def build_pipeline(model: Any) -> Pipeline:
    """Scaler + modelo em um único artefato (evita training-serving skew)."""
    return Pipeline([("scaler", StandardScaler()), ("model", model)])


def compute_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> dict[str, float]:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }


def log_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> str:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    ConfusionMatrixDisplay(cm, display_labels=["benigno", "maligno"]).plot(ax=ax, colorbar=False)
    ax.set_title(f"Matriz de Confusão — {name}")
    path = REPORTS_DIR / f"cm_{name}.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return str(path)


def train_and_log(
    name: str,
    model: Any,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_test: pd.DataFrame,
    y_test: np.ndarray,
) -> dict[str, float]:
    pipe = build_pipeline(model)
    with mlflow.start_run(run_name=name):
        pipe.fit(x_train, y_train)
        y_pred = pipe.predict(x_test)
        y_proba = pipe.predict_proba(x_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params({f"model__{k}": v for k, v in model.get_params().items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(log_confusion_matrix(y_test, y_pred, name))
        mlflow.sklearn.log_model(pipe, artifact_path="model", input_example=x_train.iloc[:2])

        print(f"[train] {name}: " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))
    return metrics


def main() -> None:  # pragma: no cover - coberto pelo smoke test do CI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_BASELINE)
    x_train, x_test, y_train, y_test = load_processed()

    train_and_log(
        "logreg-baseline",
        LogisticRegression(max_iter=5000, random_state=RANDOM_STATE),
        x_train, y_train, x_test, y_test,
    )
    train_and_log(
        "random-forest-default",
        RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
        x_train, y_train, x_test, y_test,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

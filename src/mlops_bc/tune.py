"""Tuning de hiperparâmetros do Random Forest.

Compara RandomizedSearchCV vs HalvingGridSearchCV (scoring = recall, ADR-004),
loga tudo no MLflow e registra o campeão no Model Registry com alias `champion`.
Também exporta o Pipeline vencedor para models/model.joblib (consumido pela imagem Docker).
"""

from __future__ import annotations

import time

import joblib
import mlflow
from sklearn.ensemble import RandomForestClassifier

# HalvingGridSearchCV é experimental — o enable DEVE vir antes do import
from sklearn.experimental import enable_halving_search_cv  # noqa: F401
from sklearn.model_selection import HalvingGridSearchCV, RandomizedSearchCV, StratifiedKFold

from mlops_bc.config import (
    CHAMPION_ALIAS,
    EXPERIMENT_TUNING,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    REGISTERED_MODEL_NAME,
    TUNING_SCORING,
)
from mlops_bc.data import load_processed
from mlops_bc.train import build_pipeline, compute_metrics, log_confusion_matrix

PARAM_DISTRIBUTIONS = {
    "model__n_estimators": [100, 200, 300, 500],
    "model__max_depth": [None, 5, 10, 20],
    "model__min_samples_split": [2, 5, 10],
    "model__min_samples_leaf": [1, 2, 4],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": [None, "balanced"],
}

# Grid menor para o Halving (successive halving já poda combinações fracas)
PARAM_GRID_HALVING = {
    "model__n_estimators": [100, 300, 500],
    "model__max_depth": [None, 10, 20],
    "model__min_samples_split": [2, 5],
    "model__max_features": ["sqrt", "log2"],
    "model__class_weight": [None, "balanced"],
}


def run_search(name: str, search, x_train, y_train, x_test, y_test) -> dict:
    """Executa um buscador, avalia no teste e loga o run completo no MLflow."""
    with mlflow.start_run(run_name=name) as run:
        t0 = time.perf_counter()
        search.fit(x_train, y_train)
        elapsed = time.perf_counter() - t0

        best = search.best_estimator_
        y_pred = best.predict(x_test)
        y_proba = best.predict_proba(x_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(search.best_params_)
        mlflow.log_param("search_strategy", name)
        mlflow.log_metric("cv_best_recall", float(search.best_score_))
        mlflow.log_metric("search_time_s", elapsed)
        mlflow.log_metrics({f"test_{k}": v for k, v in metrics.items()})
        mlflow.log_artifact(log_confusion_matrix(y_test, y_pred, name))
        mlflow.sklearn.log_model(best, artifact_path="model", input_example=x_train.iloc[:2])

        run_id = run.info.run_id
        print(
            f"[tune] {name}: cv_recall={search.best_score_:.4f} "
            f"test_recall={metrics['recall']:.4f} test_f1={metrics['f1']:.4f} "
            f"tempo={elapsed:.1f}s"
        )
        return {
            "run_id": run_id,
            "cv_best_recall": float(search.best_score_),
            "metrics": metrics,
            "estimator": best,
            "time_s": elapsed,
        }


def main() -> None:  # pragma: no cover - coberto pelo smoke test do CI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_TUNING)
    x_train, x_test, y_train, y_test = load_processed()

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    pipe = build_pipeline(RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1))

    random_search = RandomizedSearchCV(
        pipe,
        PARAM_DISTRIBUTIONS,
        n_iter=40,
        scoring=TUNING_SCORING,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    halving_search = HalvingGridSearchCV(
        pipe,
        PARAM_GRID_HALVING,
        scoring=TUNING_SCORING,
        cv=cv,
        factor=3,
        min_resources=120,  # evita folds iniciais sem amostras da classe positiva
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    results = {
        "randomized-search": run_search(
            "randomized-search", random_search, x_train, y_train, x_test, y_test
        ),
        "halving-grid-search": run_search(
            "halving-grid-search", halving_search, x_train, y_train, x_test, y_test
        ),
    }

    # Campeão = melhor recall em validação cruzada (nunca decidir com base no teste,
    # senão o teste deixa de ser uma estimativa não-enviesada da performance final).
    winner_name = max(results, key=lambda k: results[k]["cv_best_recall"])
    winner = results[winner_name]
    print(
        f"[tune] campeão: {winner_name} "
        f"(cv_recall={winner['cv_best_recall']:.4f}, test_recall={winner['metrics']['recall']:.4f})"
    )

    # Registro no Model Registry + alias champion
    model_uri = f"runs:/{winner['run_id']}/model"
    mv = mlflow.register_model(model_uri, REGISTERED_MODEL_NAME)
    client = mlflow.MlflowClient()
    client.set_registered_model_alias(REGISTERED_MODEL_NAME, CHAMPION_ALIAS, mv.version)
    print(f"[tune] registrado: {REGISTERED_MODEL_NAME} v{mv.version} @ {CHAMPION_ALIAS}")

    # Export para a imagem Docker (runtime independente do MLflow server)
    export_path = MODELS_DIR / "model.joblib"
    joblib.dump(winner["estimator"], export_path)
    (MODELS_DIR / "MODEL_VERSION").write_text(f"{REGISTERED_MODEL_NAME}:v{mv.version}\n")
    print(f"[tune] exportado: {export_path}")


if __name__ == "__main__":  # pragma: no cover
    main()

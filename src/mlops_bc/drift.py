"""Monitoramento: simulação de data drift + log de métricas de produção no MLflow.

Simula lotes de produção com shift gaussiano progressivo nas features, mede o
Population Stability Index (PSI) contra a distribuição de treino e loga a
degradação das métricas do modelo — o gatilho clássico de retraining.

Interpretação padrão do PSI:
  < 0.10  → estável
  0.10–0.25 → atenção
  > 0.25  → drift significativo (retreinar)
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import joblib
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

from mlops_bc.config import (
    EXPERIMENT_MONITORING,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
)
from mlops_bc.data import load_processed
from mlops_bc.train import compute_metrics

PSI_ALERT = 0.25


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10) -> float:
    """PSI entre duas amostras 1-D usando quantis da distribuição esperada."""
    edges = np.quantile(expected, np.linspace(0, 1, bins + 1))
    edges[0], edges[-1] = -np.inf, np.inf
    e_pct = np.clip(np.histogram(expected, edges)[0] / len(expected), 1e-6, None)
    a_pct = np.clip(np.histogram(actual, edges)[0] / len(actual), 1e-6, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def simulate_drift(
    x: pd.DataFrame, shift_std: float, rng: np.random.Generator
) -> pd.DataFrame:
    """Aplica shift gaussiano proporcional ao desvio padrão de cada feature."""
    noise = rng.normal(loc=shift_std, scale=0.05, size=x.shape) * x.std(axis=0).to_numpy()
    return x + noise


def main() -> None:  # pragma: no cover - coberto pelo smoke test do CI
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_MONITORING)

    model = joblib.load(MODELS_DIR / "model.joblib")
    x_train, x_test, _, y_test = load_processed()
    rng = np.random.default_rng(RANDOM_STATE)

    # 6 "semanas" de produção com drift progressivo
    shifts = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
    history: list[dict] = []

    for week, shift in enumerate(shifts, start=1):
        x_prod = simulate_drift(x_test, shift, rng)
        y_pred = model.predict(x_prod)
        y_proba = model.predict_proba(x_prod)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)
        psis = [psi(x_train[c].to_numpy(), x_prod[c].to_numpy()) for c in x_train.columns]
        mean_psi = float(np.mean(psis))

        with mlflow.start_run(run_name=f"prod-week-{week:02d}"):
            mlflow.log_param("shift_std", shift)
            mlflow.log_metrics(metrics)
            mlflow.log_metric("psi_mean", mean_psi)
            mlflow.log_metric("drift_alert", float(mean_psi > PSI_ALERT))

        history.append({"semana": week, "shift": shift, "psi_medio": mean_psi, **metrics})
        alerta = "[ALERTA] RETREINAR" if mean_psi > PSI_ALERT else "[ok]"
        print(
            f"[drift] semana {week}: shift={shift:.2f} psi={mean_psi:.3f} "
            f"recall={metrics['recall']:.3f} f1={metrics['f1']:.3f} {alerta}"
        )

    df = pd.DataFrame(history)
    df.to_csv(REPORTS_DIR / "drift_report.csv", index=False)

    fig, ax1 = plt.subplots(figsize=(8, 4.5))
    ax1.plot(df["semana"], df["recall"], "o-", color="#00c9a7", label="recall")
    ax1.plot(df["semana"], df["f1"], "s-", color="#3b82f6", label="f1")
    ax1.set_xlabel("Semana de produção (simulada)")
    ax1.set_ylabel("Métrica do modelo")
    ax1.legend(loc="lower left")
    ax2 = ax1.twinx()
    ax2.plot(df["semana"], df["psi_medio"], "^--", color="#d4a843", label="PSI médio")
    ax2.axhline(PSI_ALERT, color="#ef4444", ls=":", label=f"limite PSI {PSI_ALERT}")
    ax2.set_ylabel("PSI médio")
    ax2.legend(loc="upper left")
    ax1.set_title("Degradação do modelo sob data drift simulado")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "drift_report.png", dpi=120)
    print(f"[drift] relatório salvo em {REPORTS_DIR}")


if __name__ == "__main__":  # pragma: no cover
    main()

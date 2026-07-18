"""Configuração central do projeto. Nenhum outro módulo define caminhos ou constantes."""

from pathlib import Path

# Raiz do projeto (src/mlops_bc/config.py -> 2 níveis acima)
ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

# MLflow — SQLite é obrigatório para habilitar o Model Registry local
MLFLOW_TRACKING_URI = f"sqlite:///{ROOT / 'mlflow.db'}"
EXPERIMENT_BASELINE = "baseline-vs-rf"
EXPERIMENT_TUNING = "hyperparameter-tuning"
EXPERIMENT_MONITORING = "production-monitoring"
REGISTERED_MODEL_NAME = "breast-cancer-classifier"
CHAMPION_ALIAS = "champion"

RANDOM_STATE = 42
TEST_SIZE = 0.2

# Métrica primária de tuning (ADR-004): recall da classe maligna
TUNING_SCORING = "recall"

for d in (DATA_DIR, MODELS_DIR, REPORTS_DIR):
    d.mkdir(parents=True, exist_ok=True)

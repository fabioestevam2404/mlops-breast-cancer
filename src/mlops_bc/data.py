"""Carga, diagnóstico de qualidade e pré-processamento do Breast Cancer Wisconsin.

Convenção do dataset sklearn: target 0 = maligno, 1 = benigno.
Para que "recall" signifique "recall da classe maligna" (ADR-004), invertemos o
target: 1 = maligno (classe positiva), 0 = benigno.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

from mlops_bc.config import DATA_DIR, RANDOM_STATE, TEST_SIZE


def load_raw() -> pd.DataFrame:
    """Retorna o dataset completo com a coluna `target` (1 = maligno)."""
    ds = load_breast_cancer(as_frame=True)
    df = ds.frame.copy()
    df["target"] = 1 - df["target"]  # 1 = maligno (positivo), 0 = benigno
    return df


def quality_report(df: pd.DataFrame) -> pd.DataFrame:
    """Diagnóstico de qualidade: completude, duplicatas e outliers (IQR) por coluna."""
    rows = []
    n_dup = int(df.duplicated().sum())
    for col in df.select_dtypes("number").columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        outliers = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
        rows.append(
            {
                "coluna": col,
                "completude_pct": round((1 - df[col].isna().mean()) * 100, 2),
                "outliers_iqr": outliers,
                "skewness": round(float(df[col].skew()), 3),
            }
        )
    report = pd.DataFrame(rows)
    report.attrs["n_duplicatas"] = n_dup
    return report


def make_splits(
    df: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split estratificado 80/20. O scaling NÃO acontece aqui.

    O StandardScaler vive dentro do Pipeline sklearn (train.py) para que
    fit ocorra apenas no treino — evitando data leakage — e para que scaler
    e modelo sejam versionados como artefato único.
    """
    if df is None:
        df = load_raw()
    x = df.drop(columns=["target"])
    y = df["target"]
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    return x_train, x_test, y_train, y_test


def save_processed() -> None:
    """Persiste os splits em Parquet para reprodutibilidade entre etapas."""
    x_train, x_test, y_train, y_test = make_splits()
    x_train.to_parquet(DATA_DIR / "x_train.parquet")
    x_test.to_parquet(DATA_DIR / "x_test.parquet")
    y_train.to_frame().to_parquet(DATA_DIR / "y_train.parquet")
    y_test.to_frame().to_parquet(DATA_DIR / "y_test.parquet")
    print(f"[data] splits salvos em {DATA_DIR}")
    print(f"[data] treino: {x_train.shape} | teste: {x_test.shape}")
    print(f"[data] proporção maligno (treino): {y_train.mean():.3f}")


def load_processed() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray]:
    """Carrega splits persistidos; gera se ainda não existirem."""
    if not (DATA_DIR / "x_train.parquet").exists():
        save_processed()
    x_train = pd.read_parquet(DATA_DIR / "x_train.parquet")
    x_test = pd.read_parquet(DATA_DIR / "x_test.parquet")
    y_train = pd.read_parquet(DATA_DIR / "y_train.parquet")["target"].to_numpy()
    y_test = pd.read_parquet(DATA_DIR / "y_test.parquet")["target"].to_numpy()
    return x_train, x_test, y_train, y_test


if __name__ == "__main__":  # pragma: no cover
    save_processed()

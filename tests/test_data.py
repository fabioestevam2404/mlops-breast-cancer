"""Testes da camada de dados: integridade, split e ausência de leakage."""

import numpy as np

from mlops_bc.data import load_raw, make_splits, quality_report


def test_load_raw_shape_and_target():
    df = load_raw()
    assert df.shape == (569, 31)
    # 1 = maligno (212 casos no dataset original)
    assert int(df["target"].sum()) == 212
    assert df.isna().sum().sum() == 0


def test_quality_report_columns():
    report = quality_report(load_raw())
    assert {"coluna", "completude_pct", "outliers_iqr", "skewness"} <= set(report.columns)
    assert (report["completude_pct"] == 100.0).all()


def test_split_stratified_and_disjoint():
    x_train, x_test, y_train, y_test = make_splits()
    assert len(x_train) + len(x_test) == 569
    # Estratificação: proporções de malignos próximas entre treino e teste
    assert abs(y_train.mean() - y_test.mean()) < 0.02
    # Sem vazamento: nenhum índice compartilhado
    assert set(x_train.index).isdisjoint(set(x_test.index))


def test_features_not_scaled_in_data_layer():
    """O scaling deve ocorrer só dentro do Pipeline (train.py), nunca em data.py."""
    x_train, _, _, _ = make_splits()
    assert x_train["mean area"].max() > 100  # dados em escala original
    assert not np.allclose(x_train.mean().to_numpy(), 0, atol=0.5)


def test_save_and_load_processed_roundtrip():
    from mlops_bc.data import load_processed, save_processed

    save_processed()
    x_train, x_test, y_train, y_test = load_processed()
    assert x_train.shape == (455, 30)
    assert len(y_test) == 114

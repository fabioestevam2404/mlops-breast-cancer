"""Testes do monitoramento: PSI e simulação de drift."""

import numpy as np
import pandas as pd

from mlops_bc.drift import psi, simulate_drift


def test_psi_identical_distributions_is_near_zero():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    assert psi(a, a) < 0.01


def test_psi_shifted_distribution_alerts():
    rng = np.random.default_rng(0)
    a = rng.normal(size=5000)
    b = rng.normal(loc=1.5, size=5000)
    assert psi(a, b) > 0.25


def test_simulate_drift_preserves_shape_and_shifts_mean():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.normal(size=(200, 5)), columns=list("abcde"))
    x_drift = simulate_drift(x, shift_std=1.0, rng=rng)
    assert x_drift.shape == x.shape
    assert (x_drift.mean() > x.mean()).all()

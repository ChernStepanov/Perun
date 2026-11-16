import numpy as np
import pandas as pd
import pytest

from perun import events, models
from perun.config import week_start


def _synthetic(effect: float, seed: int = 0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-04", periods=260, freq="W-MON")
    t = np.arange(len(idx))
    mu = np.exp(np.log(80) + 0.004 * t + 0.1 * np.sin(2 * np.pi * t / 52))
    ev = pd.Series(idx[np.arange(20, 240, 11)])
    exp = events.exposure(idx, ev, range(0, 13)).sum(axis=1).to_numpy()
    y = rng.poisson(mu * np.exp(np.log1p(effect) * exp))
    return pd.Series(y, index=idx, dtype=float), ev


def test_week_start_is_monday():
    s = pd.Series(["2026-11-16T15:26:23+00:00", "2026-11-22", "2026-11-23"])
    assert week_start(s).dt.dayofweek.eq(0).all()
    assert week_start(s).tolist() == [pd.Timestamp("2026-11-16")] * 2 + [pd.Timestamp("2026-11-23")]


def test_exposure_lags():
    idx = pd.date_range("2024-01-01", periods=10, freq="W-MON")
    e = events.exposure(idx, pd.Series([idx[5]]), range(-2, 3))
    assert e.loc[idx[3], "L-2"] == 1
    assert e.loc[idx[7], "L2"] == 1
    assert e.to_numpy().sum() == 5


def test_shift_wraps_and_keeps_spacing():
    idx = pd.date_range("2024-01-01", periods=50, freq="W-MON")
    ev = pd.Series(idx[[3, 10, 48]])
    sh = models.shift_weeks(ev, idx, 5)
    assert list(idx.get_indexer(sh)) == [8, 15, 3]


@pytest.mark.parametrize("effect", [0.0, 0.2])
def test_lag_study_recovers_effect(effect):
    y, ev = _synthetic(effect)
    r = models.lag_study(y, ev, n_placebo=60)
    assert abs(models.pct(r.observed) - effect * 100) < 8
    if effect:
        assert r.p_placebo < 0.05

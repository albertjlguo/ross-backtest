import numpy as np
import pandas as pd

from trendbt.lab import LabParams, bonferroni_t, panel, raw_weights, run, stats


def _px(n=3000, k=8, seed=0, drift=None):
    rng = np.random.default_rng(seed)
    r = rng.normal(0, 0.01, (n, k)) + (0 if drift is None else drift)
    idx = pd.bdate_range("2005-01-03", periods=n)
    return pd.DataFrame(100 * np.exp(np.cumsum(r, axis=0)), index=idx, columns=[f"A{i}" for i in range(k)])


def test_no_lookahead():
    px = _px()
    cost = pd.Series(5.0, index=px.columns)
    full = run(px, "trend4", cost)["ret"]
    part = run(px.iloc[:2000], "trend4", cost)["ret"]
    assert np.allclose(full.reindex(part.index).to_numpy(), part.to_numpy())


def test_xs_weights_market_neutral_before_vol_scaling():
    px = _px()
    w = raw_weights(px, "xs_mom", LabParams())
    assert (w.iloc[400:] != 0).any().any()


def test_random_walk_has_no_edge_and_trend_is_found():
    cost = pd.Series(0.0, index=_px().columns)
    sh = [stats(run(_px(seed=s), "tsmom12", cost)["ret"])["sharpe"] for s in range(6)]
    assert abs(np.mean(sh)) < 0.35
    tr = stats(run(_px(drift=np.linspace(-0.0008, 0.0008, 8)), "tsmom12", cost)["ret"])
    assert tr["sharpe"] > 1


def test_vol_target_and_costs():
    px = _px(seed=3)
    a = run(px, "bh_vt", pd.Series(0.0, index=px.columns))
    b = run(px, "bh_vt", pd.Series(50.0, index=px.columns))
    assert 0.07 < stats(a["ret"])["vol"] < 0.13
    assert b["ret"].sum() < a["ret"].sum()


def test_bonferroni():
    assert 1.9 < bonferroni_t(1) < 2.0 and bonferroni_t(50) > 3


def test_panel_business_days():
    d = {"X": pd.DataFrame({"close": [1.0, 2, 3]}, index=pd.to_datetime(["2024-01-05", "2024-01-06", "2024-01-08"]))}
    p = panel(d)
    assert list(p["X"]) == [1.0, 3.0]


def test_boost_no_leverage_and_no_lookahead():
    from trendbt.lab import run_boost
    px = _px(seed=5, drift=0.0002)
    cost = pd.Series(5.0, index=px.columns)
    a = run_boost(px, cost, k=0.5)
    assert a["max_exposure"] <= 1.0 + 1e-9 and a["avg_exposure"] < 1.0
    b = run_boost(px.iloc[:2000], cost, k=0.5)["ret"]
    assert np.allclose(a["ret"].reindex(b.index).to_numpy(), b.to_numpy())
    base = run_boost(px, cost, k=None)
    full = run_boost(px, cost, k=None, base=1.0)
    assert abs(stats(base["ret"])["sharpe"] - stats(full["ret"])["sharpe"]) < 0.05
    assert a["avg_exposure"] >= base["avg_exposure"]

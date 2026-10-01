"""
趋势研究模块的测试：无未来函数（截断测试）、成交规则、期权定价、方差比。
"""
import numpy as np
import pandas as pd
import pytest

from trendbt.indicators import add_indicators, resample
from trendbt.options import CSPParams, bs_put, put_delta, simulate_csp, strike_for_delta
from trendbt.stats import metrics, variance_ratio
from trendbt.strategy import StratParams, prepare, simulate, simulate_mtf
from trendbt.structure import fib_anchors, fib_level, zigzag


def gbm_daily(n=4000, seed=0, mu=0.0004, vol=0.02):
    rng = np.random.default_rng(seed)
    r = rng.normal(mu, vol, n)
    # 加一点周期性趋势，让波段明显
    r += 0.003 * np.sin(np.arange(n) / 150)
    c = 100 * np.exp(np.cumsum(r))
    o = np.concatenate([[100], c[:-1]]) * (1 + rng.normal(0, 0.002, n))
    h = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.006, n)))
    l = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.006, n)))
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_monthly_resample_labels_last_trading_day():
    d = gbm_daily(300)
    m = resample(d, "M")
    jan = d.loc["2000-01"]
    assert m.index[0] == jan.index[-1]
    assert m.iloc[0]["high"] == jan["high"].max() and m.iloc[0]["open"] == jan["open"].iloc[0]


def test_zigzag_pivots_confirmed_after_they_happen():
    b = add_indicators(resample(gbm_daily(), "W"))
    piv = zigzag(b, 3.0)
    assert len(piv) >= 4
    assert all(conf > i for i, _, _, conf in piv)
    types = [t for _, _, t, _ in piv]
    # 高低点基本交替（作废替换时允许连续同类）
    assert types.count("H") > 0 and types.count("L") > 0


def test_anchors_no_lookahead():
    b = add_indicators(resample(gbm_daily(), "W"))
    full = fib_anchors(b)
    for cut in (300, 450, 600, len(b) - 1):
        part = fib_anchors(add_indicators(resample(gbm_daily(), "W").iloc[:cut + 1]))
        a, p = full.iloc[cut], part.iloc[cut]
        assert (np.isnan(a["H"]) and np.isnan(p["H"])) or (a["H"] == p["H"] and a["L"] == p["L"])


def test_simulate_no_lookahead_on_closed_trades():
    d = gbm_daily()
    b = prepare(d, "W")
    tr_full, _ = simulate(b, StratParams(entry="any", exit="struct"))
    assert len(tr_full) > 0
    first = tr_full[0]
    cut = b.index.get_loc(first["exit_t"])
    tr_part, _ = simulate(prepare(d.loc[:b.index[cut]], "W"), StratParams(entry="any", exit="struct"))
    assert tr_part[0]["entry_t"] == first["entry_t"]
    assert tr_part[0]["entry_px"] == pytest.approx(first["entry_px"])
    assert tr_part[0]["exit_px"] == pytest.approx(first["exit_px"])


def _toy_bars():
    # 手工构造：健康状态下挂单在 0.5 位，下一根触及成交；随后收盘跌破 0.236 止损
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    b = pd.DataFrame({
        "open":  [100, 100, 100, 96, 80, 70],
        "high":  [101, 101, 101, 97, 81, 71],
        "low":   [99, 99, 99, 54, 60, 68],
        "close": [100, 100, 100, 95, 62, 69],
    }, index=idx, dtype=float)
    b["ma50"] = np.nan; b["ma80"] = 50.0
    b["K"] = 50.0; b["D"] = 50.0; b["macd"] = 1.0; b["macd_sig"] = 0.0
    b["H"] = 110.0; b["L"] = 10.0; b["H_idx"] = 0; b["L_idx"] = 0
    return b


def test_limit_fill_and_stop():
    b = _toy_bars()
    sp = StratParams(entry="fib", exit="struct", health="fib", cost_bps=0)
    tr, ret = simulate(b, sp)
    # 0.5 位 = 60，0.618 = 71.8（低于收盘 100 的最高支撑是 0.618 → 71.8）
    t = tr[0]
    assert t["entry_px"] == pytest.approx(fib_level(110, 10, 0.618))
    assert t["entry_t"] == b.index[3]
    # 0.236 = 33.6；收盘 62 未破 → 持有到期末
    assert t["exit_why"] == "open_end"


def test_stop_exits_next_open():
    b = _toy_bars()
    b.loc[b.index[4], "close"] = 30.0          # 跌破 0.236（33.6）
    sp = StratParams(entry="fib", exit="struct", health="fib", cost_bps=0)
    tr, _ = simulate(b, sp)
    assert tr[0]["exit_why"] == "stop" and tr[0]["exit_t"] == b.index[5]
    assert tr[0]["exit_px"] == 70.0


def test_target_exit_at_prior_high():
    b = _toy_bars()
    b.loc[b.index[5], ["high", "close"]] = [115.0, 112.0]
    sp = StratParams(entry="fib", exit="target", health="fib", cost_bps=0)
    tr, _ = simulate(b, sp)
    assert tr[0]["exit_why"] == "target" and tr[0]["exit_px"] == 110.0


def test_bs_put_and_delta():
    assert bs_put(100, 95, 30 / 365, 0.2) == pytest.approx(0.567, abs=0.01)
    K = strike_for_delta(100, 30 / 365, 0.2, -0.25)
    assert put_delta(100, K, 30 / 365, 0.2) == pytest.approx(-0.25, abs=1e-3)


def test_csp_runs_and_assignment_is_bool():
    df = simulate_csp(gbm_daily(3000), CSPParams(mode="always"))
    sold = df[df["sold"]]
    assert len(sold) > 50 and df["assigned"].dtype == bool
    assert 0.05 < sold["assigned"].mean() < 0.5


def test_mtf_runs():
    tr, ret = simulate_mtf(gbm_daily(4000))
    assert isinstance(tr, list) and len(ret) == 4000


def test_variance_ratio():
    rng = np.random.default_rng(1)
    rw = pd.Series(rng.normal(0, 0.01, 5000))
    assert variance_ratio(rw, 5)[0] == pytest.approx(1, abs=0.1)
    x = np.zeros(5000); e = rng.normal(0, 0.01, 5000)
    for i in range(1, 5000):
        x[i] = 0.3 * x[i - 1] + e[i]
    v, z = variance_ratio(pd.Series(x), 2)
    assert v == pytest.approx(1.3, abs=0.05) and z > 5


def test_metrics():
    m = pd.Series([0.01] * 24, index=pd.period_range("2020-01", periods=24, freq="M"))
    out = metrics(m)
    assert out["cagr"] == pytest.approx(1.01 ** 12 - 1, abs=1e-3) and out["max_dd"] == 0

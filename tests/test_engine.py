"""
用手工构造的 K 线逐条验证规则和成交假设。成本默认清零，数字可以手算核对。

基础场景（昨收 5.00，ADV30 = 10 万股，流通股 500 万，06:00 有新闻）：
  06:50  5.00→5.50  60 万股       → 涨 10%、相对量 6 倍，入选
  07:00  5.50→6.00  10 万股       → 创新高 6.00（拉升段低点 5.00）
  07:01  阴线 高 6.00 低 5.85      → 第一次回调；挂单：触发 6.01，止损 5.85，1R=0.16，T1=6.33
  07:02  触发 K 线（各场景不同）
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from rossbt import Config, load_bars, load_daily, load_news, run_backtest, prepare_daily
from rossbt.backtest import apply_portfolio_rules
from rossbt.synth import generate_synthetic

ET = "America/New_York"
DAY = "2026-03-10"

ZERO_COST = dict(slippage_bps=0.0, slippage_min=0.0, commission_per_share=0.0, commission_min=0.0)
QUIET = dict(sig_stall=False, sig_bid_pressure=False, sig_topping_tail=False,
             sig_rejections=False, sig_volume_divergence=False)

BASE = [
    ("06:50", 5.00, 5.50, 5.00, 5.50, 600_000),
    ("07:00", 5.50, 6.00, 5.50, 6.00, 100_000),
    ("07:01", 6.00, 6.00, 5.85, 5.88, 20_000),
]


def make(rows, extra_cols=None, adv30=100_000, float_shares=5e6, news_time="06:00"):
    df = pd.DataFrame(rows, columns=["hm", "open", "high", "low", "close", "volume"])
    df["ts"] = pd.to_datetime(DAY + " " + df["hm"]).dt.tz_localize(ET)
    df["symbol"] = "TEST"
    if extra_cols:
        for k, vals in extra_cols.items():
            df[k] = vals
    bars = load_bars(df.drop(columns="hm"))
    daily = load_daily(pd.DataFrame([{"date": DAY, "symbol": "TEST", "prev_close": 5.0,
                                      "adv30": adv30, "float_shares": float_shares}]))
    news = load_news(pd.DataFrame([{"symbol": "TEST", "ts": f"{DAY} {news_time}"}]))
    return bars, daily, news


def run(rows, cfg_kw=None, **make_kw):
    kw = {**ZERO_COST, **QUIET, **(cfg_kw or {})}
    bars, daily, news = make(rows, **make_kw)
    return run_backtest(bars, daily, Config(**kw), news=news)


def only_trade(res):
    assert len(res.trades) == 1, res.trades
    return res.trades.iloc[0]


# ------------------------------------------------------------------------- #
def test_entry_then_stop():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),   # 阳线，开盘低于 6.01 → 6.01 成交
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),   # 阴线跌破 5.85 → 止损
    ]
    t = only_trade(run(rows))
    assert t.trigger == pytest.approx(6.01)
    assert t.stop_init == pytest.approx(5.85)
    assert t.shares_init == 625                     # 100 / 0.16
    assert t.exit_reason == "stop"
    assert t.pnl == pytest.approx(-100.0, abs=0.01)
    assert t.r_multiple == pytest.approx(-1.0, abs=1e-3)


def test_t1_half_then_breakeven():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.08, 6.40, 6.05, 6.38, 60_000),   # 触及 T1=6.33 → 卖 312 股
        ("07:04", 6.30, 6.35, 6.00, 6.02, 40_000),   # 剩余止损已上移到 6.01 → 保本出
    ]
    t = only_trade(run(rows))
    fills = json.loads(t.fills)
    assert [f["why"] for f in fills] == ["entry", "t1", "breakeven_stop"]
    assert fills[1]["sh"] == 312 and fills[1]["px"] == pytest.approx(6.33)
    assert fills[2]["px"] == pytest.approx(6.01)
    assert t.pnl == pytest.approx(312 * 0.32, abs=0.01)
    assert bool(t.t1_hit)


def test_breakeven_not_active_on_t1_bar():
    # T1 那根是阴线：先到高点卖一半，再回落到 5.90（高于原止损 5.85）——保本止损下一根才生效
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.30, 6.40, 5.90, 5.95, 60_000),
        ("07:04", 6.03, 6.05, 5.94, 6.04, 10_000),   # 阳线，低点 5.94 跌破已生效的保本价 6.01
    ]
    t = only_trade(run(rows))
    whys = [f["why"] for f in json.loads(t.fills)]
    assert whys == ["entry", "t1", "breakeven_stop"]


def test_setup_broken_before_trigger_is_skipped_under_ohlc():
    # 阳线、开盘低于触发价、最低价先跌破回调低点 → 形态已坏，撤单
    rows = BASE + [("07:02", 5.90, 6.10, 5.80, 6.08, 50_000)]
    res = run(rows)
    assert len(res.trades) == 0
    assert res.counters["setup_broken_before_trigger"] == 1


def test_worst_case_takes_same_bar_stop():
    rows = BASE + [("07:02", 5.90, 6.10, 5.80, 6.08, 50_000)]
    t = only_trade(run(rows, {"intrabar_path": "worst_case"}))
    assert t.exit_reason == "stop_same_bar"
    assert t.r_multiple == pytest.approx(-1.0, abs=1e-3)


def test_red_trigger_bar_stops_same_bar():
    # 阴线：开→高（6.01 成交）→低（5.84 破止损）
    rows = BASE + [("07:02", 5.90, 6.05, 5.84, 5.86, 50_000)]
    t = only_trade(run(rows))
    assert t.exit_reason == "stop_same_bar"
    assert t.pnl == pytest.approx(-100.0, abs=0.01)


def test_gap_through_stop_fills_at_open():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 5.70, 5.75, 5.60, 5.65, 80_000),
    ]
    t = only_trade(run(rows))
    assert t.exit_reason == "stop_gap"
    assert t.pnl == pytest.approx(625 * (5.70 - 6.01), abs=0.01)


def test_gap_above_trigger_fills_at_open():
    rows = BASE + [
        ("07:02", 6.05, 6.12, 6.04, 6.10, 50_000),
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),
    ]
    t = only_trade(run(rows))
    assert json.loads(t.fills)[0]["px"] == pytest.approx(6.05)
    assert t.shares_init == 625                       # 仓位按挂单时的 1R 计算


def test_costs_applied():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),
    ]
    kw = {"slippage_bps": 0.0, "slippage_min": 0.01, "commission_per_share": 0.01,
          "commission_min": 1.0}
    t = only_trade(run(rows, kw))
    # 买 6.02、卖 5.84，佣金 2 × 6.25
    assert t.pnl == pytest.approx(625 * (5.84 - 6.02) - 12.5, abs=0.01)


def test_no_entry_outside_window():
    rows = BASE + [("07:02", 5.90, 6.10, 5.88, 6.08, 50_000)]
    assert len(run(rows, {"entry_end": "07:02"}).trades) == 0
    assert len(run(rows, {"entry_blackouts": (("07:02", "07:03"),)}).trades) == 0


def test_not_qualified_low_rvol():
    rows = BASE + [("07:02", 5.90, 6.10, 5.88, 6.08, 50_000)]
    assert len(run(rows, adv30=1_000_000).trades) == 0


def test_float_and_price_filters():
    rows = BASE + [("07:02", 5.90, 6.10, 5.88, 6.08, 50_000)]
    assert len(run(rows, float_shares=25e6).trades) == 0
    assert len(run(rows, {"max_price": 5.5}).trades) == 0


def test_catalyst_must_precede_bar():
    rows = BASE + [("07:02", 5.90, 6.10, 5.88, 6.08, 50_000)]
    assert len(run(rows, news_time="07:30").trades) == 0       # 新闻在之后 → 不算
    assert len(run(rows, {"require_catalyst": False}, news_time="07:30").trades) == 1


def test_vwap_and_retrace_filters():
    rows = BASE + [("07:02", 5.90, 6.10, 5.88, 6.08, 50_000)]
    assert len(run(rows, {"max_retrace": 0.10}).trades) == 0   # 实际回撤 15%


def test_first_pullback_only():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),   # 第一笔止损
        ("07:04", 5.82, 6.30, 5.82, 6.30, 120_000),  # 新高
        ("07:05", 6.30, 6.30, 6.15, 6.18, 20_000),   # 第二次回调
        ("07:06", 6.18, 6.40, 6.17, 6.38, 60_000),   # 触发
    ]
    res1 = run(rows)
    assert len(res1.trades) == 1
    assert res1.counters["blocked_pullback_number"] >= 1
    res2 = run(rows, {"max_pullback_number": 2, "daily_max_loss": None})
    assert len(res2.trades) == 2
    assert list(res2.trades["episode"]) == [1, 2]


def test_daily_max_loss_blocks_next_trade():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),
        ("07:04", 5.82, 6.30, 5.82, 6.30, 120_000),
        ("07:05", 6.30, 6.30, 6.15, 6.18, 20_000),
        ("07:06", 6.18, 6.40, 6.17, 6.38, 60_000),
    ]
    res = run(rows, {"max_pullback_number": 2, "daily_max_loss": 90.0})
    assert len(res.trades) == 1 and len(res.dropped) == 1
    assert res.dropped.iloc[0]["drop_reason"] == "daily_max_loss"


def test_first_red_candle_exit_before_t1():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.08, 6.12, 6.00, 6.02, 30_000),   # 收阴，未破止损
        ("07:04", 6.03, 6.05, 6.00, 6.04, 10_000),   # 下一根开盘离场
    ]
    t = only_trade(run(rows))
    assert t.exit_reason == "first_red_candle"
    assert json.loads(t.fills)[-1]["px"] == pytest.approx(6.03)


def test_sell_at_bid_signal_uses_tape_columns():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.08, 6.20, 6.06, 6.15, 300_000),  # 阳线但 80% 成交在买一价
        ("07:04", 6.14, 6.16, 6.10, 6.12, 10_000),
    ]
    extra = {"vol_at_bid": [0, 0, 0, 0, 240_000, 0], "vol_at_ask": [1, 1, 1, 1, 60_000, 1]}
    cfg = {"sig_bid_pressure": True}
    t = only_trade(run(rows, cfg, extra_cols=extra))
    assert t.exit_reason == "sell_at_bid"
    assert json.loads(t.fills)[-1]["px"] == pytest.approx(6.14)


def test_topping_tail_signal():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.08, 6.30, 6.07, 6.10, 40_000),   # 长上影
        ("07:04", 6.09, 6.12, 6.05, 6.10, 10_000),
    ]
    t = only_trade(run(rows, {"sig_topping_tail": True}))
    assert t.exit_reason == "topping_tail"


def test_volume_divergence_signal():
    rows = [
        ("06:50", 5.00, 5.50, 5.00, 5.50, 600_000),
        ("07:00", 5.50, 6.00, 5.50, 6.00, 700_000),
        ("07:01", 6.00, 6.00, 5.85, 5.88, 20_000),
        ("07:02", 5.90, 6.10, 5.88, 6.08, 300_000),  # 新一轮冲高（入场）
        ("07:03", 6.08, 6.25, 6.07, 6.24, 200_000),  # 再创新高，本轮峰值 30 万 < 0.5 × 70 万
        ("07:04", 6.24, 6.26, 6.20, 6.22, 10_000),
    ]
    t = only_trade(run(rows, {"sig_volume_divergence": True}))
    assert t.exit_reason == "volume_divergence"


def test_aggressive_adds_only_when_risk_locked():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.08, 6.40, 6.05, 6.38, 300_000),  # 新高
        ("07:04", 6.38, 6.38, 6.20, 6.22, 30_000),   # 回调低点 6.20 ≥ 成本 6.01
        ("07:05", 6.22, 6.50, 6.21, 6.48, 300_000),  # 触发加仓，整笔止损上移到 6.20
        ("07:06", 6.45, 6.46, 6.15, 6.18, 30_000),   # 跌破 6.20
    ]
    t = only_trade(run(rows, {"mode": "aggressive", "add_min_rvol": 5.0}))
    fills = json.loads(t.fills)
    assert [f["why"] for f in fills] == ["entry", "add_1", "raised_stop"]
    assert fills[1]["sh"] == 312
    assert fills[2]["px"] == pytest.approx(6.20)
    assert t.pnl > 0                                   # 加仓后最坏结果仍盈利


def test_summary_preset_targets_hod():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),
    ]
    # 第一目标 = 回调前高点 6.00 < 触发价 6.01 → 盈亏比不足 1:1，不做
    res = run(rows, {"target_mode": "hod"})
    assert len(res.trades) == 0 and res.counters["rejected_rr"] >= 1


# ---------------------------- 无未来函数 ---------------------------------- #
def test_no_lookahead_truncation():
    bars, daily, news = generate_synthetic(n_days=15, n_symbols=12, seed=3)
    bars, daily, news = load_bars(bars), load_daily(daily), load_news(news)
    cfg = Config(top_n_gainers=3)
    full = run_backtest(bars, daily, cfg, news=news).trades_all
    assert len(full) >= 3
    for _, tr in full.head(5).iterrows():
        cut = bars[~((bars["date"] == tr["date"]) & (bars["ts"] > tr["exit_time"]))]
        part = run_backtest(cut, daily, cfg, news=news).trades_all
        same = part[(part["date"] == tr["date"]) & (part["symbol"] == tr["symbol"])
                    & (part["entry_time"] == tr["entry_time"])]
        assert len(same) == 1
        a, b = same.iloc[0], tr
        for col in ["trigger", "stop_init", "shares_init", "exit_reason", "pnl", "fills"]:
            assert a[col] == b[col], (col, a[col], b[col])


def test_prepare_daily_uses_only_past():
    d = pd.DataFrame({"date": pd.bdate_range("2026-01-01", periods=40), "symbol": "X",
                      "close": np.arange(40) + 1.0, "volume": np.arange(40) * 1000.0})
    out = prepare_daily(d)
    row = out[out["date"] == pd.Timestamp(d["date"].iloc[35]).date()].iloc[0]
    assert row["prev_close"] == 35.0
    assert row["adv30"] == pytest.approx(np.mean(np.arange(5, 35) * 1000.0))


def test_portfolio_concurrency():
    t0 = pd.Timestamp("2026-03-10 07:00", tz=ET)
    tr = pd.DataFrame([
        {"date": t0.date(), "entry_time": t0, "exit_time": t0 + pd.Timedelta(minutes=10), "pnl": 50},
        {"date": t0.date(), "entry_time": t0 + pd.Timedelta(minutes=5),
         "exit_time": t0 + pd.Timedelta(minutes=8), "pnl": 20},
        {"date": t0.date(), "entry_time": t0 + pd.Timedelta(minutes=12),
         "exit_time": t0 + pd.Timedelta(minutes=15), "pnl": -10},
    ])
    kept, dropped = apply_portfolio_rules(tr, Config(max_concurrent_positions=1))
    assert list(kept["pnl"]) == [50, -10]
    assert list(dropped["drop_reason"]) == ["max_concurrent"]


def test_top_n_gainers_rank():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.05, 6.06, 5.80, 5.82, 30_000),
    ]
    b1, d1, n1 = make(rows)
    hot = [(hm, o * 1.2, h * 1.2, l * 1.2, c * 1.2, v) for hm, o, h, l, c, v in rows]
    b2, d2, n2 = make(hot)
    b2["symbol"] = "HOT"; d2["symbol"] = "HOT"; n2["symbol"] = "HOT"
    bars = pd.concat([b1, b2]).sort_values(["date", "symbol", "ts"]).reset_index(drop=True)
    daily, news = pd.concat([d1, d2]), pd.concat([n1, n2])
    cfg = Config(**ZERO_COST, **QUIET, top_n_gainers=1, max_concurrent_positions=5)
    res = run_backtest(bars, daily, cfg, news=news)
    assert set(res.trades["symbol"]) == {"HOT"}
    cfg2 = Config(**ZERO_COST, **QUIET, top_n_gainers=2, max_concurrent_positions=5)
    assert set(run_backtest(bars, daily, cfg2, news=news).trades["symbol"]) == {"HOT", "TEST"}


# ------------------------------ 逐笔结论 ------------------------------------ #
def _resolver_key(hm, up, down):
    t = pd.Timestamp(f"{DAY} {hm}", tz=ET).tz_convert("UTC").isoformat()
    return ("TEST", t, round(up, 4), round(down, 4))


def test_ticks_resolve_entry_ambiguity():
    rows = BASE + [("07:02", 5.90, 6.10, 5.80, 6.08, 50_000)]   # 阳线，触发价和止损都触及
    bars, daily, news = make(rows)
    cfg = Config(**ZERO_COST, **QUIET)
    # 默认（ohlc）：阳线先低后高 → 形态先破，不入场；同时记录歧义事件
    r0 = run_backtest(bars, daily, cfg, news=news, collect_ambiguous=True)
    assert len(r0.trades) == 0 and r0.ambiguous[0]["kind"] == "entry"
    # 逐笔说先到触发价 → 真实成交后被止损
    res_up = {_resolver_key("07:02", 6.01, 5.85): "up"}
    t = only_trade(run_backtest(bars, daily, cfg, news=news, resolver=res_up))
    assert t.exit_reason == "stop_same_bar" and t.r_multiple == pytest.approx(-1.0, abs=1e-3)
    # 逐笔说先破止损 → 不入场
    res_dn = {_resolver_key("07:02", 6.01, 5.85): "down"}
    assert len(run_backtest(bars, daily, cfg, news=news, resolver=res_dn).trades) == 0


def test_ticks_resolve_position_ambiguity():
    rows = BASE + [
        ("07:02", 5.90, 6.10, 5.88, 6.08, 50_000),
        ("07:03", 6.08, 6.40, 5.80, 6.30, 60_000),   # 阳线：T1=6.33 和止损 5.85 都触及
    ]
    bars, daily, news = make(rows)
    cfg = Config(**ZERO_COST, **QUIET)
    t0 = only_trade(run_backtest(bars, daily, cfg, news=news))
    assert t0.exit_reason == "stop"                          # ohlc：阳线先低 → 先止损
    res = {_resolver_key("07:03", 6.33, 5.85): "up"}
    t1 = only_trade(run_backtest(bars, daily, cfg, news=news, resolver=res))
    assert [f["why"] for f in json.loads(t1.fills)] == ["entry", "t1", "stop"]


def test_tick_resolution_logic():
    from rossbt.ticks import resolve, first_touch
    assert first_touch([5.9, 6.02, 5.8], 6.01, 5.85) == "up"
    assert first_touch([5.9, 5.84, 6.2], 6.01, 5.85) == "down"
    assert first_touch([5.9, 5.95], 6.01, 5.85) is None
    ev = pd.DataFrame([{"symbol": "X", "minute_utc": "m", "up": 6.01, "down": 5.85}])
    tr = pd.DataFrame({"ts": pd.to_datetime(["2026-01-01T00:00:01Z"] * 3), "id": [1, 2, 3],
                       "price": [5.80, 6.05, 5.70], "conditions": ["Z", "@,T", ""]})
    out = resolve(ev, {("X", "m"): tr})
    assert out["first"].iloc[0] == "up"           # 5.80 那笔是乱序上报，被剔除

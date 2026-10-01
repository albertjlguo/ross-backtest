"""
绩效汇总：整体指标 + 按时段 / 价格 / 流通股 / 离场原因 / 回调序号分组。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtest import BacktestResult


def _core(t: pd.DataFrame) -> dict:
    if len(t) == 0:
        return {"trades": 0}
    r = t["r_multiple"]
    wins, losses = t[t["pnl"] > 0], t[t["pnl"] <= 0]
    gp, gl = wins["pnl"].sum(), -losses["pnl"].sum()
    return {
        "trades": len(t),
        "win_rate": round(len(wins) / len(t), 4),
        "avg_r": round(r.mean(), 4),
        "median_r": round(r.median(), 4),
        "avg_win_r": round(wins["r_multiple"].mean(), 4) if len(wins) else np.nan,
        "avg_loss_r": round(losses["r_multiple"].mean(), 4) if len(losses) else np.nan,
        "profit_factor": round(gp / gl, 4) if gl > 0 else np.inf,
        "total_pnl": round(t["pnl"].sum(), 2),
        "t1_hit_rate": round(t["t1_hit"].mean(), 4),
    }


def _max_consec_losses(pnl: pd.Series) -> int:
    best = cur = 0
    for x in pnl:
        cur = cur + 1 if x <= 0 else 0
        best = max(best, cur)
    return best


def summarize(res: BacktestResult) -> dict:
    t = res.trades
    out = {"overall": _core(t)}
    days = pd.Index(res.days)
    if len(t):
        daily = t.groupby("date")["pnl"].sum().reindex(days, fill_value=0.0)
        eq = daily.cumsum()
        dd = eq - eq.cummax()
        sd = daily.std(ddof=1)
        out["overall"].update({
            "days_in_sample": len(days),
            "days_traded": int(t["date"].nunique()),
            "max_drawdown": round(dd.min(), 2),
            "max_consec_losses": _max_consec_losses(t["pnl"]),
            "daily_sharpe_ann": round(daily.mean() / sd * np.sqrt(252), 3) if sd > 0 else np.nan,
            "expectancy_per_trade": round(t["pnl"].mean(), 2),
        })
        out["equity"] = eq

        tt = t.copy()
        bins_min = [0, 8 * 60, 9 * 60, 9 * 60 + 30, 10 * 60, 24 * 60]
        labels = ["07:00-08:00", "08:00-09:00", "09:00-09:30", "09:30-10:00", "10:00+"]
        tt["time_bucket"] = pd.cut(tt["entry_minute"], [7 * 60] + bins_min[1:], right=False,
                                   labels=labels)
        tt["price_bucket"] = pd.cut(tt["trigger"], [0, 5, 10, 20, np.inf], right=False,
                                    labels=["$2-5", "$5-10", "$10-20", "$20+"])
        tt["float_bucket"] = pd.cut(tt["float_shares"].astype(float), [0, 5e6, 10e6, 20e6, np.inf],
                                    right=False, labels=["<5M", "5-10M", "10-20M", "20M+"])
        tt["float_bucket"] = tt["float_bucket"].cat.add_categories("unknown").fillna("unknown")
        for col in ["time_bucket", "price_bucket", "float_bucket", "exit_reason", "episode"]:
            rows = []
            for k, g in tt.groupby(col, observed=True):
                d = _core(g); d[col] = k; rows.append(d)
            out[f"by_{col}"] = pd.DataFrame(rows).set_index(col) if rows else pd.DataFrame()
    out["funnel"] = {
        "symbol_days_loaded": res.counters.get("symbol_days", 0),
        "symbol_days_qualified": res.counters.get("symbol_days_qualified", 0),
        "pullback_setups_armed": res.counters.get("episodes_armed", 0),
        "entries_signal": res.counters.get("entries", 0),
        "entries_after_portfolio_rules": len(t),
        "dropped_by_portfolio_rules": len(res.dropped),
        "blocked_not_first_pullback": res.counters.get("blocked_pullback_number", 0),
        "blocked_max_trades_symbol_day": res.counters.get("blocked_max_trades", 0),
        "skipped_size_zero": res.counters.get("skipped_size_zero", 0),
        "setup_broken_before_trigger": res.counters.get("setup_broken_before_trigger", 0),
        "rejected_rr": res.counters.get("rejected_rr", 0),
        "adds": res.counters.get("adds", 0),
        **{k: v for k, v in sorted(res.counters.items()) if k.startswith("ambiguous_")},
    }
    return out


def print_summary(s: dict, notes=()) -> None:
    for n in notes:
        print(f"[note] {n}")
    print("\n=== 漏斗 ===")
    for k, v in s["funnel"].items():
        print(f"  {k:32s} {v}")
    print("\n=== 整体 ===")
    for k, v in s["overall"].items():
        print(f"  {k:32s} {v}")
    with pd.option_context("display.width", 160, "display.max_columns", 20):
        for key in ["by_time_bucket", "by_price_bucket", "by_float_bucket", "by_exit_reason",
                    "by_episode"]:
            if key in s and len(s[key]):
                print(f"\n=== {key} ===")
                print(s[key][["trades", "win_rate", "avg_r", "profit_factor", "total_pnl"]])


def save_results(res: BacktestResult, out_dir) -> dict:
    """把一次回测的结果写进目录，返回 overall 指标（附运行名）。"""
    import json
    from pathlib import Path
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    s = summarize(res)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.dropped.to_csv(out / "dropped_by_portfolio.csv", index=False)
    if "equity" in s:
        s["equity"].rename("cum_pnl").to_csv(out / "equity_daily.csv")
    (out / "config_used.json").write_text(res.cfg.to_json())
    summary_json = {k: v for k, v in s.items() if not k.startswith("by_") and k != "equity"}
    summary_json["notes"] = list(res.notes)
    (out / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2,
                                                 default=str))
    for k, v in s.items():
        if k.startswith("by_") and len(v):
            v.to_csv(out / f"{k}.csv")
    return {**s["overall"], **{f"funnel_{k}": v for k, v in s["funnel"].items()}}

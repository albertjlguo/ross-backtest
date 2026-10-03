#!/usr/bin/env python3
"""
在 2000 年和 2007 年两次高点买入个股，后来怎么样；一次买入 vs 分三批。

    python peak_lab.py --push          # 用 snowball_backtest 已缓存的行情；结果在 results_peak/

范围   当前标普 500 里的非金融、非地产公司中，高点当天已上市、且已在指数里的（纳入日期缺失的按已在算）。
       只有今天还活着的公司 → 朗讯、世通、安然、雷曼这类不在里面，结果偏乐观。
高点   2000-03-24、2007-10-09（标普 500 收盘高点），另加 2021-12-31 作参照
指标   含股息总回报：之后 15 年内最大跌幅、最后一次低于买入价是几年后、5/10/15 年年化、到今天的年化
分组   按高点前 3 年涨幅分五组（涨得最多的那组就是当时的"泡沫股"）——只用高点之前的数据
分批   先买 1/3；较买入价跌 20% 再买 1/3；跌 40% 再买 1/3；没用上的钱按 3 个月国库券利率滚存
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from snowball_backtest import load_px, universe

PEAKS = ["2000-03-24", "2007-10-09", "2021-12-31"]
NAMED = ["MSFT", "CSCO", "INTC", "ORCL", "IBM", "WMT", "KO", "HD", "MRK", "PFE", "JNJ", "PG", "XOM", "GE",
         "QCOM", "AMZN", "AAPL", "TXN", "DIS", "T", "VZ", "PEP", "MCD", "ABT", "MMM", "BA", "AMGN", "ADBE", "NVDA",
         "COST", "UNH", "CVX", "LLY", "NKE", "SBUX", "EBAY", "GILD", "AMAT", "MU", "GLW"]


def ann(x, years):
    return (1 + x) ** (1 / years) - 1 if pd.notna(x) and x > -1 and years > 0 else np.nan


def study(adj: pd.Series, cash: pd.Series, t0: pd.Timestamp, last: pd.Timestamp) -> dict | None:
    s = adj.loc[t0 - pd.Timedelta(days=5):]
    if s.empty or s.index[0] > t0 + pd.Timedelta(days=5):
        return None
    s = s.loc[s.index[s.index <= t0][-1]:] if (s.index <= t0).any() else s
    p0 = s.iloc[0]
    rel = s / p0
    out = {"start": s.index[0].date()}
    pre = adj.loc[:t0]
    past = pre.loc[:t0 - pd.DateOffset(years=3)]
    out["runup_3y"] = p0 / past.iloc[-1] - 1 if len(past) and (t0 - past.index[-1]).days < 3 * 365 + 40 else np.nan
    w15 = rel.loc[:t0 + pd.DateOffset(years=15)]
    out["max_dd"] = w15.min() - 1
    out["dd_date"] = w15.idxmin().date()
    under = rel.index[rel < 1]
    out["yrs_last_below"] = (under[-1] - s.index[0]).days / 365.25 if len(under) else 0.0
    out["still_below_today"] = bool(rel.iloc[-1] < 1)
    for y in (5, 10, 15, 20):
        e = t0 + pd.DateOffset(years=y)
        if e <= last:
            v = rel.loc[:e].iloc[-1] - 1
            out[f"ret_{y}y"] = v; out[f"ann_{y}y"] = ann(v, y)
    yrs = (rel.index[-1] - s.index[0]).days / 365.25
    out["ann_to_today"] = ann(rel.iloc[-1] - 1, yrs)
    # 分三批，10 年后比较（不足 10 年的比到今天）
    e = min(t0 + pd.DateOffset(years=10), last)
    win = s.loc[:e]; c = cash.reindex(win.index, method="ffill")
    cw = (1 + c.fillna(0) / 100 / 252).cumprod()
    end_px, end_c = win.iloc[-1], cw.iloc[-1]
    w = end_px / p0 / 3; adds = 0
    for lvl in (0.8, 0.6):
        hit = win.index[win <= p0 * lvl]
        if len(hit):
            k = hit[0]; w += (cw.loc[k] / cw.iloc[0]) / 3 * (end_px / win.loc[k]); adds += 1
        else:
            w += (end_c / cw.iloc[0]) / 3
    out["lump_10y"] = end_px / p0 - 1; out["staged_10y"] = w - 1; out["adds"] = adds
    out["cash_10y"] = end_c / cw.iloc[0] - 1
    return out


def summarize(d: pd.DataFrame, label: str) -> dict:
    q = lambda c, p: d[c].quantile(p) if c in d and d[c].notna().any() else np.nan
    r = {"group": label, "n": len(d), "max_dd_median": q("max_dd", .5), "max_dd_worst_quarter": q("max_dd", .25),
         "yrs_last_below_median": q("yrs_last_below", .5), "yrs_last_below_p75": q("yrs_last_below", .75),
         "pct_below_after_5y": (d["yrs_last_below"] > 5).mean(), "pct_below_after_10y": (d["yrs_last_below"] > 10).mean(),
         "pct_below_after_15y": (d["yrs_last_below"] > 15).mean(), "pct_still_below_today": d["still_below_today"].mean(),
         "lump_10y_median": q("lump_10y", .5), "staged_10y_median": q("staged_10y", .5),
         "lump_10y_worst": d["lump_10y"].min(), "staged_10y_worst": d["staged_10y"].min(),
         "staged_better_pct": (d["staged_10y"] > d["lump_10y"]).mean(), "avg_adds": d["adds"].mean(),
         "cash_10y": q("cash_10y", .5)}
    for y in (5, 10, 15, 20):
        if f"ann_{y}y" in d:
            r[f"ann_{y}y_median"] = q(f"ann_{y}y", .5); r[f"pct_loss_{y}y"] = (d[f"ret_{y}y"] < 0).mean()
    r["ann_to_today_median"] = q("ann_to_today", .5)
    return r


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data_trend")
    ap.add_argument("--results-dir", default="results_peak")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    dd = Path(a.data_dir); pc = dd / "sb_px"; pc.mkdir(parents=True, exist_ok=True)
    u = universe(dd / "sp500_constituents.csv")
    irx = load_px("^IRX", pc)["close"]                      # 3 个月国库券收益率（%）
    spy = load_px("SPY", pc); last = spy.index.max()
    rows = []
    for _, r in u.iterrows():
        px = load_px(r["yahoo"], pc)
        if px.empty:
            continue
        for pk in PEAKS:
            t0 = pd.Timestamp(pk)
            if pd.notna(r["added"]) and r["added"] > t0:
                continue
            s = study(px["adj"].dropna(), irx, t0, last)
            if s:
                rows.append({"peak": pk, "ticker": r["ticker"], "name": r["name"], "sector": r["sector"], **s})
    for sym in ("SPY", "QQQ"):
        px = load_px(sym, pc)
        for pk in PEAKS:
            s = study(px["adj"].dropna(), irx, pd.Timestamp(pk), last) if len(px) else None
            if s:
                rows.append({"peak": pk, "ticker": sym, "name": sym, "sector": "ETF", **s})
    m = pd.DataFrame(rows); m.to_csv(out / "peak_members.csv", index=False)
    srows = []
    for pk, d in m[m["sector"] != "ETF"].groupby("peak"):
        srows.append({"peak": pk, **summarize(d, "全部")})
        named = d[d["ticker"].isin(NAMED)]
        if len(named):
            srows.append({"peak": pk, **summarize(named, "当时的大公司（名单）")})
        x = d.dropna(subset=["runup_3y"]).copy()
        if len(x) >= 25:
            x["q"] = pd.qcut(x["runup_3y"], 5, labels=["涨得最少", "2", "3", "4", "涨得最多"])
            for qn, g in x.groupby("q", observed=True):
                srows.append({"peak": pk, **summarize(g, f"高点前3年{qn}"), "runup_3y_median": g["runup_3y"].median()})
        for sec, g in d.groupby("sector"):
            if len(g) >= 8:
                srows.append({"peak": pk, **summarize(g, f"行业:{sec}")})
    for _, e in m[m["sector"] == "ETF"].iterrows():
        srows.append({"peak": e["peak"], **summarize(m[(m.ticker == e.ticker) & (m.peak == e.peak)], e["ticker"])})
    pd.DataFrame(srows).to_csv(out / "peak_summary.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"generated": pd.Timestamp.now(tz="UTC").isoformat(),
                                                   "peaks": PEAKS, "last_price_date": str(last.date()), "rows": len(m)}, indent=1))
    print(f"完成：{len(m)} 条。结果在 {out.resolve()}")
    if a.push:
        from pipeline import git_push
        git_push(out, "peak lab")


if __name__ == "__main__":
    main()

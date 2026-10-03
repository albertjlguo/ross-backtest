#!/usr/bin/env python3
"""
历史检验：财务指纹合格（第二道门）且价格到线（第四道门）之后，发生了什么。

    python snowball_backtest.py --push          # 结果在 results_snowball_bt/

规则（跑之前写死，不调）：
  范围     当前标普 500 成分里的非金融、非地产公司（免费数据拿不到已退市公司 → 有幸存者偏差，见 README 段）
  时点     每年 6 月最后一个交易日建一组（12 月财年的年报此时都已披露）；只用当天之前已申报的年报，
           且用"最早披露"的数字，不用后来的重述
  指纹合格 最近 5 个财年：每年 ROIC ≥ 15%、每年所有者收益 > 0、每年营业利润 > 0；
           净负债 ≤ 3 × 营业利润；5 年收入没有下降
  内在价值 所有者收益（维持性开支 = 折旧）按"保守增长"增长 10 年，之后永续 2.5%
           保守增长 = min(8%, 过去 5 年收入年增速的一半)，不为负
  价格到线 市值 ≤ 70% × 内在价值。折现率三档都报：国债 + 2%（下限 6%）/ 固定 8% / 固定 10%
  相对便宜 合格公司里"市值 ÷ 内在价值"最低的五分之一（保证每年都有样本）
  之后     持有 1 / 3 / 5 年的总回报（含分红），对比：全体等权、合格但不便宜、不合格但便宜、SPY

输出：
  cohorts.csv   每年每组的家数、1/3/5 年回报的均值与中位数
  summary.csv   各组跨年份平均，以及"合格且便宜 − 合格但不便宜"的差值和按年份的 t 值
  members.csv   每年每家公司的指纹、估值、分组和之后的回报（第一、三道门你可以拿这张表自己筛）
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from rossbt.sec import SecClient
from snowball.fundamentals import annual_table_pit, cagr, dcf_value, derive

log = logging.getLogger("snowball_bt")
CONSTITUENTS = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"
EXCLUDE_SECTORS = {"Financials", "Real Estate"}
RATES = {"tsy2": None, "r8": 0.08, "r10": 0.10}
HORIZONS = {"1y": 12, "3y": 36, "5y": 60}


def universe(path: Path) -> pd.DataFrame:
    if not path.exists():
        r = requests.get(CONSTITUENTS, timeout=60); r.raise_for_status()
        path.write_text(r.text)
    u = pd.read_csv(path)
    u = u.rename(columns={"Symbol": "ticker", "GICS Sector": "sector", "Security": "name"})
    u["yahoo"] = u["ticker"].str.replace(".", "-", regex=False)
    return u[~u["sector"].isin(EXCLUDE_SECTORS)].reset_index(drop=True)


def load_fund(sec: SecClient, cik: int, cache: Path) -> pd.DataFrame:
    f = cache / f"{cik}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    js = sec.company_facts(cik)
    t = annual_table_pit(js) if js else pd.DataFrame()
    d = derive(t) if len(t) >= 2 else pd.DataFrame()
    if len(d):
        d["filed"] = t["filed"]
    (d if len(d) else pd.DataFrame({"empty": []})).to_parquet(f)
    return d


def load_px(sym: str, cache: Path):
    f = cache / f"{sym}.parquet"
    if f.exists():
        return pd.read_parquet(f)
    import yfinance as yf
    for attempt in range(3):
        try:
            h = yf.Ticker(sym).history(period="max", auto_adjust=False, actions=True)
            break
        except Exception as e:
            log.warning("Yahoo %s: %s", sym, e); time.sleep(5 * (attempt + 1)); h = None
    if h is None or h.empty:
        d = pd.DataFrame({"close": [], "adj": [], "split": []})
    else:
        h.index = pd.DatetimeIndex(h.index).tz_localize(None).normalize()
        d = pd.DataFrame({"close": h["Close"], "adj": h["Adj Close"],
                          "split": h["Stock Splits"] if "Stock Splits" in h else 0.0})
    d.to_parquet(f)
    return d


def at(s: pd.Series, date):
    """date 当天或之前最近一个交易日的值（不超过 10 天前）。"""
    x = s.loc[:date]
    if x.empty or (pd.Timestamp(date) - x.index[-1]).days > 10:
        return np.nan
    return float(x.iloc[-1])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default="data_trend")
    ap.add_argument("--results-dir", default="results_snowball_bt")
    ap.add_argument("--first-year", type=int, default=2014)
    ap.add_argument("--limit", type=int, default=0, help="只跑前 N 家（调试用）")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    dd = Path(a.data_dir); fc = dd / "sb_fund"; pc = dd / "sb_px"
    fc.mkdir(parents=True, exist_ok=True); pc.mkdir(parents=True, exist_ok=True)
    u = universe(dd / "sp500_constituents.csv")
    if a.limit:
        u = u.head(a.limit)
    sec = SecClient()
    spy = load_px("SPY", pc); tnx = load_px("^TNX", pc)
    last_day = spy.index.max()
    cohorts = [spy.loc[f"{y}-06"].index.max() for y in range(a.first_year, last_day.year + 1)
               if len(spy.loc[f"{y}-06"]) and spy.loc[f"{y}-06"].index.max() <= last_day]
    rows = []
    for k, r in u.iterrows():
        try:
            fund = load_fund(sec, int(r["CIK"]), fc)
            px = load_px(r["yahoo"], pc)
        except Exception as e:
            log.warning("%s 失败: %s", r["ticker"], e); continue
        if "filed" not in fund or px.empty:
            continue
        splits = px["split"][px["split"] > 0]
        for c in cohorts:
            f = fund[fund["filed"] <= c]
            if len(f) < 5 or (c - f.index[-1]).days > 550:
                continue
            L, w = f.iloc[-1], f.tail(5)
            p0 = at(px["close"], c)
            if not (p0 > 0) or not (L["shares"] > 0):
                continue
            sf = float(np.prod(splits[splits.index > L["filed"]].to_numpy())) if len(splits) else 1.0
            sf_c = float(np.prod(splits[(splits.index > L["filed"]) & (splits.index <= c)].to_numpy())) if len(splits) else 1.0
            # Yahoo 的收盘价已按之后所有拆股折算：c 日的真实市值 = 折算价 × 折算到今天口径的股数
            mcap = p0 * L["shares"] * sf
            oe = L["owner_earnings"]
            rev_g = cagr(f["revenue"], 5)
            g_c = max(0.0, min(0.08, (rev_g if pd.notna(rev_g) else 0.0) / 2))
            t10 = at(tnx["close"], c) / 100
            quality = bool((w["roic"] >= 0.15).all() and (w["owner_earnings"] > 0).all()
                           and (w["op_income"] > 0).all() and pd.notna(rev_g) and rev_g >= 0
                           and L["net_debt"] <= 3 * L["op_income"])
            row = {"cohort": c.date(), "ticker": r["ticker"], "name": r["name"], "sector": r["sector"],
                   "fy_end": f.index[-1].date(), "filed": L["filed"].date(), "mcap_bn": mcap / 1e9,
                   "split_factor_after": sf, "roic_min5": w["roic"].min(), "roic_med5": w["roic"].median(),
                   "op_margin": L["op_margin"], "rev_cagr5": rev_g, "net_debt_to_oi": L["net_debt"] / L["op_income"] if L["op_income"] > 0 else np.nan,
                   "owner_earnings_bn": oe / 1e9, "oe_yield": oe / mcap, "tsy10": t10, "g_cons": g_c,
                   "quality": quality}
            for name, rr in RATES.items():
                disc = max(0.06, t10 + 0.02) if rr is None else rr
                iv = dcf_value(oe, g_c, disc) if oe > 0 else np.nan
                row[f"p_iv_{name}"] = mcap / iv if pd.notna(iv) and iv > 0 else np.nan
                row[f"cheap_{name}"] = bool(pd.notna(iv) and mcap <= 0.7 * iv)
            a0 = at(px["adj"], c)
            for h, m in HORIZONS.items():
                e = c + pd.DateOffset(months=m)
                a1 = at(px["adj"], e) if e <= last_day else np.nan
                row[f"ret_{h}"] = a1 / a0 - 1 if a0 > 0 and pd.notna(a1) else np.nan
            rows.append(row)
        if (k + 1) % 25 == 0:
            log.info("%d / %d 家", k + 1, len(u))
            pd.DataFrame(rows).to_csv(out / "members.csv", index=False)
    m = pd.DataFrame(rows)
    # 相对便宜：合格公司里 p_iv_r8 最低的五分之一
    m["rel_cheap"] = False
    for c, g in m[m["quality"] & m["p_iv_r8"].notna()].groupby("cohort"):
        if len(g) >= 5:
            m.loc[g.index[g["p_iv_r8"] <= g["p_iv_r8"].quantile(0.2)], "rel_cheap"] = True
    m.to_csv(out / "members.csv", index=False)

    spy_ret = {}
    for c in cohorts:
        a0 = at(spy["adj"], c)
        spy_ret[c.date()] = {h: (at(spy["adj"], c + pd.DateOffset(months=mm)) / a0 - 1
                                 if c + pd.DateOffset(months=mm) <= last_day else np.nan) for h, mm in HORIZONS.items()}
    groups = {"all": lambda d: d["quality"].notna(),
              "quality": lambda d: d["quality"],
              "not_quality": lambda d: ~d["quality"],
              "q_rel_cheap": lambda d: d["rel_cheap"],
              "q_not_rel_cheap": lambda d: d["quality"] & ~d["rel_cheap"]}
    for name in RATES:
        groups[f"q_cheap_{name}"] = (lambda n: lambda d: d["quality"] & d[f"cheap_{n}"])(name)
        groups[f"q_notcheap_{name}"] = (lambda n: lambda d: d["quality"] & ~d[f"cheap_{n}"])(name)
        groups[f"nq_cheap_{name}"] = (lambda n: lambda d: ~d["quality"] & d[f"cheap_{n}"])(name)
    crow = []
    for c, d in m.groupby("cohort"):
        for gname, fn in groups.items():
            x = d[fn(d)]
            rec = {"cohort": c, "group": gname, "n": len(x)}
            for h in HORIZONS:
                rec[f"mean_{h}"] = x[f"ret_{h}"].mean(); rec[f"median_{h}"] = x[f"ret_{h}"].median()
                rec[f"pct_loss_{h}"] = (x[f"ret_{h}"] < 0).mean() if x[f"ret_{h}"].notna().any() else np.nan
            crow.append(rec)
        crow.append({"cohort": c, "group": "SPY", "n": 1, **{f"mean_{h}": spy_ret[c][h] for h in HORIZONS}})
    co = pd.DataFrame(crow); co.to_csv(out / "cohorts.csv", index=False)

    srow = []
    piv = {h: co.pivot(index="cohort", columns="group", values=f"mean_{h}") for h in HORIZONS}
    nn = co.pivot(index="cohort", columns="group", values="n")
    for g in co["group"].unique():
        rec = {"group": g, "avg_n": nn[g].mean(), "cohorts_with_names": int((nn[g] > 0).sum())}
        for h in HORIZONS:
            s = piv[h][g].dropna()
            rec[f"avg_{h}"] = s.mean(); rec[f"cohorts_{h}"] = len(s)
        srow.append(rec)
    pairs = [("q_rel_cheap", "q_not_rel_cheap"), ("quality", "all"), ("quality", "SPY"), ("q_rel_cheap", "SPY")] + \
            [(f"q_cheap_{n}", f"q_notcheap_{n}") for n in RATES] + [(f"q_cheap_{n}", "SPY") for n in RATES] + \
            [(f"q_cheap_{n}", f"nq_cheap_{n}") for n in RATES]
    for x, y in pairs:
        rec = {"group": f"{x} − {y}"}
        for h in HORIZONS:
            d = (piv[h][x] - piv[h][y]).dropna()
            rec[f"avg_{h}"] = d.mean(); rec[f"cohorts_{h}"] = len(d)
            rec[f"t_{h}"] = d.mean() / d.std(ddof=1) * np.sqrt(len(d)) if len(d) > 2 and d.std(ddof=1) > 0 else np.nan
            rec[f"pct_cohorts_pos_{h}"] = (d > 0).mean() if len(d) else np.nan
        srow.append(rec)
    pd.DataFrame(srow).to_csv(out / "summary.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"generated": pd.Timestamp.now(tz="UTC").isoformat(),
        "universe": len(u), "cohorts": [str(c.date()) for c in cohorts], "rows": len(m)}, indent=1))
    print(f"\n完成：{len(u)} 家，{len(cohorts)} 个年份，{len(m)} 条记录。结果在 {out.resolve()}")
    if a.push:
        from pipeline import git_push
        git_push(out, "snowball backtest")


if __name__ == "__main__":
    main()

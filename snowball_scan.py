#!/usr/bin/env python3
"""
/snowball 的第二道门（财务指纹）和第四道门（价格算术），用 SEC 免费财报数据。

    python snowball_scan.py --push                 # 自选 18 个里在 SEC 有财报的；结果在 results_snowball/
    python snowball_scan.py --tickers MA,V,AAPL    # 任意美股

输出：
  gate2_quality.csv   每家公司的 10 年财务指纹（ROIC、增量 ROIC、利润率稳定性、负债、股本变化）
  gate4_price.csv     所有者收益（基准/压力）、收益率、现价隐含增长 vs 历史增长、70% 安全边际线
  annual.csv          逐年原始科目和派生指标（查错用）
第一道门（看不看得懂）和第三道门（管理层）不在这里——那是人的判断。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rossbt.sec import SecClient, norm_ticker
from snowball.fundamentals import annual_table, cagr, dcf_value, derive, implied_growth

DEFAULT = ["MA", "AXP", "BRK-B", "META", "SHOP", "COIN", "NET", "TSLA", "SPCX", "ABNB", "MAR", "HLT", "H",
           "V", "GOOGL", "AMZN", "MSFT", "AAPL"]          # 后 5 个是对照
FINANCIAL = {"AXP", "BRK-B", "JPM", "BAC", "WFC", "GS", "COIN"}   # COIN 的现金流混着客户资金
SHARE_MULT = {"BRK-B": 1500.0}                                     # BRK 的加权股数按 A 股计，折成 B 股
PRICE_FILE = {"BRK-B": "BRK-B"}


def last_price(data_dir: Path, t: str):
    f = data_dir / "daily" / f"{t}.parquet"
    if f.exists():
        d = pd.read_parquet(f)
        return float(d["close"].iloc[-1]), str(d.index[-1].date())
    try:
        import yfinance as yf
        h = yf.Ticker(t).history(period="5d", auto_adjust=False)
        return float(h["Close"].iloc[-1]), str(h.index[-1].date())
    except Exception:
        return np.nan, None


def r3(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tickers", default=",".join(DEFAULT))
    ap.add_argument("--data-dir", default="data_trend")
    ap.add_argument("--results-dir", default="results_snowball")
    ap.add_argument("--treasury", type=float, default=0.0524, help="10 年期美债收益率")
    ap.add_argument("--premium", type=float, default=0.02, help="折现率 = 国债 + 这个溢价")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    sec = SecClient(cache_dir=Path(a.data_dir) / "sec_facts")
    tmap = sec.ticker_map()
    r = a.treasury + a.premium
    q_rows, p_rows, ann = [], [], []
    for t in [x.strip().upper() for x in a.tickers.split(",") if x.strip()]:
        cik = tmap.get(norm_ticker(t))
        if cik is None:
            q_rows.append({"ticker": t, "note": "SEC 代码表里没有（非美国申报人或代码不同）"}); continue
        js = sec.company_facts(cik)
        raw = annual_table(js)
        if raw.empty or raw["net_income"].notna().sum() < 2:
            q_rows.append({"ticker": t, "note": "没有可用的 us-gaap 年度数据（可能按 IFRS 申报）"}); continue
        fin = t in FINANCIAL
        d = derive(raw, financial=fin)
        ann.append(d.assign(ticker=t).reset_index())
        L = d.iloc[-1]; n = len(d)
        px, px_date = last_price(Path(a.data_dir), PRICE_FILE.get(t, t))
        shares = L["shares"] * SHARE_MULT.get(t, 1.0) if pd.notna(L["shares"]) else np.nan
        mcap = px * shares if pd.notna(px) and pd.notna(shares) else np.nan
        base = {"ticker": t, "fy_end": str(d.index[-1].date()), "years": n, "financial": fin}
        if fin:
            roe = d["roe"].dropna().tail(10)
            q_rows.append({**base, "roe_10y_median": r3(roe.median()), "roe_min": r3(roe.min()),
                           "roe_latest": r3(L["roe"]), "years_roe_ge_15": int((roe >= .15).sum()),
                           "shares_chg_5y": r3(d["shares"].iloc[-1] / d["shares"].iloc[-6] - 1) if n > 5 else None})
            p_rows.append({**base, "price": px, "price_date": px_date, "mcap_bn": r3(mcap / 1e9),
                           "pb": r3(mcap / L["equity"]) if pd.notna(mcap) else None,
                           "pe": r3(mcap / L["net_income"]) if pd.notna(mcap) and L["net_income"] > 0 else None,
                           "earnings_yield": r3(L["net_income"] / mcap) if pd.notna(mcap) else None,
                           "implied_growth_10y": r3(implied_growth(mcap, L["net_income"], r)),
                           "ni_ps_cagr_10y": r3(cagr(d["net_income"] / d["shares"], min(10, n - 1)))})
            continue
        roic = d["roic"].dropna().tail(10); om = d["op_margin"].dropna().tail(10)
        q_rows.append({**base,
                       "roic_10y_median": r3(roic.median()), "roic_min": r3(roic.min()), "roic_latest": r3(L["roic"]),
                       "years_roic_ge_15": int((roic >= .15).sum()), "years_counted": int(len(roic)),
                       "roic_incr_5y": r3(L["roic_incr_5y"]),
                       "op_margin_median": r3(om.median()), "op_margin_min": r3(om.min()),
                       "op_margin_std": r3(om.std()),
                       "rev_cagr_5y": r3(cagr(d["revenue"], 5)), "rev_cagr_10y": r3(cagr(d["revenue"], 10)),
                       "capex_to_dep_3y": r3(d["capex_to_dep"].tail(3).mean()),
                       "maint_share_of_capex": r3(L["maint_base"] / L["capex"]) if L["capex"] else None,
                       "net_debt_to_op_income": r3(L["net_debt"] / L["op_income"]) if L["op_income"] > 0 else None,
                       "interest_cover": r3(L["interest_cover"]),
                       "sbc_to_revenue": r3(L["sbc"] / L["revenue"]),
                       "shares_chg_5y": r3(d["shares"].iloc[-1] / d["shares"].iloc[-6] - 1) if n > 5 else None,
                       "pos_oe_years_10": int((d["owner_earnings"].tail(10) > 0).sum())})
        oe, oes = L["owner_earnings"], L["owner_earnings_stress"]
        hist_g = cagr(d["oe_ps"], min(10, n - 1)); rev_g = cagr(d["revenue"], min(10, n - 1))
        cons_g = np.nanmin([0.08, (rev_g if pd.notna(rev_g) else 0.0) / 2])     # 保守增长：历史收入增速的一半，封顶 8%
        cons_g = max(cons_g, 0.0)
        iv, ivs = dcf_value(oe, cons_g, r), dcf_value(oes, cons_g, r)
        p_rows.append({**base, "price": px, "price_date": px_date, "mcap_bn": r3(mcap / 1e9),
                       "net_income_bn": r3(L["net_income"] / 1e9), "da_bn": r3(L["da"] / 1e9),
                       "capex_bn": r3(L["capex"] / 1e9), "sbc_bn": r3(L["sbc"] / 1e9),
                       "maint_dep_bn": r3(L["dep"] / 1e9), "maint_greenwald_bn": r3(L["maint_greenwald"] / 1e9),
                       "maint_full_bn": r3(L["maint_full"] / 1e9), "useful_life_yrs": r3(L["useful_life"]),
                       "owner_earnings_bn": r3(oe / 1e9), "owner_earnings_stress_bn": r3(oes / 1e9),
                       "fcf_bn": r3(L["fcf"] / 1e9), "fcf_ex_sbc_bn": r3(L["fcf_ex_sbc"] / 1e9),
                       "oe_yield": r3(oe / mcap), "oe_yield_stress": r3(oes / mcap),
                       "yield_minus_treasury": r3(oe / mcap - a.treasury),
                       "implied_growth_10y": r3(implied_growth(mcap, oe, r)),
                       "implied_growth_10y_stress": r3(implied_growth(mcap, oes, r)),
                       "oe_ps_cagr_hist": r3(hist_g), "rev_cagr_hist": r3(rev_g),
                       "conservative_growth_used": r3(cons_g),
                       "price_to_iv": r3(mcap / iv) if pd.notna(iv) else None,
                       "price_to_iv_stress": r3(mcap / ivs) if pd.notna(ivs) else None,
                       "passes_70pct": bool(pd.notna(iv) and mcap <= 0.7 * iv)})
        print("done", t)
    pd.DataFrame(q_rows).to_csv(out / "gate2_quality.csv", index=False)
    pd.DataFrame(p_rows).to_csv(out / "gate4_price.csv", index=False)
    if ann:
        pd.concat(ann, ignore_index=True).to_csv(out / "annual.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({"generated": pd.Timestamp.now(tz="UTC").isoformat(),
                                                   "treasury": a.treasury, "discount": r, "terminal_growth": 0.025,
                                                   "tax": 0.21}, indent=1))
    print(f"\n完成。结果在 {out.resolve()}")
    if a.push:
        from pipeline import git_push
        git_push(out, "snowball scan")


if __name__ == "__main__":
    main()

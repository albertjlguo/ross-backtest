#!/usr/bin/env python3
"""
只做多趋势 + 低位拐头向上时加到 4 倍（不加杠杆，平时留现金）。

    python boost_lab.py --push            # 结果在 results_boost/

对照：
  base      趋势向上就持有 1 份（= 资金上限的 25%），从不加注
  full      趋势向上就持有 4 份（满额，等于 base 的 4 倍，夏普相同）
  boost_k   平时 1 份；趋势翻上且当时离一年高点回撤 ≥ k×年化波动率 → 加到 4 份，创一年新高或趋势转弱时退回
            k = 0.5 / 1.0 / 1.5 三档都报告，不挑。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from trendbt.lab import PERIODS, cut, panel, run_boost, stats

KS = [0.5, 1.0, 1.5]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--data-dir", default="data_trend")
    ap.add_argument("--results-dir", default="results_boost")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    u = json.loads(Path(a.universe).read_text())
    groups, cost, daily = {}, {}, {}
    dup = {"SPX", "NDX", "RUT", "DJI"}
    for g, assets in u.items():
        if g.startswith("_"):
            continue
        for n, s in assets.items():
            f = Path(a.data_dir) / "daily" / f"{n}.parquet"
            if f.exists() and n not in dup:
                daily[n] = pd.read_parquet(f); cost[n] = s.get("cost_bps", 5.0)
                groups.setdefault(g, []).append(n)
    px = panel(daily); cost = pd.Series(cost)
    main_g = [g for g in groups if g != "watch18"]
    groups["ALL_ex_stocks"] = [n for g in main_g if g != "stocks" for n in groups[g]]
    groups["ALL"] = [n for g in main_g for n in groups[g]]
    rows, mon = [], {}
    for g, names in groups.items():
        variants = {"base": dict(k=None, base=0.25), "full": dict(k=None, base=1.0),
                    **{f"boost_{k}": dict(k=k, base=0.25) for k in KS}}
        for v, kw in variants.items():
            res = run_boost(px[names], cost, **kw)
            r = res.pop("ret")
            mon[f"{v}|{g}"] = (1 + r).groupby(r.index.to_period("M")).prod() - 1
            for per in PERIODS:
                rows.append({"group": g, "variant": v, "period": per, "assets": len(names), **stats(cut(r, per)),
                             **({k2: round(v2, 4) if isinstance(v2, float) else v2 for k2, v2 in res.items()}
                                if per == "FULL" else {})})
            print("done", g, v)
    pd.DataFrame(rows).to_csv(out / "boost_summary.csv", index=False)
    pd.DataFrame(mon).to_csv(out / "boost_monthly.csv")
    print(f"\n完成。结果在 {out.resolve()}")
    if a.push:
        from pipeline import git_push
        git_push(out, "boost lab")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
因子实验室：在已下载的全部品种上检验 8 类系统化策略（参数写死，不调参）。

    python trend_pipeline.py download --no-hourly        # 如果 data_trend/ 里还没有日线
    python alpha_lab.py --push                           # 跑全部，结果在 results_lab/

输出：
  lab_summary.csv   策略 × 品种组 × 时段（IS/OOS/HOLD/FULL）的年化、夏普、t 值、回撤、月胜率、换手
  lab_pass.csv      三个时段夏普都为正、且全样本 t 值过多重检验门槛的组合
  lab_combo.csv     通过的组合等风险合并后的表现，以及 2 倍、3 倍杠杆下的回撤
  lab_corr.csv      各策略（全品种）月收益相关性
  lab_monthly.csv   每个组合的月收益
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from trendbt.lab import PERIODS, STRATS, XS, LabParams, bonferroni_t, cut, panel, run, stats

SKIP_GROUPS = {"watch18"}          # 与其他组重复，且是事后挑的


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--data-dir", default="data_trend")
    ap.add_argument("--results-dir", default="results_lab")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.results_dir); out.mkdir(parents=True, exist_ok=True)
    u = json.loads(Path(a.universe).read_text())
    groups, cost, daily = {}, {}, {}
    for g, assets in u.items():
        if g.startswith("_") or g in SKIP_GROUPS:
            continue
        for n, s in assets.items():
            f = Path(a.data_dir) / "daily" / f"{n}.parquet"
            if f.exists():
                daily[n] = pd.read_parquet(f); cost[n] = s.get("cost_bps", 5.0)
                groups.setdefault(g, []).append(n)
    if not daily:
        raise SystemExit("data_trend/daily 里没有数据，先跑：python trend_pipeline.py download --no-hourly")
    px = panel(daily); cost = pd.Series(cost); p = LabParams()
    # 指数（SPX/NDX/RUT/DJI）与对应 ETF 重复，组合里去掉
    dup = {"SPX", "NDX", "RUT", "DJI"}
    groups = {g: [n for n in v if n not in dup] for g, v in groups.items()}
    groups["ALL"] = [n for v in groups.values() for n in v]
    groups["ALL_ex_stocks"] = [n for g, v in groups.items() if g not in ("stocks", "ALL") for n in v]
    print({g: len(v) for g, v in groups.items()}, px.index.min().date(), "→", px.index.max().date())

    rets, rows = {}, []
    for g, names in groups.items():
        for st in STRATS:
            if st in XS and g.startswith("ALL"):
                continue                      # 横截面排名只在同类品种内做
            res = run(px[names], st, cost, p)
            key = f"{st}|{g}"
            rets[key] = res["ret"]
            for per in PERIODS:
                rows.append({"strategy": st, "group": g, "period": per, "assets": len(names),
                             **stats(cut(res["ret"], per)),
                             **({"gross_lev": round(res["gross"], 2), "turnover_x": round(res["turnover"], 1),
                                 "cost_drag": round(res["cost_drag"], 4)} if per == "FULL" else {})})
            print("done", key)
    summ = pd.DataFrame(rows); summ.to_csv(out / "lab_summary.csv", index=False)
    n_trials = len(rets); thr = bonferroni_t(n_trials)
    piv = summ.pivot_table(index=["strategy", "group"], columns="period", values="sharpe")
    full = summ[summ.period == "FULL"].set_index(["strategy", "group"])
    ps = piv.join(full[["t", "ann_ret", "max_dd", "pct_pos_months", "worst_month", "turnover_x", "cost_drag"]])
    ps["all_periods_positive"] = (ps[["IS", "OOS", "HOLD"]] > 0).all(axis=1)
    ps["t_threshold"] = thr
    ps["passes"] = ps["all_periods_positive"] & (ps["t"] > thr) & (ps.index.get_level_values(0) != "bh_vt")
    ps.sort_values("FULL", ascending=False).to_csv(out / "lab_pass.csv")

    mon = pd.DataFrame({k: (1 + r).groupby(r.index.to_period("M")).prod() - 1 for k, r in rets.items()})
    mon.to_csv(out / "lab_monthly.csv")
    mon[[c for c in mon if c.endswith("|ALL") or c.split("|")[0] in XS]].corr().round(2).to_csv(out / "lab_corr.csv")

    # 组合：通过检验的（每个策略只取 t 值最高的那个组），日收益等权合并，再调到 10% 波动率
    passed = ps[ps["passes"]].reset_index().sort_values("t", ascending=False).drop_duplicates("strategy")
    crow = []
    sets = {"passed": [f"{r.strategy}|{r.group}" for r in passed.itertuples()],
            "prior_trend+bh": ["trend4|ALL_ex_stocks", "bh_vt|ALL_ex_stocks"]}
    for name, keys in sets.items():
        keys = [k for k in keys if k in rets]
        if not keys:
            continue
        c = pd.DataFrame({k: rets[k] for k in keys}).dropna(how="all").fillna(0.0).mean(axis=1)
        rv = c.ewm(span=p.vol_span, min_periods=p.vol_span).std() * np.sqrt(252)
        base = c * (p.target_vol / rv).clip(upper=3).shift(1).fillna(0.0)
        for lev in (1, 2, 3):
            for per in PERIODS:
                crow.append({"combo": name, "members": ";".join(keys), "leverage": lev, "period": per,
                             **stats(cut(base * lev, per))})
    pd.DataFrame(crow).to_csv(out / "lab_combo.csv", index=False)
    (out / "run_info.json").write_text(json.dumps({
        "generated": pd.Timestamp.now(tz="UTC").isoformat(), "trials": n_trials, "t_threshold": thr,
        "params": p.__dict__, "groups": groups}, ensure_ascii=False, indent=1))
    print(f"\n完成：{n_trials} 个组合，多重检验门槛 |t| > {thr}，通过 {int(ps['passes'].sum())} 个。结果在 {out.resolve()}")
    if a.push:
        from pipeline import git_push
        git_push(out, "factor lab")


if __name__ == "__main__":
    main()

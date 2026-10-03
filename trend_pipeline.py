#!/usr/bin/env python3
"""
长周期趋势方法研究（月线均线 + KD/MACD + 斐波那契）。

    python trend_pipeline.py all --push          # 下载 + 全部研究 + 推送 results_trend/
    python trend_pipeline.py download            # 只下载
    python trend_pipeline.py study               # 用已下载的数据跑全部研究
    python trend_pipeline.py study --only scan,csp --groups crypto,etf_index

研究内容（结果都在 results_trend/）：
  scan        同一套规则在 月/周/日/小时 上跑，4 种支撑位 × 有无 KD 确认 × 3 种离场 → tf_scan.csv
  fib_null    斐波那契比例 vs 任意比例（同样 3 个价位）→ fib_null.csv
  rand_null   健康状态下随机入场、同样离场 → random_null.csv
  mtf         月线定方向 + 日线入场 → mtf.csv
  csp         支撑位卖看跌期权（模型定价）→ csp.csv
  vr          方差比：各品种在哪个周期上更像趋势 → vr.csv
  per_asset   逐个品种：月/周/日线上"任一支撑位 + 结构止损" vs 买入持有 → per_asset.csv、bucket_portfolio.csv
  snapshot    现在的状态：健康与否、斐波那契位、均线、离最近支撑多远、是否在持仓 → snapshot.csv

自选 18 个标的（核心 / 抄底 / 观望 + BTC）：
    python trend_pipeline.py all --groups watch18 --results-dir results_watch18 --push
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from trendbt.data import fetch_daily, fetch_hourly
from trendbt.options import CSPParams, simulate_csp
from trendbt.stats import bar_to_monthly, metrics, portfolio_monthly, trade_stats, variance_ratio
from trendbt.strategy import MTFParams, StratParams, healthy, prepare, simulate, simulate_mtf
from trendbt.structure import fib_level

log = logging.getLogger("trend")

TFS = ["M", "W", "D", "H"]
ENTRIES = [("ma", False), ("fib", False), ("confluence", False), ("any", False), ("any", True)]
EXITS = ["struct", "target", "momo"]
FIB_SETS = {"fib": (0.382, 0.5, 0.618), "alt_a": (0.33, 0.45, 0.56), "alt_b": (0.42, 0.55, 0.68),
            "alt_c": (0.30, 0.47, 0.64)}
RAND_P = {"M": 0.15, "W": 0.05, "D": 0.012}


class TrendStudy:
    def __init__(self, universe: dict, data_dir, results_dir, use_hourly=True, alpaca=None,
                 groups=None, assets=None):
        self.u = {g: {n: s for n, s in a.items() if not assets or n in assets}
                  for g, a in universe.items() if not g.startswith("_") and (not groups or g in groups)}
        self.d = Path(data_dir); self.r = Path(results_dir)
        (self.d / "daily").mkdir(parents=True, exist_ok=True)
        (self.d / "hourly").mkdir(parents=True, exist_ok=True)
        self.r.mkdir(parents=True, exist_ok=True)
        self.use_hourly = use_hourly
        self._alp = alpaca
        self._bars = {}

    @property
    def alp(self):
        if self._alp is None:
            from rossbt.alpaca import AlpacaClient
            self._alp = AlpacaClient()
        return self._alp

    def items(self):
        for g, a in self.u.items():
            for n, s in a.items():
                yield g, n, s

    # ------------------------------------------------------------------ #
    def download(self, force=False):
        cov = []
        for g, n, s in self.items():
            p = self.d / "daily" / f"{n}.parquet"
            if force or not p.exists():
                df = fetch_daily(s)
                if not df.empty:
                    df.to_parquet(p)
            df = pd.read_parquet(p) if p.exists() else pd.DataFrame()
            row = {"group": g, "asset": n, "daily_from": str(df.index.min().date()) if len(df) else None,
                   "daily_to": str(df.index.max().date()) if len(df) else None, "daily_rows": len(df)}
            if self.use_hourly and (s.get("alpaca") or s.get("alpaca_crypto")):
                ph = self.d / "hourly" / f"{n}.parquet"
                if force or not ph.exists():
                    try:
                        h = fetch_hourly(s, self.alp)
                        if not h.empty:
                            h.to_parquet(ph)
                    except Exception as e:
                        log.warning("小时线 %s 失败: %s", n, e)
                if ph.exists():
                    h = pd.read_parquet(ph)
                    row.update(hourly_from=str(h.index.min()), hourly_rows=len(h))
            cov.append(row)
            log.info("数据 %s/%s: %s", g, n, row)
        pd.DataFrame(cov).to_csv(self.r / "data_coverage.csv", index=False)

    def daily(self, n):
        p = self.d / "daily" / f"{n}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def hourly(self, n):
        p = self.d / "hourly" / f"{n}.parquet"
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()

    def bars(self, n, tf):
        k = (n, tf)
        if k not in self._bars:
            src = self.hourly(n) if tf == "H" else self.daily(n)
            self._bars[k] = prepare(src, tf) if len(src) > 200 else pd.DataFrame()
        return self._bars[k]

    @staticmethod
    def ready_from(b: pd.DataFrame):
        ok = b["H"].notna() & b["ma80"].notna()
        return b.index[ok.to_numpy().argmax()] if ok.any() else None

    # ------------------------------------------------------------------ #
    def _run(self, tf, sp_base: StratParams, group_filter=None):
        """对某个周期、某组参数跑所有品种，返回 {group: (trades, sleeves, bh, exposure)}。"""
        out = {}
        for g, n, s in self.items():
            if group_filter and g != group_filter:
                continue
            b = self.bars(n, tf)
            if b.empty:
                continue
            t0 = self.ready_from(b)
            if t0 is None:
                continue
            sp = sp_base.with_(cost_bps=s.get("cost_bps", 5.0))
            tr, ret = simulate(b, sp, n, tf)
            ret = ret[ret.index >= t0]
            o = out.setdefault(g, {"trades": [], "sleeves": {}, "bh": {}, "exposure": []})
            o["trades"] += tr
            o["sleeves"][n] = bar_to_monthly(ret)
            bh = b["close"][b.index >= t0].pct_change().fillna(0)
            o["bh"][n] = bar_to_monthly(bh)
            o["exposure"].append(sum(x["bars"] for x in tr) / max(1, len(ret)))
        return out

    def _summ(self, o):
        tr = pd.DataFrame(o["trades"])
        pm = portfolio_monthly(o["sleeves"])
        st = {**trade_stats(tr), **metrics(pm)}
        st["exposure"] = round(float(np.mean(o["exposure"])), 3) if o["exposure"] else np.nan
        st["trades_per_month"] = round(len(tr) / max(1, len(pm)), 3)
        st["assets"] = len(o["sleeves"])
        return st, tr

    def scan(self):
        rows, all_tr, bh_rows = [], [], []
        for tf in TFS:
            for entry, conf in ENTRIES:
                for ex in EXITS:
                    t0 = time.time()
                    res = self._run(tf, StratParams(entry=entry, exit=ex, confirm=conf))
                    for g, o in res.items():
                        st, tr = self._summ(o)
                        rows.append({"group": g, "tf": tf, "entry": entry + ("+kd" if conf else ""),
                                     "exit": ex, **st})
                        if len(tr):
                            all_tr.append(tr.assign(group=g, entry=entry + ("+kd" if conf else ""), exit=ex))
                        if entry == "any" and not conf and ex == "struct":
                            bh_rows.append({"group": g, "tf": tf, **metrics(portfolio_monthly(o["bh"]))})
                    log.info("scan %s %s%s %s (%.0fs)", tf, entry, "+kd" if conf else "", ex, time.time() - t0)
            pd.DataFrame(rows).to_csv(self.r / "tf_scan.csv", index=False)
        pd.DataFrame(bh_rows).to_csv(self.r / "buy_hold.csv", index=False)
        if all_tr:
            pd.concat(all_tr, ignore_index=True).to_csv(self.r / "trades_scan.csv.gz", index=False)

    def fib_null(self):
        rows = []
        for tf in ["M", "W", "D"]:
            for name, lv in FIB_SETS.items():
                for ex in ["struct", "target"]:
                    res = self._run(tf, StratParams(entry="fib", exit=ex, fib_levels=lv))
                    tr = pd.DataFrame([t for o in res.values() for t in o["trades"]])
                    rows.append({"tf": tf, "levels": name, "ratios": lv, "exit": ex, **trade_stats(tr)})
                    log.info("fib_null %s %s %s", tf, name, ex)
        pd.DataFrame(rows).to_csv(self.r / "fib_null.csv", index=False)

    def rand_null(self, seeds=10):
        rows = []
        for tf in ["M", "W", "D"]:
            for seed in range(seeds):
                res = self._run(tf, StratParams(entry="random", exit="struct", random_p=RAND_P[tf], seed=seed))
                tr = pd.DataFrame([t for o in res.values() for t in o["trades"]])
                pm = portfolio_monthly({k: v for o in res.values() for k, v in o["sleeves"].items()})
                rows.append({"tf": tf, "seed": seed, **trade_stats(tr), **metrics(pm)})
            res = self._run(tf, StratParams(entry="any", exit="struct"))
            tr = pd.DataFrame([t for o in res.values() for t in o["trades"]])
            pm = portfolio_monthly({k: v for o in res.values() for k, v in o["sleeves"].items()})
            rows.append({"tf": tf, "seed": "ACTUAL_any_struct", **trade_stats(tr), **metrics(pm)})
            log.info("rand_null %s", tf)
        pd.DataFrame(rows).to_csv(self.r / "random_null.csv", index=False)

    def mtf(self):
        rows, all_tr = [], []
        for ex in ["momo", "r2", "monthly"]:
            per_g = {}
            for g, n, s in self.items():
                dly = self.daily(n)
                if len(dly) < 2000:
                    continue
                tr, ret = simulate_mtf(dly, MTFParams(exit=ex, cost_bps=s.get("cost_bps", 5.0)), n)
                m = self.bars(n, "M")
                t0 = self.ready_from(m) if not m.empty else None
                if t0 is None:
                    continue
                ret = ret[ret.index > t0]
                o = per_g.setdefault(g, {"trades": [], "sleeves": {}, "exposure": []})
                o["trades"] += [t for t in tr if t["entry_t"] > t0]
                o["sleeves"][n] = bar_to_monthly(ret)
                o["exposure"].append(sum(x["bars"] for x in tr) / max(1, len(ret)))
            for g, o in per_g.items():
                st, tr = self._summ(o)
                rows.append({"group": g, "exit": ex, **st})
                if len(tr):
                    all_tr.append(tr.assign(group=g, exit=ex))
            log.info("mtf %s", ex)
        pd.DataFrame(rows).to_csv(self.r / "mtf.csv", index=False)
        if all_tr:
            pd.concat(all_tr, ignore_index=True).to_csv(self.r / "trades_mtf.csv.gz", index=False)

    def csp(self):
        rows, detail = [], []
        for mode in ["signal", "delta", "always"]:
            per_g = {}
            for g, n, s in self.items():
                if not s.get("options"):
                    continue
                dly = self.daily(n)
                if len(dly) < 2000:
                    continue
                df = simulate_csp(dly, CSPParams(mode=mode, ann_days=s.get("ann_days", 252)), n)
                if df.empty:
                    continue
                m = self.bars(n, "M")
                t0 = self.ready_from(m) if not m.empty else None
                if t0 is None:
                    continue
                df = df[pd.PeriodIndex(df["month"], freq="M") > t0.to_period("M")]
                detail.append(df.assign(mode=mode, group=g))
                per_g.setdefault(g, {})[n] = df.set_index(pd.PeriodIndex(df["month"], freq="M"))["ret"]
            for g, sl in per_g.items():
                pm = portfolio_monthly(sl)
                d = pd.concat([x for x in detail if x["mode"].iloc[0] == mode and x["group"].iloc[0] == g])
                sold = d[d["sold"]]
                rows.append({"group": g, "mode": mode, **metrics(pm),
                             "assets": len(sl), "pct_asset_months_sold": round(d["sold"].mean(), 3),
                             "avg_premium_pct": round(sold["premium_pct"].mean(), 4) if len(sold) else np.nan,
                             "assign_rate": round(sold["assigned"].mean(), 3) if len(sold) else np.nan,
                             "avg_ret_when_sold": round(sold["ret"].mean(), 4) if len(sold) else np.nan,
                             "worst_single": round(sold["ret"].min(), 4) if len(sold) else np.nan})
            log.info("csp %s", mode)
        pd.DataFrame(rows).to_csv(self.r / "csp.csv", index=False)
        if detail:
            pd.concat(detail, ignore_index=True).to_csv(self.r / "csp_detail.csv.gz", index=False)

    def vr(self):
        rows = []
        for g, n, s in self.items():
            dly = self.daily(n)
            if len(dly) > 500:
                lr = np.log(dly["close"]).diff()
                for q in (2, 5, 10, 21, 63, 126, 252):
                    v, z = variance_ratio(lr, q)
                    rows.append({"group": g, "asset": n, "base": "D", "q": q, "vr": v, "z": z})
            h = self.hourly(n)
            if len(h) > 2000:
                lr = np.log(h["close"]).diff()
                for q in (2, 4, 7, 24, 35):
                    v, z = variance_ratio(lr, q)
                    rows.append({"group": g, "asset": n, "base": "H", "q": q, "vr": v, "z": z})
        pd.DataFrame(rows).to_csv(self.r / "vr.csv", index=False)
        log.info("vr done")

    # ------------------------------------------------------------------ #
    @staticmethod
    def _complete(b: pd.DataFrame, tf: str, crypto: bool) -> pd.DataFrame:
        """去掉最后一根还没走完的 K 线（比如月初的当月月线）。"""
        if b.empty or tf not in ("M", "W"):
            return b
        last = b.index[-1]
        end = last + pd.offsets.MonthEnd(0) if tf == "M" else pd.offsets.Week(weekday=4).rollforward(last)
        nxt = last + (pd.Timedelta(days=1) if crypto else pd.offsets.BDay(1))
        return b.iloc[:-1] if nxt <= end else b

    def per_asset(self):
        rows, sl, bh = [], {}, {}
        for tf in ["M", "W", "D"]:
            for g, n, s in self.items():
                b = self.bars(n, tf)
                t0 = self.ready_from(b) if not b.empty else None
                base = {"group": g, "bucket": s.get("bucket", g), "asset": n, "tf": tf,
                        "daily_from": str(self.daily(n).index.min().date()) if len(self.daily(n)) else None}
                if t0 is None:
                    rows.append({**base, "note": "历史不够（需要 80 根 K 线 + 一个确认的波段）"})
                    continue
                tr, ret = simulate(b, StratParams(entry="any", exit="struct", cost_bps=s.get("cost_bps", 5.0)), n, tf)
                ret = ret[ret.index >= t0]
                tr = [t for t in tr if t["entry_t"] >= t0]
                sm = bar_to_monthly(ret)
                bm = bar_to_monthly(b["close"][b.index >= t0].pct_change().fillna(0))
                sl.setdefault((tf, base["bucket"]), {})[n] = sm
                bh.setdefault((tf, base["bucket"]), {})[n] = bm
                ms, mb = metrics(sm), metrics(bm)
                last = tr[-1] if tr else None
                rows.append({**base, "ready_from": str(t0.date()), **trade_stats(pd.DataFrame(tr)),
                             **{k: v for k, v in ms.items()},
                             **{"bh_" + k: v for k, v in mb.items() if k != "months"},
                             "exposure": round(sum(x["bars"] for x in tr) / max(1, len(ret)), 3),
                             "in_position_now": bool(last and last["exit_why"] == "open_end"),
                             "last_entry": str(last["entry_t"].date()) if last else None,
                             "last_entry_px": round(last["entry_px"], 4) if last else None,
                             "last_level": last["level"] if last else None,
                             "last_exit": str(last["exit_t"].date()) if last else None,
                             "last_exit_why": last["exit_why"] if last else None,
                             "last_ret": round(last["ret"], 4) if last else None})
            log.info("per_asset %s", tf)
        pd.DataFrame(rows).to_csv(self.r / "per_asset.csv", index=False)
        prow = []
        for tf in ["M", "W", "D"]:
            buckets = sorted({k[1] for k in sl if k[0] == tf})
            for bk in buckets + ["ALL"]:
                keys = [k for k in sl if k[0] == tf and (bk == "ALL" or k[1] == bk)]
                s_ = {a: v for k in keys for a, v in sl[k].items()}
                b_ = {a: v for k in keys for a, v in bh[k].items()}
                if not s_:
                    continue
                prow.append({"tf": tf, "bucket": bk, "assets": len(s_), "names": ",".join(s_),
                             **metrics(portfolio_monthly(s_)),
                             **{"bh_" + k: v for k, v in metrics(portfolio_monthly(b_)).items() if k != "months"}})
        pd.DataFrame(prow).to_csv(self.r / "bucket_portfolio.csv", index=False)

    def snapshot(self):
        rows = []
        sp = StratParams()
        for g, n, s in self.items():
            dly = self.daily(n)
            if dly.empty:
                rows.append({"asset": n, "note": "没有数据"})
                continue
            px = float(dly["close"].iloc[-1])
            for tf in ["M", "W"]:
                b = self._complete(self.bars(n, tf), tf, s.get("ann_days") == 365)
                base = {"bucket": s.get("bucket", g), "asset": n, "tf": tf, "ccy": s.get("ccy", "USD"),
                        "price": round(px, 4), "price_date": str(dly.index[-1].date()), "bars": len(b)}
                if b.empty or np.isnan(b["H"].iloc[-1]):
                    rows.append({**base, "note": "历史不够，还画不出波段"})
                    continue
                r = b.iloc[-1]
                H, L = r["H"], r["L"]
                m3 = b["ma80"].iloc[-4] if len(b) >= 4 else np.nan
                lv = {f"f{k}": fib_level(H, L, k) for k in (0.236, 0.382, 0.5, 0.618, 0.786)}
                sup = {**{k: v for k, v in lv.items() if k not in ("f0.236", "f0.786")},
                       "ma50": r["ma50"], "ma80": r["ma80"]}
                below = sorted([(v, k) for k, v in sup.items() if not np.isnan(v) and lv["f0.236"] < v < px],
                               reverse=True)
                rows.append({**base, "bar_date": str(b.index[-1].date()),
                             "healthy": healthy(r, sp, m3),
                             "above_f236": bool(px > lv["f0.236"]),
                             "ma80_rising": bool(not np.isnan(r["ma80"]) and not np.isnan(m3) and r["ma80"] >= m3),
                             "H": round(H, 4), "H_date": str(b.index[int(r["H_idx"])].date()),
                             "L": round(L, 4), "L_date": str(b.index[int(r["L_idx"])].date()),
                             "retrace_from_H": round(px / H - 1, 4),
                             "fib_pos": round((px - L) / (H - L), 3),
                             **{k: round(v, 4) for k, v in lv.items()},
                             "ma20": r["ma20"], "ma50": r["ma50"], "ma80": r["ma80"],
                             "K": round(r["K"], 1), "D": round(r["D"], 1),
                             "macd_hist": r["macd"] - r["macd_sig"], "macd": r["macd"],
                             "next_support": below[0][1] if below else None,
                             "next_support_px": round(below[0][0], 4) if below else None,
                             "pct_to_support": round(below[0][0] / px - 1, 4) if below else None,
                             "pct_to_stop": round(lv["f0.236"] / px - 1, 4),
                             "supports_below": ";".join(f"{k}={v:.2f}" for v, k in below)})
        pd.DataFrame(rows).to_csv(self.r / "snapshot.csv", index=False)
        log.info("snapshot done")

    def study(self, only=None):
        steps = only or ["vr", "scan", "fib_null", "rand_null", "mtf", "csp", "per_asset", "snapshot"]
        for st in steps:
            t0 = time.time()
            getattr(self, st)()
            log.info("== %s 完成 (%.0fs)", st, time.time() - t0)
        (self.r / "run_info.json").write_text(json.dumps({
            "generated": pd.Timestamp.now(tz="UTC").isoformat(), "steps": steps,
            "groups": {g: list(a) for g, a in self.u.items()}}, ensure_ascii=False, indent=1))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["all", "download", "study"])
    ap.add_argument("--universe", default="universe.json")
    ap.add_argument("--data-dir", default="data_trend")
    ap.add_argument("--results-dir", default="results_trend")
    ap.add_argument("--groups"); ap.add_argument("--assets"); ap.add_argument("--only")
    ap.add_argument("--no-hourly", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--push", action="store_true")
    a = ap.parse_args(argv)
    Path(a.data_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path(a.data_dir) / "trend.log")])
    u = json.loads(Path(a.universe).read_text())
    ts = TrendStudy(u, a.data_dir, a.results_dir, use_hourly=not a.no_hourly,
                    groups=a.groups.split(",") if a.groups else None,
                    assets=a.assets.split(",") if a.assets else None)
    if a.step in ("all", "download"):
        ts.download(a.force)
    if a.step in ("all", "study"):
        ts.study(a.only.split(",") if a.only else None)
    print(f"\n完成。结果在 {Path(a.results_dir).resolve()}")
    if a.push:
        from pipeline import git_push
        git_push(Path(a.results_dir), f"trend study {a.step}")


if __name__ == "__main__":
    main()

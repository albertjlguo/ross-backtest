#!/usr/bin/env python3
"""
数据下载 + 回测流水线（在 Replit 上运行）。

    python pipeline.py smoke              # 先跑最近约 3 周的小样本，确认 key、接口、数据都正常
    python pipeline.py all                # 全量：默认 2024-10-01 至昨天
    python pipeline.py all --start 2023-01-01 --push   # 跑完把 results/ 提交并推送回 GitHub

也可以单独跑某一步（每一步的产出都落盘，重跑会自动跳过已完成的部分）：
    universe → daily → candidates → minute → news → sec → dq → backtest

需要的 Replit Secrets：
    APCA_API_KEY_ID, APCA_API_SECRET_KEY    Alpaca（免费账户即可）
    SEC_USER_AGENT                          例如 "Your Name your@email.com"（SEC 要求）
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from rossbt import PRESETS, load_bars, load_daily, load_news, run_backtest
from rossbt.backtest import BacktestResult
from rossbt.data import ET, candidate_days, prepare_daily_split_aware
from rossbt.report import save_results

log = logging.getLogger("pipeline")

EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "ARCA", "BATS"}

_BASE = {k: v.with_(missing_float="include") for k, v in PRESETS.items()}
RUNS = {
    "ross": _BASE["ross"],
    "summary": _BASE["summary"],
    "aggressive": _BASE["aggressive"],
    # 稳健性检验
    "ross_worst_case": _BASE["ross"].with_(intrabar_path="worst_case"),
    "ross_slip30": _BASE["ross"].with_(slippage_bps=30.0),
    "ross_no_vol_div": _BASE["ross"].with_(sig_volume_divergence=False),
    "ross_top3": _BASE["ross"].with_(top_n_gainers=3),
    "ross_pullback2": _BASE["ross"].with_(max_pullback_number=2),
    "ross_no_catalyst": _BASE["ross"].with_(require_catalyst=False),
}


def keep_symbol(sym: str) -> bool:
    """普通股：去掉优先股/类别股（含 . / -）和 5 位代码的权证、单位、权利（W/U/R 结尾）。"""
    if not sym.isalpha():
        return False
    return not (len(sym) == 5 and sym[-1] in "WUR")


def et_iso(date, hhmm: str) -> str:
    return pd.Timestamp(f"{date} {hhmm}", tz=ET).isoformat()


class Pipeline:
    def __init__(self, data_dir: str | Path, results_dir: str | Path, start: str, end: str,
                 alpaca=None, sec=None, cand_min_pct: float = 0.05, cand_min_rvol: float = 2.0,
                 dq_samples: int = 20, batch_symbols: int = 100, minute_chunk: int = 50):
        self.d = Path(data_dir)
        self.r = Path(results_dir)
        self.d.mkdir(parents=True, exist_ok=True)
        self.r.mkdir(parents=True, exist_ok=True)
        self.start, self.end = start, end
        self._alp, self._sec = alpaca, sec
        self.cand_min_pct, self.cand_min_rvol = cand_min_pct, cand_min_rvol
        self.dq_samples = dq_samples
        self.batch_symbols = batch_symbols
        self.minute_chunk = minute_chunk

    # 惰性创建客户端（测试时可注入假的）
    @property
    def alp(self):
        if self._alp is None:
            from rossbt.alpaca import AlpacaClient
            self._alp = AlpacaClient()
        return self._alp

    @property
    def sec(self):
        if self._sec is None:
            from rossbt.sec import SecClient
            self._sec = SecClient(cache_dir=self.d / "sec_cache")
        return self._sec

    # ------------------------------------------------------------------ #
    def step_universe(self, force=False) -> pd.DataFrame:
        p = self.d / "universe.parquet"
        if p.exists() and not force:
            return pd.read_parquet(p)
        parts = [self.alp.assets("active"), self.alp.assets("inactive")]
        u = pd.concat(parts, ignore_index=True).drop_duplicates("symbol")
        u = u[u["exchange"].isin(EXCHANGES) & u["symbol"].map(keep_symbol)]
        u.to_parquet(p, index=False)
        log.info("股票池: %d 只（其中已停止交易 %d）", len(u), int((u["status"] == "inactive").sum()))
        return u

    def step_daily(self, force=False):
        u = self.step_universe()
        syms = sorted(u["symbol"])
        dstart = (pd.Timestamp(self.start) - pd.Timedelta(days=75)).strftime("%Y-%m-%d")
        out = self.d / "daily"
        out.mkdir(exist_ok=True)
        n = (len(syms) + self.batch_symbols - 1) // self.batch_symbols
        for i in range(n):
            batch = syms[i * self.batch_symbols:(i + 1) * self.batch_symbols]
            for kind, adj in (("raw", "raw"), ("adj", "split")):
                p = out / f"{kind}_{i:04d}.parquet"
                if p.exists() and not force:
                    continue
                df = self.alp.bars(batch, "1Day", dstart, self.end, adjustment=adj)
                df.to_parquet(p, index=False)
            if i % 10 == 0:
                log.info("日线 %d/%d 批", i + 1, n)

    def load_daily_bars(self, kind: str, symbols=None) -> pd.DataFrame:
        cols = ["symbol", "ts", "high", "low", "close", "volume"] if kind == "raw" \
            else ["symbol", "ts", "close"]
        files = sorted((self.d / "daily").glob(f"{kind}_*.parquet"))
        parts = []
        for f in files:
            x = pd.read_parquet(f, columns=cols)
            if symbols is not None:
                x = x[x["symbol"].isin(symbols)]
            parts.append(x)
        df = pd.concat(parts, ignore_index=True)
        df["date"] = df["ts"].dt.tz_convert(ET).dt.date
        return df

    def trading_days(self) -> list:
        """区间内所有交易日（含没有交易信号的日子，算日度夏普要用）。"""
        days = set()
        for f in sorted((self.d / "daily").glob("raw_*.parquet")):
            ts = pd.read_parquet(f, columns=["ts"])["ts"]
            days |= set(ts.dt.tz_convert(ET).dt.date.unique())
        s, e = pd.Timestamp(self.start).date(), pd.Timestamp(self.end).date()
        return sorted(d for d in days if s <= d <= e)

    def step_candidates(self, force=False) -> pd.DataFrame:
        p = self.d / "candidates.parquet"
        if p.exists() and not force:
            return pd.read_parquet(p)
        raw, adj = self.load_daily_bars("raw"), self.load_daily_bars("adj")
        table = prepare_daily_split_aware(raw, adj)
        cands = candidate_days(raw, table, min_pct=self.cand_min_pct,
                               min_rvol=self.cand_min_rvol)
        s, e = pd.Timestamp(self.start).date(), pd.Timestamp(self.end).date()
        cands = cands[(cands["date"] >= s) & (cands["date"] <= e)]
        cands.to_parquet(p, index=False)
        log.info("候选: %d 个股票×交易日，%d 个交易日", len(cands), cands["date"].nunique())
        return cands

    def _per_date(self, sub: str, fetch, force=False):
        cands = self.step_candidates()
        out = self.d / sub
        out.mkdir(exist_ok=True)
        dates = sorted(cands["date"].unique())
        for k, (date, g) in enumerate(cands.groupby("date", sort=True)):
            p = out / f"{date}.parquet"
            if p.exists() and not force:
                continue
            syms = sorted(g["symbol"].unique())
            parts = [fetch(date, syms[i:i + self.minute_chunk])
                     for i in range(0, len(syms), self.minute_chunk)]
            pd.concat(parts, ignore_index=True).to_parquet(p, index=False)
            if k % 20 == 0:
                log.info("%s %s (%d/%d)", sub, date, k + 1, len(dates))

    def step_minute(self, force=False):
        self._per_date("minute", lambda d, s: self.alp.bars(
            s, "1Min", et_iso(d, "04:00"), et_iso(d, "12:00"), adjustment="raw"), force)

    def step_news(self, force=False):
        def fetch(d, s):
            prev = pd.Timestamp(d) - pd.Timedelta(days=1)
            return self.alp.news(s, et_iso(prev.date(), "04:00"), et_iso(d, "12:00"))
        self._per_date("news", fetch, force)

    def step_sec(self, force=False):
        p = self.d / "float.parquet"
        if p.exists() and not force:
            return
        from rossbt.sec import shares_table
        cands = self.step_candidates()
        tbl, stats = shares_table(self.sec, cands["symbol"].unique())
        tbl.to_parquet(p, index=False)
        (self.r / "sec_coverage.json").write_text(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------ #
    def step_dq(self, force=False):
        """数据质量：日线是否含盘前盘后、预筛会不会漏掉只在盘前拉升的票、已退市覆盖、股本覆盖。"""
        p = self.r / "data_quality.json"
        if p.exists() and not force:
            return
        cands = self.step_candidates()
        raw = self.load_daily_bars("raw")
        u = self.step_universe()
        rep: dict = {"candidates": int(len(cands)), "candidate_dates": int(cands["date"].nunique())}

        inactive = set(u.loc[u["status"] == "inactive", "symbol"])
        with_bars = set(raw["symbol"].unique())
        rep["inactive_symbols"] = len(inactive)
        rep["inactive_with_daily_bars"] = len(inactive & with_bars)
        rep["inactive_in_candidates"] = int(cands["symbol"].isin(inactive).sum())

        sample = cands.sample(min(self.dq_samples, len(cands)), random_state=0)
        rows = []
        for r in sample.itertuples():
            m = self.alp.bars([r.symbol], "1Min", et_iso(r.date, "04:00"), et_iso(r.date, "20:00"))
            if m.empty:
                continue
            m["minute"] = m["ts"].dt.tz_convert(ET).dt.hour * 60 + m["ts"].dt.tz_convert(ET).dt.minute
            reg = m[(m["minute"] >= 570) & (m["minute"] < 960)]
            pre = m[m["minute"] < 570]
            dd = raw[(raw["symbol"] == r.symbol) & (raw["date"] == r.date)]
            if dd.empty:
                continue
            dv, dh = float(dd["volume"].iloc[0]), float(dd["high"].iloc[0])
            rows.append({
                "symbol": r.symbol, "date": str(r.date),
                "daily_vol_over_regular_minutes": dv / max(1.0, reg["volume"].sum()),
                "daily_vol_over_all_minutes": dv / max(1.0, m["volume"].sum()),
                "premarket_vol_share": pre["volume"].sum() / max(1.0, m["volume"].sum()),
                "premarket_high_above_daily_high": bool(len(pre) and pre["high"].max() > dh * 1.001),
                "premarket_minutes_with_bars": int(len(pre)),
            })
        df = pd.DataFrame(rows)
        if len(df):
            rep["sample_size"] = len(df)
            rep["median_daily_vol_over_regular_minutes"] = round(df["daily_vol_over_regular_minutes"].median(), 3)
            rep["median_daily_vol_over_all_minutes"] = round(df["daily_vol_over_all_minutes"].median(), 3)
            rep["median_premarket_vol_share"] = round(df["premarket_vol_share"].median(), 3)
            rep["share_premarket_high_above_daily_high"] = round(df["premarket_high_above_daily_high"].mean(), 3)
            rep["median_premarket_minutes_with_bars"] = float(df["premarket_minutes_with_bars"].median())
            rep["interpretation"] = (
                "daily_vol_over_all≈1 → 日线含盘前盘后；daily_vol_over_regular≈1 → 日线只含常规时段，"
                "相对量会偏高，需要调高 min_rvol 或改用分钟线自算 ADV。"
                "premarket_high_above_daily_high 比例高 → 日线最高价不含盘前，预筛可能漏票。")
            df.to_csv(self.r / "data_quality_sample.csv", index=False)
        p.write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str))
        log.info("数据质量: %s", rep)

    # ------------------------------------------------------------------ #
    def step_backtest(self, runs: dict | None = None):
        runs = runs or RUNS
        csyms = set(self.step_candidates()["symbol"])
        raw = self.load_daily_bars("raw", csyms)
        adj = self.load_daily_bars("adj", csyms)
        fpath = self.d / "float.parquet"
        ftbl = pd.read_parquet(fpath) if fpath.exists() else None
        daily = load_daily(prepare_daily_split_aware(raw, adj, ftbl))
        by_date = {k: g for k, g in daily.groupby("date")}
        all_days = self.trading_days()

        news_files = sorted((self.d / "news").glob("*.parquet")) if (self.d / "news").exists() else []
        news_by_date = {}
        for f in news_files:
            n = pd.read_parquet(f)
            news_by_date[f.stem] = load_news(n) if len(n) else load_news(
                pd.DataFrame({"symbol": pd.Series(dtype=str), "ts": pd.Series(dtype="datetime64[ns, UTC]")}))

        acc = {name: {"trades": [], "all": [], "dropped": [], "counters": Counter(), "notes": None}
               for name in runs}
        files = sorted((self.d / "minute").glob("*.parquet"))
        for k, f in enumerate(files):
            raw_bars = pd.read_parquet(f)
            if raw_bars.empty:
                continue
            bars = load_bars(raw_bars)
            date = bars["date"].iloc[0]
            dsub = by_date.get(date)
            if dsub is None:
                continue
            news = news_by_date.get(f.stem) if news_files else None
            for name, cfg in runs.items():
                res = run_backtest(bars, dsub, cfg, news=news)
                a = acc[name]
                a["trades"].append(res.trades); a["all"].append(res.trades_all)
                a["dropped"].append(res.dropped); a["counters"].update(res.counters)
                a["notes"] = a["notes"] or res.notes
            if k % 50 == 0:
                log.info("回测 %s (%d/%d)", date, k + 1, len(files))

        overview = []
        for name, cfg in runs.items():
            a = acc[name]
            cat = lambda xs: (pd.concat([x for x in xs if len(x)], ignore_index=True)
                              if any(len(x) for x in xs) else pd.DataFrame())
            res = BacktestResult(cat(a["trades"]), cat(a["all"]), cat(a["dropped"]),
                                 a["counters"], all_days, cfg, a["notes"] or [])
            m = save_results(res, self.r / name)
            overview.append({"run": name, **m})
            log.info("%s: %s", name, {k: m.get(k) for k in ("trades", "win_rate", "avg_r",
                                                            "profit_factor", "total_pnl")})
        ov = pd.DataFrame(overview)
        ov.to_csv(self.r / "overview.csv", index=False)
        (self.r / "run_info.json").write_text(json.dumps({
            "start": self.start, "end": self.end, "generated": dt.datetime.utcnow().isoformat() + "Z",
            "minute_files": len(files), "news_files": len(news_files),
            "float_table": fpath.exists(), "prescreen": {"min_pct": self.cand_min_pct,
                                                          "min_rvol": self.cand_min_rvol}},
            indent=2))
        return ov

    def run_all(self, force=False):
        self.step_universe(force); self.step_daily(force); self.step_candidates(force)
        self.step_minute(force); self.step_news(force)
        try:
            self.step_sec(force)
        except Exception as e:  # SEC 失败不影响主流程
            log.warning("SEC 股本步骤失败，跳过（float 视为缺失）: %s", e)
        self.step_dq(force)
        return self.step_backtest()


def git_push(results_dir: Path, msg: str):
    cmds = [["git", "add", "-f", str(results_dir)], ["git", "commit", "-m", msg], ["git", "push"]]
    for c in cmds:
        r = subprocess.run(c, capture_output=True, text=True)
        print("$", " ".join(c), "\n", r.stdout[-2000:], r.stderr[-2000:])
        if r.returncode != 0 and c[1] != "commit":
            print("推送失败：请在 Replit 的 Git 面板里确认已连接 GitHub 账户")
            return


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("step", choices=["smoke", "all", "universe", "daily", "candidates", "minute",
                                     "news", "sec", "dq", "backtest"])
    yesterday = (pd.Timestamp.now(tz=ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default=yesterday)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--cand-min-pct", type=float, default=0.05)
    ap.add_argument("--cand-min-rvol", type=float, default=2.0)
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--push", action="store_true", help="完成后 git add/commit/push results/")
    a = ap.parse_args(argv)

    if a.step == "smoke":
        a.start = (pd.Timestamp(a.end) - pd.Timedelta(days=21)).strftime("%Y-%m-%d")
        a.data_dir, a.results_dir = "data_smoke", "results/smoke"

    Path(a.data_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(Path(a.data_dir) / "pipeline.log")])
    p = Pipeline(a.data_dir, a.results_dir, a.start, a.end,
                 cand_min_pct=a.cand_min_pct, cand_min_rvol=a.cand_min_rvol,
                 dq_samples=5 if a.step == "smoke" else 20)
    log.info("区间 %s → %s，数据目录 %s", a.start, a.end, a.data_dir)

    if a.step == "smoke":
        p.run_all(a.force)
    elif a.step == "all":
        p.run_all(a.force)
    elif a.step == "backtest":
        p.step_backtest()
    else:
        getattr(p, f"step_{a.step}")(a.force)

    print(f"\n完成。结果在 {Path(a.results_dir).resolve()}")
    if a.push:
        git_push(Path(a.results_dir), f"results {a.step} {a.start}..{a.end}")


if __name__ == "__main__":
    main()

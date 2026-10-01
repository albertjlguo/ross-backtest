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
from rossbt.ticks import resolve as resolve_ticks, to_resolver

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
    "ross_pb_window": _BASE["ross"].with_(count_pullbacks_from="window"),
    "ross_no_catalyst": _BASE["ross"].with_(require_catalyst=False),
    # 诊断：离场信号是帮忙还是添乱
    "ross_no_first_red": _BASE["ross"].with_(exit_on_first_red_before_t1=False),
    # 样本内发现"07:00–08:00 入场接近打平"，固定成一组配置，用更早年份做样本外检验
    "ross_7to8": _BASE["ross"].with_(entry_end="08:00"),
    "ross_bracket_only": _BASE["ross"].with_(
        exit_on_first_red_before_t1=False, sig_stall=False, sig_bid_pressure=False,
        sig_topping_tail=False, sig_rejections=False, sig_volume_divergence=False),
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
                 alpaca=None, sec=None, prescreen: str = "volume",
                 dq_samples: int = 20, batch_symbols: int = 100, minute_chunk: int = 50):
        self.d = Path(data_dir)
        self.r = Path(results_dir)
        self.d.mkdir(parents=True, exist_ok=True)
        self.r.mkdir(parents=True, exist_ok=True)
        self.start, self.end = start, end
        self._alp, self._sec = alpaca, sec
        cfgs = list(RUNS.values())
        # 预筛取所有配置里最宽松的阈值，保证任何一组配置能选中的票都被下载
        self.prescreen = {"mode": prescreen,
                          "min_pct": min(c.min_pct_change for c in cfgs),
                          "min_rvol": min(c.min_rvol for c in cfgs),
                          "min_price": min(c.min_price for c in cfgs),
                          "max_price": max(c.max_price for c in cfgs)}
        self.signature = {"start": start, "end": end, **self.prescreen}
        # 日线按区间分目录：改起止日期会重新下载日线，分钟线/新闻按日期文件复用
        self.daily_start = (pd.Timestamp(start) - pd.Timedelta(days=75)).strftime("%Y-%m-%d")
        self.daily_dir = self.d / "daily" / f"{self.daily_start}_{end}"
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
        out = self.daily_dir
        out.mkdir(parents=True, exist_ok=True)
        dstart = self.daily_start
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
        files = sorted(self.daily_dir.glob(f"{kind}_*.parquet"))
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
        for f in sorted(self.daily_dir.glob("raw_*.parquet")):
            ts = pd.read_parquet(f, columns=["ts"])["ts"]
            days |= set(ts.dt.tz_convert(ET).dt.date.unique())
        s, e = pd.Timestamp(self.start).date(), pd.Timestamp(self.end).date()
        return sorted(d for d in days if s <= d <= e)

    def _meta_ok(self, path: Path) -> bool:
        return path.exists() and json.loads(path.read_text()) == self.signature

    def step_candidates(self, force=False) -> pd.DataFrame:
        p, meta = self.d / "candidates.parquet", self.d / "candidates.meta.json"
        if p.exists() and self._meta_ok(meta) and not force:
            return pd.read_parquet(p)
        raw, adj = self.load_daily_bars("raw"), self.load_daily_bars("adj")
        table = prepare_daily_split_aware(raw, adj)
        ps = self.prescreen
        cands = candidate_days(raw, table, min_pct=ps["min_pct"], min_rvol=ps["min_rvol"],
                               min_price=ps["min_price"], max_price=ps["max_price"],
                               mode=ps["mode"])
        s, e = pd.Timestamp(self.start).date(), pd.Timestamp(self.end).date()
        cands = cands[(cands["date"] >= s) & (cands["date"] <= e)].reset_index(drop=True)
        cands.to_parquet(p, index=False)
        meta.write_text(json.dumps(self.signature))
        log.info("候选: %d 个股票×交易日，%d 个交易日（预筛 %s）", len(cands),
                 cands["date"].nunique(), ps)
        return cands

    def _per_date(self, sub: str, fetch, force=False):
        """
        按日期下载，增量：每个日期旁边存一份 manifest，记录已经请求过的股票。
        候选名单变了（例如改了预筛），只补下新增的股票。
        """
        cands = self.step_candidates()
        out = self.d / sub
        out.mkdir(exist_ok=True)
        dates = sorted(cands["date"].unique())
        for k, (date, g) in enumerate(cands.groupby("date", sort=True)):
            p, man = out / f"{date}.parquet", out / f"{date}.json"
            want = set(g["symbol"].unique())
            have = pd.read_parquet(p) if (p.exists() and not force) else None
            if have is not None:
                done = set(json.loads(man.read_text())) if man.exists() else set(have["symbol"])
            else:
                done = set()
            need = sorted(want - done)
            if not need:
                continue
            parts = [fetch(date, need[i:i + self.minute_chunk])
                     for i in range(0, len(need), self.minute_chunk)]
            new = pd.concat(parts, ignore_index=True)
            if have is not None and len(have):
                new = pd.concat([have, new], ignore_index=True) if len(new) else have
            new.to_parquet(p, index=False)
            man.write_text(json.dumps(sorted(done | set(need))))
            if k % 20 == 0:
                log.info("%s %s (%d/%d) +%d 只", sub, date, k + 1, len(dates), len(need))

    def step_minute(self, force=False):
        self._per_date("minute", lambda d, s: self.alp.bars(
            s, "1Min", et_iso(d, "04:00"), et_iso(d, "12:00"), adjustment="raw"), force)

    def step_news(self, force=False):
        def fetch(d, s):
            prev = pd.Timestamp(d) - pd.Timedelta(days=1)
            return self.alp.news(s, et_iso(prev.date(), "04:00"), et_iso(d, "12:00"))
        self._per_date("news", fetch, force)

    def step_sec(self, force=False):
        p, meta = self.d / "float.parquet", self.d / "float.meta.json"
        syms = set(self.step_candidates()["symbol"].unique())
        if p.exists() and meta.exists() and not force and syms <= set(json.loads(meta.read_text())):
            return
        from rossbt.sec import shares_table          # 公司数据有本地缓存，重算很快
        tbl, stats = shares_table(self.sec, syms)
        tbl.to_parquet(p, index=False)
        meta.write_text(json.dumps(sorted(syms)))
        (self.r / "sec_coverage.json").write_text(json.dumps(stats, indent=2))

    # ------------------------------------------------------------------ #
    def step_dq(self, force=False):
        """数据质量：日线是否含盘前盘后、预筛会不会漏掉只在盘前拉升的票、已退市覆盖、股本覆盖。"""
        p = self.r / "data_quality.json"
        if p.exists() and not force and json.loads(p.read_text()).get("signature") == self.signature:
            return
        cands = self.step_candidates()
        raw = self.load_daily_bars("raw")
        u = self.step_universe()
        rep: dict = {"signature": self.signature, "candidates": int(len(cands)),
                     "candidate_dates": int(cands["date"].nunique())}

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

        rpath = self.d / "tick_resolutions.parquet"
        resolver = to_resolver(pd.read_parquet(rpath)) if rpath.exists() else None
        log.info("逐笔结论: %d 条", len(resolver or {}))
        amb_all: list[dict] = []
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
            cache: dict = {}
            for name, cfg in runs.items():
                res = run_backtest(bars, dsub, cfg, news=news, resolver=resolver,
                                   collect_ambiguous=True, feat_cache=cache)
                amb_all += res.ambiguous
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
        amb = pd.DataFrame(amb_all, columns=["symbol", "minute_utc", "up", "down", "kind"])
        amb = amb.drop_duplicates(["symbol", "minute_utc", "up", "down"])
        amb.to_parquet(self.d / "ambiguous_events.parquet", index=False)
        n_res = 0 if resolver is None else sum(
            (r.symbol, r.minute_utc, r.up, r.down) in resolver for r in amb.itertuples())
        (self.r / "run_info.json").write_text(json.dumps({
            "start": self.start, "end": self.end, "generated": dt.datetime.now(dt.timezone.utc).isoformat(),
            "minute_files": len(files), "news_files": len(news_files),
            "float_table": fpath.exists(), "prescreen": self.prescreen,
            "ambiguous_events": int(len(amb)), "ambiguous_resolved_by_ticks": int(n_res)},
            indent=2))
        return ov

    def step_ticks(self, force=False) -> int:
        """拉歧义分钟的逐笔，返回本次新增的结论条数（含"逐笔也判断不了"的）。"""
        ep = self.d / "ambiguous_events.parquet"
        if not ep.exists():
            return 0
        ev = pd.read_parquet(ep)
        rpath = self.d / "tick_resolutions.parquet"
        old = pd.read_parquet(rpath) if (rpath.exists() and not force) else None
        if old is not None:
            done = set(zip(old["symbol"], old["minute_utc"], old["up"].round(4), old["down"].round(4)))
            ev = ev[[(s, m, round(u, 4), round(d, 4)) not in done
                     for s, m, u, d in zip(ev["symbol"], ev["minute_utc"], ev["up"], ev["down"])]]
        if ev.empty:
            log.info("逐笔: 没有新的歧义事件")
            return 0
        keys = ev[["symbol", "minute_utc"]].drop_duplicates()
        log.info("逐笔: %d 个事件，%d 个分钟待下载", len(ev), len(keys))
        tbk = {}
        for k, r in enumerate(keys.itertuples(index=False)):
            t0 = pd.Timestamp(r.minute_utc)
            tbk[(r.symbol, r.minute_utc)] = self.alp.trades(
                [r.symbol], t0.isoformat(), (t0 + pd.Timedelta(seconds=60)).isoformat())
            if k % 200 == 0:
                log.info("逐笔 %d/%d", k + 1, len(keys))
        res = resolve_ticks(ev, tbk)
        allres = pd.concat([old, res], ignore_index=True) if old is not None else res
        allres.to_parquet(rpath, index=False)
        log.info("逐笔结论: 新增 %d 条（其中判断不了 %d 条）", len(res), int(res["first"].isna().sum()))
        return len(res)

    def step_resolve(self, max_passes: int = 3):
        """回测 → 拉歧义分钟逐笔 → 再回测，直到没有新的歧义事件。"""
        ov = None
        for it in range(max_passes):
            log.info("回测第 %d 轮", it + 1)
            ov = self.step_backtest()
            if self.step_ticks() == 0:
                break
        else:
            ov = self.step_backtest()
        return ov

    def run_all(self, force=False):
        self.step_universe(force); self.step_daily(force); self.step_candidates(force)
        self.step_minute(force); self.step_news(force)
        try:
            self.step_sec(force)
        except Exception as e:  # SEC 失败不影响主流程
            log.warning("SEC 股本步骤失败，跳过（float 视为缺失）: %s", e)
        self.step_dq(force)
        return self.step_resolve()


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
                                     "news", "sec", "dq", "backtest", "ticks", "resolve"])
    yesterday = (pd.Timestamp.now(tz=ET) - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    ap.add_argument("--start", default="2024-10-01")
    ap.add_argument("--end", default=yesterday)
    ap.add_argument("--data-dir", default="data")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--prescreen", choices=["volume", "price"], default="volume")
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
                 prescreen=a.prescreen,
                 dq_samples=5 if a.step == "smoke" else 20)
    log.info("区间 %s → %s，数据目录 %s", a.start, a.end, a.data_dir)

    if a.step == "smoke":
        p.run_all(a.force)
    elif a.step == "all":
        p.run_all(a.force)
    elif a.step == "backtest":
        p.step_backtest()
    elif a.step == "resolve":
        p.step_resolve()
    else:
        getattr(p, f"step_{a.step}")(a.force)

    print(f"\n完成。结果在 {Path(a.results_dir).resolve()}")
    if a.push:
        git_push(Path(a.results_dir), f"results {a.step} {a.start}..{a.end}")


if __name__ == "__main__":
    main()

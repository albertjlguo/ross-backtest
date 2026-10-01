"""
回测入口：逐日、逐股票跑状态机，再套组合层规则（同时持仓数、单日最大亏损）。
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
import warnings

import numpy as np
import pandas as pd

from .config import Config
from .features import symbol_day_features, apply_gainer_rank, hhmm
from .engine import simulate_symbol_day


@dataclass
class BacktestResult:
    trades: pd.DataFrame            # 套用组合规则后实际"成交"的交易
    trades_all: pd.DataFrame        # 单票独立模拟的全部信号交易
    dropped: pd.DataFrame           # 被组合规则拦下的
    counters: Counter
    days: list
    cfg: Config
    notes: list = field(default_factory=list)
    ambiguous: list = field(default_factory=list)   # 同根 K 线两价位都被触及的事件（供拉逐笔）


FEATURE_FIELDS = ("session_start", "vwap_anchor", "vol_avg_window", "participation_window",
                  "news_lookback_hours", "min_pct_change", "min_rvol", "min_price", "max_price",
                  "max_float", "require_catalyst", "missing_float")


def run_backtest(bars: pd.DataFrame, daily: pd.DataFrame, cfg: Config,
                 news: pd.DataFrame | None = None, progress: bool = False,
                 resolver: dict | None = None, collect_ambiguous: bool = False,
                 feat_cache: dict | None = None) -> BacktestResult:
    """
    resolver / collect_ambiguous：见 engine.simulate_symbol_day。
    feat_cache：同一批 bars 跑多组配置时传同一个 dict，特征相关参数相同的配置共用每日特征。
    """
    notes: list[str] = []
    counters: Counter = Counter()

    if not cfg.require_catalyst:
        cat_mode = "off"
    elif news is not None:
        cat_mode = "news"
    elif "has_catalyst" in daily.columns:
        cat_mode = "flag"
    else:
        cat_mode = "off"
        msg = "未提供新闻数据（news 表或日表 has_catalyst 列），催化剂条件已关闭"
        warnings.warn(msg)
        notes.append(msg)
    notes.append(f"催化剂判定方式: {cat_mode}")

    news_map: dict[str, np.ndarray] = {}
    if cat_mode == "news":
        for s, g in news.groupby("symbol"):
            news_map[s] = np.sort(g["ts"].astype("int64").to_numpy())

    didx = daily.drop_duplicates(["date", "symbol"]).set_index(["date", "symbol"])
    e_end = hhmm(cfg.entry_end)
    s0 = hhmm(cfg.session_start)

    trades: list[dict] = []
    amb: list[dict] | None = [] if collect_ambiguous else None
    days = sorted(bars["date"].unique())
    fkey = tuple(getattr(cfg, k) for k in FEATURE_FIELDS) + (cat_mode,)
    for k, (date, dbars) in enumerate(bars.groupby("date", sort=True)):
        if progress and k % 20 == 0:
            print(f"  {date}  ({k + 1}/{len(days)})")
        ck = (date, fkey)
        if feat_cache is not None and ck in feat_cache:
            base, miss = feat_cache[ck]
            counters["symbol_days_missing_daily"] += miss
            if base is None:
                continue
            day = apply_gainer_rank(base.copy(), cfg.top_n_gainers)
            _simulate_day(day, date, cfg, didx, e_end, s0, counters, trades, resolver, amb)
            continue
        miss0 = counters["symbol_days_missing_daily"]
        feats = []
        for sym, g in dbars.groupby("symbol", sort=False):
            key = (date, sym)
            if key not in didx.index:
                counters["symbol_days_missing_daily"] += 1
                continue
            row = didx.loc[key]
            if pd.isna(row["prev_close"]) or pd.isna(row["adv30"]):
                counters["symbol_days_missing_daily"] += 1
                continue
            f = symbol_day_features(
                g, float(row["prev_close"]), float(row["adv30"]), row.get("float_shares", np.nan),
                news_map.get(sym), row.get("has_catalyst") if cat_mode == "flag" else None,
                cfg, cat_mode)
            if len(f):
                feats.append(f)
        base = pd.concat(feats, ignore_index=True) if feats else None
        if feat_cache is not None:
            feat_cache[ck] = (base, counters["symbol_days_missing_daily"] - miss0)
        if base is None:
            continue
        day = apply_gainer_rank(base.copy() if feat_cache is not None else base,
                                cfg.top_n_gainers)
        _simulate_day(day, date, cfg, didx, e_end, s0, counters, trades, resolver, amb)

    tall = pd.DataFrame(trades)
    if len(tall):
        tall = tall.sort_values("entry_time").reset_index(drop=True)
    kept, dropped = apply_portfolio_rules(tall, cfg)
    return BacktestResult(kept, tall, dropped, counters, days, cfg, notes, amb or [])


def _simulate_day(day, date, cfg, didx, e_end, s0, counters, trades, resolver, amb):
    for sym, f in day.groupby("symbol", sort=False):
        counters["symbol_days"] += 1
        win = (f["minute"] >= s0) & (f["minute"] < e_end)
        if not (f["qual"] & win).any():
            continue
        counters["symbol_days_qualified"] += 1
        fs = didx.loc[(date, sym)].get("float_shares", np.nan)
        trades += simulate_symbol_day(sym, date, f.reset_index(drop=True), cfg, fs, counters,
                                      resolver=resolver, amb_log=amb)


def apply_portfolio_rules(trades: pd.DataFrame, cfg: Config):
    """
    按入场时间顺序过一遍：同时持仓达上限、或当日已实现亏损达上限，则这笔不做。
    近似：被拦下的交易不会让该股票后续的信号重新出现（单票模拟里它已"发生"）。
    """
    if trades is None or len(trades) == 0:
        return pd.DataFrame(), pd.DataFrame()
    keep_idx, drop_idx, drop_why = [], [], []
    for _, d in trades.groupby("date", sort=True):
        kept_rows = []
        for idx, r in d.sort_values("entry_time").iterrows():
            n_open = sum(1 for k in kept_rows if k["exit_time"] > r["entry_time"])
            realized = sum(k["pnl"] for k in kept_rows if k["exit_time"] <= r["entry_time"])
            if n_open >= cfg.max_concurrent_positions:
                drop_idx.append(idx); drop_why.append("max_concurrent")
            elif cfg.daily_max_loss is not None and realized <= -cfg.daily_max_loss:
                drop_idx.append(idx); drop_why.append("daily_max_loss")
            else:
                keep_idx.append(idx); kept_rows.append(r)
    kept = trades.loc[keep_idx].sort_values("entry_time").reset_index(drop=True)
    dropped = trades.loc[drop_idx].copy()
    dropped["drop_reason"] = drop_why
    return kept, dropped.reset_index(drop=True)

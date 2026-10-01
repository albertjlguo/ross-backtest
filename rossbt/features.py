"""
逐分钟特征：累计量、相对量、涨幅、VWAP、当日高点、五条件判定、涨幅榜排名。
所有量在第 i 根 K 线收盘时只用到 ≤i 的数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import Config


def hhmm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


def symbol_day_features(g: pd.DataFrame, prev_close: float, adv30: float,
                        float_shares: float, news_ts: np.ndarray | None,
                        has_catalyst_flag: bool | None, cfg: Config,
                        catalyst_mode: str) -> pd.DataFrame:
    """g: 单个 symbol 单日的分钟线（已排序，含 minute 列）。"""
    g = g[g["minute"] >= hhmm(cfg.session_start)].copy()
    if g.empty:
        return g
    v = g["volume"].to_numpy(dtype=float)
    h = g["high"].to_numpy(dtype=float)
    l = g["low"].to_numpy(dtype=float)
    c = g["close"].to_numpy(dtype=float)

    g["cumvol"] = np.cumsum(v)
    g["rvol"] = g["cumvol"] / adv30 if adv30 and adv30 > 0 else np.nan
    g["pct"] = c / prev_close - 1.0

    anchor = g["minute"].to_numpy() >= hhmm(cfg.vwap_anchor)
    tp = (h + l + c) / 3.0
    pv = np.where(anchor, tp * v, 0.0).cumsum()
    vv = np.where(anchor, v, 0.0).cumsum()
    with np.errstate(invalid="ignore", divide="ignore"):
        g["vwap"] = np.where(vv > 0, pv / vv, np.nan)

    g["hod"] = np.maximum.accumulate(h)
    # 前 N 根的平均量（不含当前根），用于"放量"判断
    g["vol_avg_prev"] = (
        g["volume"].rolling(cfg.vol_avg_window, min_periods=3).mean().shift(1)
    )
    # 最近 N 根平均量（含当前根，收盘可知），用于下一根的仓位流动性上限
    g["part_vol"] = g["volume"].rolling(cfg.participation_window, min_periods=1).mean()

    if "vol_at_bid" in g.columns and "vol_at_ask" in g.columns:
        b = g["vol_at_bid"].astype(float)
        a = g["vol_at_ask"].astype(float)
        tot = a + b
        g["bid_ratio"] = np.where(tot > 0, b / tot.where(tot > 0, 1), np.nan)
    else:
        g["bid_ratio"] = np.nan

    # 催化剂
    if catalyst_mode == "news":
        t_ns = g["ts"].astype("int64").to_numpy()
        if news_ts is None or len(news_ts) == 0:
            cat = np.zeros(len(g), dtype=bool)
        else:
            lb = int(cfg.news_lookback_hours * 3600 * 1e9)
            hi = np.searchsorted(news_ts, t_ns, side="right")       # 新闻 ≤ t
            lo = np.searchsorted(news_ts, t_ns - lb, side="left")   # 新闻 ≥ t - lookback
            cat = hi > lo
    elif catalyst_mode == "flag":
        cat = np.full(len(g), bool(has_catalyst_flag))
    else:  # off
        cat = np.ones(len(g), dtype=bool)
    g["catalyst"] = cat

    if float_shares is None or pd.isna(float_shares):
        float_ok = cfg.missing_float == "include"
    else:
        float_ok = float_shares <= cfg.max_float

    g["qual_base"] = (
        (g["pct"] >= cfg.min_pct_change)
        & (g["rvol"] >= cfg.min_rvol)
        & (g["close"] >= cfg.min_price)
        & (g["close"] <= cfg.max_price)
        & bool(float_ok)
        & g["catalyst"]
    )
    return g


def apply_gainer_rank(day_feats: pd.DataFrame, top_n: int | None) -> pd.DataFrame:
    """
    涨幅榜排名：每分钟在"满足其余四条件"的股票里按涨幅排序，只保留前 top_n。
    某只股票这一分钟没成交时沿用它上一分钟的状态（ffill），避免它"消失"让别人排名虚高。
    """
    if top_n is None:
        day_feats["qual"] = day_feats["qual_base"]
        return day_feats
    piv = day_feats.pivot_table(index="ts", columns="symbol", values="pct", aggfunc="last")
    okp = day_feats.pivot_table(index="ts", columns="symbol", values="qual_base",
                                aggfunc="last").astype(float)
    piv = piv.sort_index().ffill()
    okp = okp.reindex(piv.index).ffill().fillna(0).astype(bool)
    masked = piv.where(okp)
    rank = masked.rank(axis=1, ascending=False, method="first")
    r = rank.stack(future_stack=True).rename("gain_rank").reset_index()
    day_feats = day_feats.merge(r, on=["ts", "symbol"], how="left")
    day_feats["qual"] = day_feats["qual_base"] & (day_feats["gain_rank"] <= top_n)
    return day_feats

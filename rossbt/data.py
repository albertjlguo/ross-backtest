"""
数据加载与准备。

需要三张表（CSV 或 Parquet 均可）：

1) 分钟线 bars —— 必须含盘前（从 04:00 ET 起）
   symbol, ts, open, high, low, close, volume
   可选: vol_at_bid, vol_at_ask  （每分钟按买一价 / 卖一价成交的量，来自逐笔 + 报价聚合）
   ts: 带时区的时间戳，或不带时区（默认按美东解释，可用 naive_tz 改）

2) 日表 daily —— 每个 (date, symbol) 一行，必须是"当时可知"的值
   date, symbol, prev_close, adv30, float_shares
   可选: has_catalyst (bool)  —— 没有新闻时间戳时的简化替代
   可以用 prepare_daily() 从日线自动算 prev_close / adv30（只用前一日及更早的数据）

3) 新闻 news（可选）
   symbol, ts      —— 新闻发布时间
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional
import warnings

import numpy as np
import pandas as pd

ET = "America/New_York"

BAR_COLS = ["symbol", "ts", "open", "high", "low", "close", "volume"]


def _read(path) -> pd.DataFrame:
    p = Path(path)
    if p.suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    return pd.read_csv(p)


def _to_et(ts: pd.Series, naive_tz: str = ET) -> pd.Series:
    ts = pd.to_datetime(ts)
    if getattr(ts.dt, "tz", None) is None:
        ts = ts.dt.tz_localize(naive_tz)
    return ts.dt.tz_convert(ET)


def load_bars(path_or_df, naive_tz: str = ET) -> pd.DataFrame:
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else _read(path_or_df)
    df = df.copy()
    missing = [c for c in BAR_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"分钟线缺少列: {missing}")
    df["ts"] = _to_et(df["ts"], naive_tz)
    df["symbol"] = df["symbol"].astype(str)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "high", "low", "close", "volume"])

    # 基本健康检查
    bad = (df["high"] < df[["open", "close"]].max(axis=1) - 1e-9) | (
        df["low"] > df[["open", "close"]].min(axis=1) + 1e-9
    ) | (df["low"] <= 0)
    if bad.any():
        warnings.warn(f"丢弃 {int(bad.sum())} 根 OHLC 不自洽的 K 线")
        df = df[~bad]
    dup = df.duplicated(["symbol", "ts"])
    if dup.any():
        warnings.warn(f"丢弃 {int(dup.sum())} 根重复 K 线")
        df = df[~dup]

    df["date"] = df["ts"].dt.date
    df["minute"] = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    return df.sort_values(["date", "symbol", "ts"]).reset_index(drop=True)


def load_daily(path_or_df) -> pd.DataFrame:
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else _read(path_or_df)
    df = df.copy()
    need = ["date", "symbol", "prev_close", "adv30", "float_shares"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"日表缺少列: {missing}（可先用 prepare_daily 从日线生成 prev_close/adv30）")
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["symbol"] = df["symbol"].astype(str)
    return df


def load_news(path_or_df, naive_tz: str = ET) -> pd.DataFrame:
    df = path_or_df if isinstance(path_or_df, pd.DataFrame) else _read(path_or_df)
    df = df.copy()
    if not {"symbol", "ts"} <= set(df.columns):
        raise ValueError("新闻表需要 symbol, ts 两列")
    df["ts"] = _to_et(df["ts"], naive_tz)
    df["symbol"] = df["symbol"].astype(str)
    return df.sort_values(["symbol", "ts"]).reset_index(drop=True)


def prepare_daily(daily_bars: pd.DataFrame, float_table: Optional[pd.DataFrame] = None,
                  adv_window: int = 30, min_periods: int = 20) -> pd.DataFrame:
    """
    从日线 (date, symbol, close, volume) 生成日表。
    prev_close = 前一交易日收盘；adv30 = 前 30 个交易日日均量（shift(1)，不含当日）。

    volume 口径要和分钟线一致：分钟线含盘前，日线最好也用含盘前盘后的总量，
    否则相对量会被系统性高估。

    float_table: (date, symbol, float_shares)，应为"当时"的流通股（point-in-time），
    按 date 做 as-of 合并（取不晚于当日的最新值）。
    """
    d = daily_bars.copy()
    d["date"] = pd.to_datetime(d["date"])
    d["symbol"] = d["symbol"].astype(str)
    d = d.sort_values(["symbol", "date"])
    g = d.groupby("symbol", group_keys=False)
    d["prev_close"] = g["close"].shift(1)
    d["adv30"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(adv_window, min_periods=min_periods).mean()
    )
    out = d[["date", "symbol", "prev_close", "adv30"]].dropna()
    out = _merge_float(out, float_table)
    out["date"] = out["date"].dt.date
    return out.reset_index(drop=True)


def _merge_float(out: pd.DataFrame, float_table: Optional[pd.DataFrame]) -> pd.DataFrame:
    if float_table is not None and len(float_table):
        f = float_table.copy()
        f["date"] = pd.to_datetime(f["date"]).astype("datetime64[ns]")
        f["symbol"] = f["symbol"].astype(str)
        out = out.copy()
        out["date"] = pd.to_datetime(out["date"]).astype("datetime64[ns]")
        out = pd.merge_asof(out.sort_values("date"), f.sort_values("date"),
                            on="date", by="symbol", direction="backward")
    else:
        out = out.copy()
        out["float_shares"] = np.nan
    return out


def prepare_daily_split_aware(raw: pd.DataFrame, adj: pd.DataFrame,
                              float_table: Optional[pd.DataFrame] = None,
                              adv_window: int = 30, min_periods: int = 20) -> pd.DataFrame:
    """
    同时用"原始"和"拆股调整后"两份日线，生成不受拆股干扰的昨收和 ADV30。

    小盘股反向拆股（如 1 拆 10 合 1）很常见。只用原始价格时，拆股当天会出现假的 +900% 跳空；
    只用调整后价格时，历史价格被放大，$2–$20 的价格过滤会错。
    做法：f(t) = 调整后收盘 / 原始收盘（未来拆股累积因子），
      昨收(t)  = 原始收盘(t-1) × f(t-1) / f(t)          —— 换算到 t 日的股本口径
      ADV30(t) = f(t) × 均值[原始量(t-k) / f(t-k)], k=1..30
    分钟线保持原始价格，和这里的口径一致。
    """
    r = raw[["symbol", "date", "close", "volume"]].copy()
    a = adj[["symbol", "date", "close"]].rename(columns={"close": "adj_close"})
    for x in (r, a):
        x["date"] = pd.to_datetime(x["date"])
        x["symbol"] = x["symbol"].astype(str)
    d = r.merge(a, on=["symbol", "date"], how="left").sort_values(["symbol", "date"])
    f = d["adj_close"] / d["close"]
    d["f"] = f.where(np.isfinite(f) & (f > 0), 1.0).fillna(1.0)
    g = d.groupby("symbol", group_keys=False)
    d["prev_close"] = g["close"].shift(1) * g["f"].shift(1) / d["f"]
    d["adj_vol"] = d["volume"] / d["f"]
    d["adv30"] = d["f"] * d.groupby("symbol", group_keys=False)["adj_vol"].transform(
        lambda s: s.shift(1).rolling(adv_window, min_periods=min_periods).mean())
    out = d[["date", "symbol", "prev_close", "adv30"]].dropna()
    out = _merge_float(out, float_table)
    out["date"] = out["date"].dt.date
    return out.reset_index(drop=True)


def candidate_days(daily_bars_ext: pd.DataFrame, daily: pd.DataFrame,
                   min_pct: float = 0.10, min_rvol: float = 5.0,
                   min_price: float = 2.0, max_price: float = 20.0,
                   mode: str = "volume", vol_tolerance: float = 0.9) -> pd.DataFrame:
    """
    预筛：决定要下载哪些 (date, symbol) 的分钟线，避免拉全市场。只缩小下载范围，不参与交易判断。

    mode="volume"（默认）——只用"日内任何时刻满足五条件"的必要条件，不会漏票：
      当日总量 ≥ vol_tolerance × min_rvol × ADV30
          入场时的累计量（04:00 起）≥ 5×ADV30，而全天量 ≥ 入场时累计量。
          前提是日线成交量含盘前盘后（Alpaca 实测成立，数据质量报告会复核）。
      昨收 × (1 + min_pct) ≤ max_price
          否则涨够 10% 时价格必然高于上限。
      不用日线最高价：Alpaca 日线最高价不含盘前，按它筛会漏掉只在盘前拉升又回落的票。

    mode="price"——旧逻辑：日线最高价较昨收涨 ≥min_pct、量 ≥min_rvol×ADV30、价格区间有交集。
    """
    x = daily_bars_ext.copy()
    x["date"] = pd.to_datetime(x["date"]).dt.date
    x["symbol"] = x["symbol"].astype(str)
    x = x.merge(daily[["date", "symbol", "prev_close", "adv30"]], on=["date", "symbol"])
    if mode == "volume":
        m = ((x["volume"] >= vol_tolerance * min_rvol * x["adv30"])
             & (x["prev_close"] * (1 + min_pct) <= max_price))
    elif mode == "price":
        m = ((x["high"] / x["prev_close"] - 1 >= min_pct)
             & (x["volume"] >= min_rvol * x["adv30"])
             & (x["low"] <= max_price)
             & (x["high"] >= min_price))
    else:
        raise ValueError(mode)
    return x.loc[m, ["date", "symbol"]].reset_index(drop=True)

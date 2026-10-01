"""
指标：均线、KDJ、MACD、ATR，以及日线 → 周/月线重采样。
全部只用当根及以前的数据。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

RULE = {"W": "W-FRI", "M": "ME", "Q": "QE"}


def resample(daily: pd.DataFrame, tf: str) -> pd.DataFrame:
    """daily: index=日期, 列 open/high/low/close。tf: D / W / M。标签为该周期最后一个交易日。"""
    if tf in ("D", "H"):
        return daily.copy()
    g = daily.resample(RULE[tf])
    out = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                        "low": g["low"].min(), "close": g["close"].last(),
                        "last_day": g["close"].apply(lambda s: s.index.max() if len(s) else pd.NaT)})
    out = out.dropna(subset=["close"])
    out.index = pd.DatetimeIndex(out.pop("last_day"))
    return out


def kdj(df: pd.DataFrame, n: int = 9, m1: int = 3, m2: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """中文软件常用 KDJ：RSV(n)，K = (m1-1)/m1·K' + 1/m1·RSV，D 同理。初值 50。"""
    low_n = df["low"].rolling(n, min_periods=n).min()
    high_n = df["high"].rolling(n, min_periods=n).max()
    rng = (high_n - low_n).replace(0, np.nan)
    rsv = ((df["close"] - low_n) / rng * 100).to_numpy()
    K = np.full(len(df), np.nan); D = np.full(len(df), np.nan)
    k = d = 50.0
    for i, v in enumerate(rsv):
        if np.isnan(v):
            continue
        k = (m1 - 1) / m1 * k + v / m1
        d = (m2 - 1) / m2 * d + k / m2
        K[i], D[i] = k, d
    return K, D


def add_indicators(df: pd.DataFrame, mas=(20, 50, 80), atr_n: int = 14) -> pd.DataFrame:
    df = df.copy()
    for n in mas:
        df[f"ma{n}"] = df["close"].rolling(n, min_periods=n).mean()
    df["K"], df["D"] = kdj(df)
    e12 = df["close"].ewm(span=12, adjust=False).mean()
    e26 = df["close"].ewm(span=26, adjust=False).mean()
    df["macd"] = e12 - e26
    df["macd_sig"] = df["macd"].ewm(span=9, adjust=False).mean()
    pc = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"], (df["high"] - pc).abs(), (df["low"] - pc).abs()],
                   axis=1).max(axis=1)
    df["atr"] = tr.rolling(atr_n, min_periods=atr_n).mean()
    return df

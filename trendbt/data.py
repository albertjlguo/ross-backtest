"""
长历史日线 + 小时线下载。

日线优先级：Yahoo（yfinance，复权）→ Stooq（备用）；加密货币再用 CoinMetrics 社区数据
（只有收盘价，2010 年起）把 Yahoo 之前的历史补上，否则月线 80MA 要等将近 7 年才有值。
小时线：Alpaca（股票/ETF 2016 年起；加密货币走 crypto 接口）。
"""
from __future__ import annotations

import io
import logging
import time

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("trendbt.data")

COINMETRICS = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/{}.csv"
STOOQ = "https://stooq.com/q/d/l/?s={}&i=d"


def _clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df[["open", "high", "low", "close"]].astype(float)
    df = df[(df["close"] > 0)].dropna()
    df["high"] = df[["open", "high", "low", "close"]].max(axis=1)
    df["low"] = df[["open", "high", "low", "close"]].min(axis=1)
    df.index = pd.DatetimeIndex(df.index).tz_localize(None).normalize()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")].sort_index()


def yahoo_daily(sym: str) -> pd.DataFrame:
    import yfinance as yf
    h = yf.Ticker(sym).history(period="max", auto_adjust=True)
    if h is None or h.empty:
        return pd.DataFrame()
    h = h.rename(columns=str.lower)
    return _clean(h)


def stooq_daily(sym: str) -> pd.DataFrame:
    r = requests.get(STOOQ.format(sym), timeout=60)
    if r.status_code != 200 or not r.text.startswith("Date"):
        return pd.DataFrame()
    df = pd.read_csv(io.StringIO(r.text), parse_dates=["Date"]).set_index("Date")
    return _clean(df.rename(columns=str.lower))


def coinmetrics_daily(asset: str) -> pd.DataFrame:
    r = requests.get(COINMETRICS.format(asset), timeout=120)
    r.raise_for_status()
    df = pd.read_csv(io.StringIO(r.text), usecols=lambda c: c in ("time", "PriceUSD"))
    df = df.dropna()
    s = df.set_index(pd.to_datetime(df["time"]))["PriceUSD"].astype(float)
    o = s.shift(1).fillna(s)
    out = pd.DataFrame({"open": o, "high": np.maximum(o, s), "low": np.minimum(o, s), "close": s})
    return _clean(out)


def splice(primary: pd.DataFrame, older: pd.DataFrame) -> pd.DataFrame:
    """用 older 补上 primary 开始之前的历史（按重叠首日的价格比例对齐）。"""
    if older.empty:
        return primary
    if primary.empty:
        return older
    first = primary.index[0]
    ov = older.index[older.index >= first]
    scale = primary["close"].iloc[0] / older.loc[ov[0], "close"] if len(ov) else 1.0
    pre = older[older.index < first] * scale
    return pd.concat([pre, primary]).sort_index()


def fetch_daily(spec: dict) -> pd.DataFrame:
    df = pd.DataFrame()
    if spec.get("yahoo"):
        for attempt in range(3):
            try:
                df = yahoo_daily(spec["yahoo"])
                break
            except Exception as e:  # yfinance 偶尔限流
                log.warning("Yahoo %s 失败（%s），重试", spec["yahoo"], e)
                time.sleep(5 * (attempt + 1))
    if df.empty and spec.get("stooq"):
        try:
            df = stooq_daily(spec["stooq"])
        except Exception as e:
            log.warning("Stooq %s 失败: %s", spec["stooq"], e)
    if spec.get("coinmetrics"):
        try:
            df = splice(df, coinmetrics_daily(spec["coinmetrics"]))
        except Exception as e:
            log.warning("CoinMetrics %s 失败: %s", spec["coinmetrics"], e)
    return df


def fetch_hourly(spec: dict, alp, start: str = "2016-01-01", end: str | None = None) -> pd.DataFrame:
    """Alpaca 小时线。股票只保留常规时段（10:00–15:00 起始的整点 K 线 + 9:00 那根含开盘）。"""
    end = end or (pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    if spec.get("alpaca_crypto"):
        df = alp.crypto_bars([spec["alpaca_crypto"]], "1Hour", start, end)
    elif spec.get("alpaca"):
        df = alp.bars([spec["alpaca"]], "1Hour", start, end, adjustment="all")
        if not df.empty:
            et = df["ts"].dt.tz_convert("America/New_York")
            df = df[(et.dt.hour >= 9) & (et.dt.hour <= 15)]
    else:
        return pd.DataFrame()
    if df.empty:
        return df
    df = df.set_index(df["ts"].dt.tz_convert("America/New_York").dt.tz_localize(None))
    df = df[["open", "high", "low", "close"]].astype(float).sort_index()
    df.index.name = "date"
    return df[~df.index.duplicated(keep="last")]

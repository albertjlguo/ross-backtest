"""
在支撑位卖看跌期权收租（cash-secured put）——用 Black-Scholes 近似定价。

没有免费的历史期权报价，所以权利金是模型算的：
  隐含波动率 ≈ 过去 21 个交易日已实现波动率 × iv_mult（默认 1.15；实际 IV 通常高于已实现波动）
  实值/虚值偏斜（虚值 put 的 IV 更高）没有建模——会低估权利金，部分抵消 iv_mult 的偏高。
  成交按理论价 × (1 − haircut) 计，模拟买卖价差。
所以这里的绝对收益只是量级参考；"按信号卖"vs"每月都卖"的相对比较更可靠。

每个月第一个交易日：
  signal 模式：上个月末月线"健康"时，在现价下方 min_otm～max_otm 之间最高的那个支撑位
               （MA50 / MA80 / 斐波那契 0.382 / 0.5 / 0.618）卖一个月到期的 put；没有就不卖。
  delta 模式：同样只在"健康"时卖，但行权价取 delta = −0.25。
  always 模式：每个月都卖 delta −0.25（对照组，类似 CBOE PUT 指数的做法）。
到期（月末收盘）：收益 = 权利金 − max(行权价 − 收盘价, 0)，除以行权价（全额现金担保）。
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt

import numpy as np
import pandas as pd

from .strategy import monthly_state_on_daily


def _ncdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def bs_put(S, K, T, sigma, r=0.0):
    if T <= 0 or sigma <= 0:
        return max(K - S, 0.0)
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return K * exp(-r * T) * _ncdf(-d2) - S * _ncdf(-d1)


def put_delta(S, K, T, sigma, r=0.0):
    d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
    return _ncdf(d1) - 1


def strike_for_delta(S, T, sigma, target=-0.25):
    lo, hi = S * 0.3, S
    for _ in range(60):
        mid = (lo + hi) / 2
        if put_delta(S, mid, T, sigma) < target:   # delta 更负 = 离现价太近 → 行权价往下调
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


@dataclass(frozen=True)
class CSPParams:
    mode: str = "signal"        # signal | delta | always
    min_otm: float = 0.03
    max_otm: float = 0.25
    iv_mult: float = 1.15
    haircut: float = 0.05
    ann_days: int = 252         # 加密货币用 365


def simulate_csp(daily: pd.DataFrame, cp: CSPParams = CSPParams(), asset: str = "") -> pd.DataFrame:
    st = monthly_state_on_daily(daily)
    lr = np.log(daily["close"]).diff()
    vol = lr.rolling(21).std() * sqrt(cp.ann_days)
    st = st.join(vol.rename("rv"))
    st["ym"] = st.index.to_period("M")
    out = []
    for ym, g in st.groupby("ym"):
        if len(g) < 5:
            continue
        d0, dT = g.index[0], g.index[-1]
        r0 = g.iloc[0]
        S = r0["open"]; S_T = g.iloc[-1]["close"]
        sigma = r0["rv"] * cp.iv_mult if not np.isnan(r0["rv"]) else np.nan
        T = max((dT - d0).days, 1) / 365
        h = r0.get("m_healthy")
        healthy = isinstance(h, (bool, np.bool_)) and bool(h)
        K = None
        if np.isnan(sigma):
            pass
        elif cp.mode == "always" or (cp.mode == "delta" and healthy):
            K = strike_for_delta(S, T, sigma)
        elif cp.mode == "signal" and healthy:
            sups = [r0.get(k) for k in ("m_ma50", "m_ma80", "m_f0.382", "m_f0.5", "m_f0.618")]
            sups = [s for s in sups if s is not None and not np.isnan(s)
                    and S * (1 - cp.max_otm) <= s <= S * (1 - cp.min_otm)]
            if sups:
                K = max(sups)
        if K is None:
            out.append({"month": str(ym), "asset": asset, "sold": False, "assigned": False, "ret": 0.0})
            continue
        prem = bs_put(S, K, T, sigma) * (1 - cp.haircut)
        pnl = prem - max(K - S_T, 0.0)
        out.append({"month": str(ym), "asset": asset, "sold": True, "S0": S, "K": K,
                    "otm": 1 - K / S, "sigma": sigma, "premium_pct": prem / K,
                    "assigned": bool(S_T < K), "S_T": S_T, "ret": pnl / K})
    df = pd.DataFrame(out)
    if len(df):
        df["assigned"] = df["assigned"].astype(bool)
        df["sold"] = df["sold"].astype(bool)
    return df

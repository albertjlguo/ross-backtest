"""
统计：方差比检验（某个周期上价格更像趋势还是更像随机游走）、组合月度收益与常用指标。
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def variance_ratio(logret: pd.Series, q: int) -> tuple[float, float]:
    """
    Lo–MacKinlay 方差比 VR(q) 及异方差稳健 z 值。
    VR > 1：q 期收益的方差大于单期的 q 倍 → 趋势/动量；VR < 1 → 均值回归；≈1 → 随机游走。
    """
    x = logret.dropna().to_numpy()
    T = len(x)
    if T < q * 10:
        return np.nan, np.nan
    mu = x.mean()
    s1 = ((x - mu) ** 2).sum() / (T - 1)
    xq = np.convolve(x, np.ones(q), "valid")
    m = q * (T - q + 1) * (1 - q / T)
    sq = ((xq - q * mu) ** 2).sum() / m
    vr = sq / s1
    # 异方差稳健的渐近方差
    dev2 = (x - mu) ** 2
    denom = dev2.sum() ** 2
    theta = 0.0
    for j in range(1, q):
        delta = (dev2[j:] * dev2[:-j]).sum() * T / denom
        theta += (2 * (q - j) / q) ** 2 * delta
    z = (vr - 1) / np.sqrt(theta / T) if theta > 0 else np.nan
    return float(vr), float(z)


def bar_to_monthly(ret: pd.Series) -> pd.Series:
    """把任意周期的持仓收益序列复利汇总到自然月。"""
    if ret.empty:
        return ret
    r = (1 + ret).groupby(ret.index.to_period("M")).prod() - 1
    return r


def portfolio_monthly(sleeves: dict[str, pd.Series]) -> pd.Series:
    """等权：每个品种一份资金，没持仓的那份就是现金（收益 0）。只统计品种已有数据的月份。"""
    df = pd.DataFrame(sleeves)
    return df.mean(axis=1, skipna=True)


def metrics(monthly: pd.Series) -> dict:
    m = monthly.dropna()
    if len(m) < 12:
        return {"months": len(m)}
    eq = (1 + m).cumprod()
    years = len(m) / 12
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = m.std() * np.sqrt(12)
    dd = (eq / eq.cummax() - 1).min()
    return {"months": len(m), "cagr": round(cagr, 4), "vol": round(vol, 4),
            "sharpe": round(m.mean() / m.std() * np.sqrt(12), 3) if m.std() > 0 else np.nan,
            "max_dd": round(dd, 4), "pct_pos_months": round((m > 0).mean(), 3),
            "worst_month": round(m.min(), 4), "best_month": round(m.max(), 4)}


def trade_stats(trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {"trades": 0}
    r = trades["ret"]
    lr = np.log1p(r)
    return {"trades": len(r), "win_rate": round((r > 0).mean(), 3), "avg_ret": round(r.mean(), 4),
            "avg_log_ret": round(lr.mean(), 4),
            "t_log": round(lr.mean() / lr.std(ddof=1) * np.sqrt(len(lr)), 2) if len(lr) > 2 else np.nan,
            "median_ret": round(r.median(), 4), "avg_bars": round(trades["bars"].mean(), 1),
            "t_stat": round(r.mean() / r.std(ddof=1) * np.sqrt(len(r)), 2) if len(r) > 2 else np.nan}

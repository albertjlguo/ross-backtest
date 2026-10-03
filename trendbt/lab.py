"""
因子实验室：用同一套"文献里的默认参数"在所有品种上检验几类最常见的系统化收益来源。

原则（防止"测到有为止"）：
  1. 参数全部事先写死，取文献常用值，不做任何调参。
  2. 样本分三段：IS（≤2014）、OOS（2015–2020）、HOLD（2021 至今）。三段都要为正才算数。
  3. 报告 t 值和多重检验门槛：测了 N 个组合，|t| 要超过 Bonferroni 门槛才不算运气。
  4. 信号在 t 日收盘算出，t+1 日才持仓；换手按每个品种的单边成本扣费。

策略（w = 权重，都先按各自波动率归一，再把整个组合的波动率调到 target_vol）：
  bh_vt     等风险买入持有（基准：只有 beta）
  ma200     价格在 200 日均线上方持有，否则空仓
  tsmom12   时间序列动量：过去 12 个月涨就做多，跌就做空
  trend4    1/3/6/12 个月四个周期方向的平均（多空）
  trend4_lo 同上，只做多
  xs_mom    横截面动量：组内 12-1 个月涨幅排名，多强空弱（市场中性）
  xs_rev    横截面短期反转：组内过去 5 日跌得多的做多、涨得多的做空（市场中性，日频换仓）
  lowvol    组内低波动做多、高波动做空（按波动率归一后市场中性）
"""
from __future__ import annotations

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd

PERIODS = {"IS": (None, "2014-12-31"), "OOS": ("2015-01-01", "2020-12-31"),
           "HOLD": ("2021-01-01", None), "FULL": (None, None)}
STRATS = ["bh_vt", "ma200", "tsmom12", "trend4", "trend4_lo", "xs_mom", "xs_rev", "lowvol"]
XS = {"xs_mom", "xs_rev", "lowvol"}


@dataclass(frozen=True)
class LabParams:
    target_vol: float = 0.10        # 组合年化波动率目标
    vol_span: int = 60              # 波动率估计：60 日指数加权
    max_lev: float = 3.0            # 组合层面杠杆上限
    rebal: int = 5                  # 每 5 个交易日调仓（xs_rev 每日）
    min_assets: int = 4             # 横截面策略组内至少 4 个品种
    min_hist: int = 273             # 至少 12 个月 + 1 个月历史才参与


def panel(daily: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """{品种: 日线} → 工作日对齐的收盘价面板（加密货币周末的涨跌并入周一）。"""
    px = pd.DataFrame({n: d["close"] for n, d in daily.items() if len(d)})
    px = px.sort_index()
    bd = pd.bdate_range(px.index.min(), px.index.max())
    return px.reindex(px.index.union(bd)).ffill(limit=5).reindex(bd)


def _xs_rank(x: pd.DataFrame, min_assets: int) -> pd.DataFrame:
    """每行按排名去均值，绝对值之和归一到 1；有效品种不够则全 0。"""
    r = x.rank(axis=1)
    n = x.notna().sum(axis=1)
    r = r.sub((n + 1) / 2, axis=0)
    r = r.div(r.abs().sum(axis=1).replace(0, np.nan), axis=0)
    return r.where(n >= min_assets, 0.0).fillna(0.0)


def raw_weights(px: pd.DataFrame, strat: str, p: LabParams) -> pd.DataFrame:
    ret = px.pct_change(fill_method=None)
    vol = ret.ewm(span=p.vol_span, min_periods=p.vol_span).std() * sqrt(252)
    ok = px.notna() & (px.notna().cumsum() >= p.min_hist) & vol.gt(0)
    inv = (p.target_vol / vol).where(ok)
    n = ok.sum(axis=1).replace(0, np.nan)
    if strat == "bh_vt":
        sig = pd.DataFrame(1.0, index=px.index, columns=px.columns)
    elif strat == "ma200":
        sig = (px > px.rolling(200, min_periods=200).mean()).astype(float)
    elif strat == "tsmom12":
        sig = np.sign(px / px.shift(252) - 1)
    elif strat in ("trend4", "trend4_lo"):
        sig = sum(np.sign(px / px.shift(k) - 1) for k in (21, 63, 126, 252)) / 4
        if strat == "trend4_lo":
            sig = sig.clip(lower=0)
    elif strat == "xs_mom":
        return _xs_rank((px.shift(21) / px.shift(252) - 1).where(ok), p.min_assets) * inv.fillna(0)
    elif strat == "xs_rev":
        return _xs_rank(-(px / px.shift(5) - 1).where(ok), p.min_assets) * inv.fillna(0)
    elif strat == "lowvol":
        return _xs_rank(-vol.where(ok), p.min_assets) * inv.fillna(0)
    else:
        raise ValueError(strat)
    return (sig * inv).div(n, axis=0).where(ok, 0.0).fillna(0.0)


def run(px: pd.DataFrame, strat: str, cost_bps: pd.Series, p: LabParams = LabParams()) -> dict:
    """返回 {'ret': 日收益（扣成本）, 'gross': 平均总杠杆, 'turnover': 年换手}。"""
    ret = px.pct_change(fill_method=None).fillna(0.0)
    w = raw_weights(px, strat, p)
    k = 1 if strat == "xs_rev" else p.rebal
    if k > 1:
        keep = np.arange(len(w)) % k == 0
        w = w.where(pd.Series(keep, index=w.index), np.nan, axis=0).ffill().fillna(0.0)
    # 组合层面的波动率目标：用"未加杠杆组合"过去的实现波动率，滞后一天
    pre = (w.shift(1) * ret).sum(axis=1)
    rv = pre.ewm(span=p.vol_span, min_periods=p.vol_span).std() * sqrt(252)
    lev = (p.target_vol / rv).clip(upper=p.max_lev).shift(1)
    if k > 1:
        lev = lev.where(pd.Series(keep, index=lev.index)).ffill()
    w = w.mul(lev.fillna(0.0), axis=0)
    pos = w.shift(1).fillna(0.0)
    c = cost_bps.reindex(px.columns).fillna(5.0) / 1e4
    cost = (pos.diff().abs().fillna(0.0) * c).sum(axis=1)
    r = (pos * ret).sum(axis=1) - cost
    live = pos.abs().sum(axis=1) > 0
    r = r[live.cummax()] if live.any() else r.iloc[0:0]
    return {"ret": r, "gross": float(pos.abs().sum(axis=1)[live].mean()) if live.any() else np.nan,
            "turnover": float(pos.diff().abs().sum(axis=1)[live].mean() * 252) if live.any() else np.nan,
            "cost_drag": float(cost[live].mean() * 252) if live.any() else np.nan}


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 250:
        return {"days": len(r)}
    m = (1 + r).groupby(r.index.to_period("M")).prod() - 1
    eq = (1 + r).cumprod()
    sd = r.std()
    sh = r.mean() / sd * sqrt(252) if sd > 0 else np.nan
    return {"days": len(r), "years": round(len(r) / 252, 1),
            "ann_ret": round(eq.iloc[-1] ** (252 / len(r)) - 1, 4), "vol": round(sd * sqrt(252), 4),
            "sharpe": round(sh, 3), "t": round(sh * sqrt(len(r) / 252), 2),
            "max_dd": round((eq / eq.cummax() - 1).min(), 4),
            "pct_pos_months": round((m > 0).mean(), 3), "worst_month": round(m.min(), 4),
            "pct_pos_years": round(((1 + r).groupby(r.index.year).prod() > 1).mean(), 3)}


def cut(r: pd.Series, period: str) -> pd.Series:
    a, b = PERIODS[period]
    return r.loc[a:b]


def bonferroni_t(n_trials: int, alpha: float = 0.05) -> float:
    """N 次检验下，单次双侧显著需要的 |t|（正态近似）。"""
    target = 1 - alpha / max(1, n_trials) / 2
    lo, hi = 0.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if 0.5 * (1 + erf(mid / sqrt(2))) < target:
            lo = mid
        else:
            hi = mid
    return round(hi, 2)

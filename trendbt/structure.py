"""
市场结构：实时 zigzag 波段 + 斐波那契锚点。全部"当时可知"。

波段确认：从上一个极值回撤超过 k×ATR 才确认那个极值（高点要等价格跌够才算成立）。
斐波那契锚点（按你的做法写成规则）：
  高点 H = 最近一个已确认的波段高点
  低点 L = H 之前 lookback 根 K 线内、处于"动能低位"的波段低点里最低的一个；
           动能低位 = 该点前后 1 根内 K ≤ kd_os，且 MACD < 0 或 MACD 低于信号线
           （月线 MACD 滞后很多，2018 年底 BTC 见底时 MACD 仍为正，但已在信号线下方）。
           找不到满足条件的，就取窗口内最低价。
价位：L + k × (H − L)，k ∈ 0.236 / 0.382 / 0.5 / 0.618 / 0.786
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StructParams:
    zz_atr_k: float = 3.0       # 回撤超过 3×ATR 确认波段
    lookback: int = 72          # 找低点的回看根数（月线 = 6 年）
    kd_os: float = 20.0         # 动能低位：K ≤ 20
    macd_rule: str = "neg_or_hist"   # 动能低位里 MACD 的条件："neg" MACD<0 / "neg_or_hist" MACD<0 或低于信号线
    low_pick: str = "recent"    # "recent" 最近一个动能低位的低点（周期底）/ "lowest" 窗口内最低的


def zigzag(df: pd.DataFrame, k: float) -> list[tuple[int, float, str, int]]:
    """返回 [(极值所在位置, 价格, 'H'/'L', 确认位置)]，按确认时间排序。"""
    hi, lo, atr = df["high"].to_numpy(), df["low"].to_numpy(), df["atr"].to_numpy()
    n = len(df)
    piv = []
    if n == 0:
        return piv
    d = 1                      # 1 = 正在寻找高点，-1 = 正在寻找低点
    ext, ext_i = hi[0], 0
    last_h = last_l = None
    for t in range(1, n):
        # 回落途中又创出新高 → 上一个高点作废，继续找高点（反之亦然）
        if d == -1 and last_h is not None and hi[t] > last_h:
            d, ext, ext_i = 1, hi[t], t
            continue
        if d == 1 and last_l is not None and lo[t] < last_l:
            d, ext, ext_i = -1, lo[t], t
            continue
        a = atr[t]
        if np.isnan(a):
            if d == 1 and hi[t] > ext:
                ext, ext_i = hi[t], t
            elif d == -1 and lo[t] < ext:
                ext, ext_i = lo[t], t
            continue
        if d == 1:
            if hi[t] >= ext:
                ext, ext_i = hi[t], t
            elif lo[t] <= ext - k * a:
                piv.append((ext_i, ext, "H", t))
                last_h = ext
                j = int(np.argmin(lo[ext_i + 1:t + 1])) + ext_i + 1
                d, ext, ext_i = -1, lo[j], j
        else:
            if lo[t] <= ext:
                ext, ext_i = lo[t], t
            elif hi[t] >= ext + k * a:
                piv.append((ext_i, ext, "L", t))
                last_l = ext
                j = int(np.argmax(hi[ext_i + 1:t + 1])) + ext_i + 1
                d, ext, ext_i = 1, hi[j], j
    return piv


def fib_anchors(df: pd.DataFrame, p: StructParams = StructParams()) -> pd.DataFrame:
    """每根 K 线收盘时可知的 (H, H_idx, L, L_idx)。df 需含 high/low/atr/K/macd。"""
    n = len(df)
    piv = zigzag(df, p.zz_atr_k)
    lo = df["low"].to_numpy()
    K = df["K"].to_numpy(); macd = df["macd"].to_numpy()
    hist = (df["macd"] - df["macd_sig"]).to_numpy()
    H = np.full(n, np.nan); Hi = np.full(n, -1); L = np.full(n, np.nan); Li = np.full(n, -1)
    lows = [(i, px) for i, px, typ, c in piv if typ == "L"]
    cur = None
    pi = 0
    highs = [(i, px, c) for i, px, typ, c in piv if typ == "H"]
    for t in range(n):
        while pi < len(highs) and highs[pi][2] <= t:
            hi_i, hi_px, conf = highs[pi]
            w0 = max(0, hi_i - p.lookback)
            cands = []
            for li, lpx in lows:
                if not (w0 <= li < hi_i):
                    continue
                # 低点本身必须在 H 确认时已经被确认（它在 H 之前，自然成立）
                seg_k = K[max(0, li - 1):li + 2]
                if np.all(np.isnan(seg_k)):
                    continue
                kk = np.nanmin(seg_k)
                mm = np.nanmin(macd[max(0, li - 1):li + 2])
                hh = np.nanmin(hist[max(0, li - 1):li + 2])
                macd_low = mm < 0 or (p.macd_rule == "neg_or_hist" and hh < 0)
                ok = (not np.isnan(kk)) and kk <= p.kd_os and macd_low
                if ok:
                    cands.append((lpx, li))
            if cands:
                if p.low_pick == "recent":
                    lpx, li = max(cands, key=lambda x: x[1])
                else:
                    lpx, li = min(cands)
            else:
                seg = lo[w0:hi_i]
                if len(seg) == 0:
                    pi += 1
                    continue
                li = int(np.argmin(seg)) + w0
                lpx = lo[li]
            if lpx < hi_px:
                cur = (hi_px, hi_i, lpx, li)
            pi += 1
        if cur:
            H[t], Hi[t], L[t], Li[t] = cur
    return pd.DataFrame({"H": H, "H_idx": Hi, "L": L, "L_idx": Li}, index=df.index)


def fib_level(H, L, k):
    return L + k * (H - L)

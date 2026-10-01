"""
策略模拟：单一周期（月/周/日/小时用同一套规则，参数按 K 线根数）和"月线定方向 + 日线入场"。

单周期规则
  健康：收盘在斐波那契 0.236 之上；可选再要求 80MA 向上（ma80 ≥ 3 根前）。
  入场：上一根收盘时，在"收盘价下方、0.236 上方"最近的支撑位挂限价买单；
        本根最低价触及即成交，成交价 = min(开盘, 支撑位)。
        支撑位来源：ma = MA50/MA80；fib = 0.382/0.5/0.618；confluence = 斐波那契位与 MA50/80
        相差 ≤ 3% 的共振位；any = ma ∪ fib；random = 健康时随机入场（对照组）。
        confirm=True 时不挂单：触及支撑后 3 根内 KD 金叉（K<50）才在下一根开盘买。
  止损（所有离场方式都有）：收盘跌破入场时的 0.236 → 下一根开盘卖。
        出现更高的新波段后，止损跟随上移到新的 0.236（只上不下）。
  离场方式：
    struct  只用上面的结构止损，一直拿着
    target  另在前高 H 挂限价卖单
    momo    动能转弱：MACD 在零轴上死叉，或 K>80 时 KD 死叉 → 下一根开盘卖
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
import pandas as pd

from .indicators import add_indicators, resample
from .structure import StructParams, fib_anchors, fib_level


@dataclass(frozen=True)
class StratParams:
    entry: str = "any"                       # ma | fib | confluence | any | random
    fib_levels: tuple = (0.382, 0.5, 0.618)
    stop_k: float = 0.236
    conf_tol: float = 0.03
    health: str = "fib_ma80up"               # fib | fib_ma80up
    exit: str = "struct"                     # struct | target | momo
    confirm: bool = False
    trail: bool = True
    cost_bps: float = 10.0                   # 单边成本
    random_p: float = 0.05
    seed: int = 0

    def with_(self, **kw):
        return replace(self, **kw)


def prepare(daily: pd.DataFrame, tf: str, sp: StructParams = StructParams()) -> pd.DataFrame:
    bars = add_indicators(resample(daily, tf))
    return bars.join(fib_anchors(bars, sp))


def _supports(row, sp: StratParams, close_prev, stop_lvl):
    """收盘价下方、止损上方的候选支撑位，返回 [(价位, 类型)] 从高到低。"""
    H, L = row["H"], row["L"]
    ma = [(row["ma50"], "ma50"), (row["ma80"], "ma80")]
    fib = [(fib_level(H, L, k), f"fib{k}") for k in sp.fib_levels]
    if sp.entry == "ma":
        c = ma
    elif sp.entry == "fib":
        c = fib
    elif sp.entry == "confluence":
        c = []
        for fv, fn in fib:
            for mv, mn in ma:
                if not np.isnan(mv) and abs(fv - mv) / fv <= sp.conf_tol:
                    c.append(((fv + mv) / 2, f"{fn}+{mn}"))
    else:  # any
        c = ma + fib
    c = [(v, n) for v, n in c if not np.isnan(v) and stop_lvl < v < close_prev]
    return sorted(c, reverse=True)


def healthy(row, sp: StratParams, ma80_prev3) -> bool:
    if np.isnan(row["H"]) or np.isnan(row["L"]):
        return False
    if row["close"] <= fib_level(row["H"], row["L"], sp.stop_k):
        return False
    if sp.health == "fib_ma80up":
        return not np.isnan(row["ma80"]) and not np.isnan(ma80_prev3) and row["ma80"] >= ma80_prev3
    return True


def simulate(bars: pd.DataFrame, sp: StratParams, asset: str = "", tf: str = "") -> tuple[list, pd.Series]:
    """返回 (交易列表, 每根 K 线的持仓收益序列)。"""
    rows = bars.to_dict("records")
    idx = bars.index
    n = len(rows)
    rng = np.random.default_rng(sp.seed)
    c = sp.cost_bps / 1e4
    ret = np.zeros(n)
    trades = []
    pos = None
    pending_exit = False
    pending_entry = None      # confirm 模式：上一根收盘出现信号，本根开盘买
    touch_t, touch_name = -10, None

    def ma80_3(i):
        return rows[i - 3]["ma80"] if i >= 3 else np.nan

    for t in range(1, n):
        r, p = rows[t], rows[t - 1]
        # ---------- 持仓：开盘挂起离场 / 目标价 / 正常持有 ----------
        if pos is not None:
            if pending_exit:
                px = r["open"]
                ret[t] += px / p["close"] - 1 - c
                _close(trades, pos, idx[t], px, pos["why"], t)
                pos, pending_exit = None, False
            elif sp.exit == "target" and r["high"] >= pos["target"]:
                px = max(r["open"], pos["target"])
                ret[t] += px / p["close"] - 1 - c
                _close(trades, pos, idx[t], px, "target", t)
                pos = None
            else:
                ret[t] += r["close"] / p["close"] - 1
        # ---------- 空仓：入场 ----------
        elif True:
            ok = healthy(p, sp, ma80_3(t - 1))
            stop_lvl = fib_level(p["H"], p["L"], sp.stop_k)
            entry_px = why = None
            if pending_entry is not None:
                entry_px, why = r["open"], pending_entry
                pending_entry = None
            elif ok and sp.entry == "random":
                if rng.random() < sp.random_p:
                    entry_px, why = r["open"], "random"
            elif ok:
                sup = _supports(p, sp, p["close"], stop_lvl)
                if sup and r["low"] <= sup[0][0]:
                    if sp.confirm:
                        touch_t, touch_name = t, sup[0][1]
                    else:
                        entry_px, why = min(r["open"], sup[0][0]), sup[0][1]
            if entry_px is not None and not np.isnan(stop_lvl) and entry_px > stop_lvl:
                pos = {"entry_i": t, "entry_t": idx[t], "entry_px": entry_px, "level": why,
                       "stop": stop_lvl, "H": p["H"], "L": p["L"], "target": p["H"], "why": None}
                ret[t] += r["close"] / entry_px - 1 - c
        # ---------- 收盘：确认信号 / 止损 / 动能离场 ----------
        if pos is None and sp.confirm and 0 <= t - touch_t <= 3:
            kx = (p["K"] <= p["D"] and r["K"] > r["D"] and r["K"] < 50)
            if kx and healthy(r, sp, ma80_3(t)):
                pending_entry, touch_t = touch_name + "+kd", -10
        if pos is not None:
            if sp.trail and not np.isnan(r["H"]) and r["H"] > pos["H"]:
                pos["stop"] = max(pos["stop"], fib_level(r["H"], r["L"], sp.stop_k))
                pos["H"], pos["L"] = r["H"], r["L"]
            if r["close"] < pos["stop"]:
                pending_exit, pos["why"] = True, "stop"
            elif sp.exit == "momo" and t > pos["entry_i"]:
                macd_x = p["macd"] >= p["macd_sig"] and r["macd"] < r["macd_sig"] and r["macd"] > 0
                kd_x = p["K"] >= p["D"] and r["K"] < r["D"] and p["K"] > 80
                if macd_x or kd_x:
                    pending_exit, pos["why"] = True, "momo"
    if pos is not None:   # 期末仍持仓：按最后收盘计
        _close(trades, pos, idx[n - 1], rows[n - 1]["close"], "open_end", n - 1)
    for tr in trades:
        tr.update(asset=asset, tf=tf)
    return trades, pd.Series(ret, index=idx)


def _close(trades, pos, t_exit, px, why, i):
    trades.append({"entry_t": pos["entry_t"], "entry_px": pos["entry_px"], "exit_t": t_exit,
                   "exit_px": px, "ret": px / pos["entry_px"] - 1, "bars": i - pos["entry_i"],
                   "level": pos["level"], "exit_why": why})


# --------------------------------------------------------------------------- #
#  月线定方向 + 日线入场
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MTFParams:
    zone_above: float = 0.08     # 日线价格在月线支撑上方 8% 以内
    zone_below: float = 0.05     # 或下方 5% 以内
    kd_os: float = 30.0          # 日线 KD 在 30 以下金叉
    stop_buf: float = 0.05       # 收盘跌破所用支撑位 5% 止损
    exit: str = "momo"           # momo（日线动能转弱）| r2（2 倍风险目标）| monthly（月线 0.236 结构止损）
    cost_bps: float = 10.0


def monthly_state_on_daily(daily: pd.DataFrame, sp: StratParams = StratParams()) -> pd.DataFrame:
    """把"上一个已完成月份"的月线状态对齐到每个交易日。"""
    m = prepare(daily, "M")
    m["ma80_p3"] = m["ma80"].shift(3)
    m["healthy"] = [healthy(r, sp, r["ma80_p3"]) for r in m.to_dict("records")]
    for k in (0.236, 0.382, 0.5, 0.618):
        m[f"f{k}"] = fib_level(m["H"], m["L"], k)
    cols = ["healthy", "ma50", "ma80", "f0.236", "f0.382", "f0.5", "f0.618"]
    mm = m[cols].add_prefix("m_").reset_index().rename(columns={m.index.name or "index": "m_date"})
    d = daily.reset_index().rename(columns={daily.index.name or "index": "date"})
    out = pd.merge_asof(d.sort_values("date"), mm.sort_values("m_date"), left_on="date",
                        right_on="m_date", allow_exact_matches=False, direction="backward")
    return out.set_index("date")


def simulate_mtf(daily: pd.DataFrame, mp: MTFParams = MTFParams(), asset: str = "") -> tuple[list, pd.Series]:
    d = add_indicators(daily)
    st = monthly_state_on_daily(daily)
    d = d.join(st[[c for c in st.columns if c.startswith("m_")]])
    rows = d.to_dict("records"); idx = d.index; n = len(rows)
    c = mp.cost_bps / 1e4
    ret = np.zeros(n); trades = []
    pos = None; pending = False
    for t in range(2, n):
        r, p, pp = rows[t], rows[t - 1], rows[t - 2]
        if pos is not None:
            if pending:
                px = r["open"]; ret[t] += px / p["close"] - 1 - c
                _close(trades, pos, idx[t], px, pos["why"], t); pos, pending = None, False
            elif mp.exit == "r2" and r["high"] >= pos["target"]:
                px = max(r["open"], pos["target"]); ret[t] += px / p["close"] - 1 - c
                _close(trades, pos, idx[t], px, "target", t); pos = None
            else:
                ret[t] += r["close"] / p["close"] - 1
        mh = p.get("m_healthy")
        if pos is None and not pending and isinstance(mh, (bool, np.bool_)) and bool(mh):
            sups = [p.get(k) for k in ("m_ma50", "m_ma80", "m_f0.382", "m_f0.5", "m_f0.618")]
            sups = [s for s in sups if s is not None and not np.isnan(s) and s > p["m_f0.236"]]
            near = [s for s in sups if s * (1 - mp.zone_below) <= p["close"] <= s * (1 + mp.zone_above)]
            kx = (not np.isnan(p["K"]) and pp["K"] <= pp["D"] and p["K"] > p["D"] and pp["K"] < mp.kd_os)
            if near and kx:
                s = max(near) if any(x <= p["close"] for x in near) else min(near)
                stop = s * (1 - mp.stop_buf)
                px = r["open"]
                if px > stop:
                    pos = {"entry_i": t, "entry_t": idx[t], "entry_px": px, "level": "mtf",
                           "stop": stop, "target": px + 2 * (px - stop), "why": None,
                           "m_stop": p["m_f0.236"]}
                    ret[t] += r["close"] / px - 1 - c
        if pos is not None and not pending and pos["entry_i"] <= t:
            stop = pos["m_stop"] if mp.exit == "monthly" else pos["stop"]
            if r["close"] < stop:
                pending, pos["why"] = True, "stop"
            elif mp.exit == "momo" and t > pos["entry_i"]:
                if (p["macd"] >= p["macd_sig"] and r["macd"] < r["macd_sig"] and r["macd"] > 0) or \
                   (p["K"] >= p["D"] and r["K"] < r["D"] and p["K"] > 80):
                    pending, pos["why"] = True, "momo"
    if pos is not None:
        _close(trades, pos, idx[n - 1], rows[n - 1]["close"], "open_end", n - 1)
    for tr in trades:
        tr.update(asset=asset, tf="MTF")
    return trades, pd.Series(ret, index=idx)

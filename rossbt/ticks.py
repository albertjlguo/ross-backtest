"""
用逐笔成交解决"同一根 1 分钟 K 线里两个价位都被触及，谁先谁后"。

事件：(symbol, 分钟起点 UTC, up, down)
  入场：up = 触发价，down = 回调低点（止损）。up 先 → 真实成交后被止损；down 先 → 形态先破，不入场。
  持仓：up = 下一个目标价，down = 止损。

剔除不反映当时成交顺序的成交条件：均价成交（B、W）、乱序上报（Z、U）、衍生定价（4）、
参考价（P）、官方开收盘价（M、Q、9）、非常规交割（C、N、R）、有条件成交（V、7）。
盘前盘后的 Form T 成交（T）保留——这套策略主要就在盘前。
"""
from __future__ import annotations

import pandas as pd

EXCLUDE = set("BWZU4PMQ9CNRV7")


def clean_trades(tr: pd.DataFrame) -> pd.DataFrame:
    if tr.empty:
        return tr
    cond = tr["conditions"].fillna("").astype(str)
    bad = cond.apply(lambda c: any(x in EXCLUDE for x in c.split(",") if x))
    return tr[~bad].sort_values(["ts", "id"], kind="stable")


def first_touch(prices, up: float, down: float) -> str | None:
    """返回先触及的一边；两边都没触及返回 None。"""
    for p in prices:
        if p >= up:
            return "up"
        if p <= down:
            return "down"
    return None


def resolve(events: pd.DataFrame, trades_by_key: dict) -> pd.DataFrame:
    """events: symbol, minute_utc, up, down；trades_by_key[(symbol, minute_utc)] = 该分钟逐笔。"""
    out = []
    for e in events.itertuples(index=False):
        tr = trades_by_key.get((e.symbol, e.minute_utc))
        if tr is None or tr.empty:
            out.append((e.symbol, e.minute_utc, e.up, e.down, None, 0))
            continue
        tr = clean_trades(tr)
        out.append((e.symbol, e.minute_utc, e.up, e.down,
                    first_touch(tr["price"].tolist(), e.up, e.down), len(tr)))
    return pd.DataFrame(out, columns=["symbol", "minute_utc", "up", "down", "first", "n_ticks"])


def to_resolver(res: pd.DataFrame) -> dict:
    r = res.dropna(subset=["first"])
    return {(s, m, round(u, 4), round(d, 4)): f
            for s, m, u, d, f in zip(r["symbol"], r["minute_utc"], r["up"], r["down"], r["first"])}

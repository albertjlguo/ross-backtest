"""
合成数据：只用来验证代码能跑通、逻辑对不对，不代表任何真实市场统计。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

ET = "America/New_York"


def _bars_from_closes(opens, closes, wick, rng):
    o = np.asarray(opens); c = np.asarray(closes)
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, wick, len(c))))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, wick, len(c))))
    return o, hi, lo, c


def generate_synthetic(n_days: int = 60, n_symbols: int = 25, runner_prob: float = 0.3,
                       seed: int = 7, start: str = "2026-03-02"):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start, periods=n_days)
    minutes = np.arange(4 * 60, 11 * 60 + 1)          # 04:00 – 11:00
    nmin = len(minutes)
    bars, daily, news = [], [], []

    for d in dates:
        for s in range(n_symbols):
            sym = f"SYN{s:02d}"
            p0 = float(rng.uniform(1.5, 22))
            adv = float(np.exp(rng.normal(np.log(1.2e6), 0.6)))
            flt = float(rng.uniform(1e6, 60e6))
            base_v = adv / 960
            ret = rng.normal(0, 0.0015, nmin)
            vol = base_v * np.exp(rng.normal(0, 0.8, nmin))
            runner = rng.random() < runner_prob
            if runner:
                k0 = int(rng.integers(150, 345))           # 06:30 – 09:45
                news.append({"symbol": sym, "ts": d + pd.Timedelta(minutes=int(minutes[k0]) - 1)})
                k = k0
                gap = rng.uniform(0.08, 0.40)
                ret[k] = gap
                vol[k] = adv * rng.uniform(0.5, 2.0)
                k += 1
                for _ in range(int(rng.integers(2, 6))):
                    for _ in range(int(rng.integers(2, 6))):          # 拉升段
                        if k >= nmin: break
                        ret[k] = rng.uniform(0.005, 0.04)
                        vol[k] = adv * rng.uniform(0.2, 1.2)
                        k += 1
                    for _ in range(int(rng.integers(1, 5))):          # 回调段
                        if k >= nmin: break
                        ret[k] = -rng.uniform(0.003, 0.02)
                        vol[k] = adv * rng.uniform(0.05, 0.4)
                        k += 1
                drift = rng.uniform(-0.004, 0.001)                     # 之后多半回落
                while k < nmin:
                    ret[k] = drift + rng.normal(0, 0.006)
                    vol[k] = adv * rng.uniform(0.01, 0.15)
                    k += 1
            elif rng.random() < 0.1:
                news.append({"symbol": sym, "ts": d + pd.Timedelta(hours=7)})

            closes = p0 * np.cumprod(1 + ret)
            opens = np.concatenate([[p0], closes[:-1]]) * (1 + rng.normal(0, 0.0005, nmin))
            o, h, l, c = _bars_from_closes(opens, closes, 0.003 if runner else 0.001, rng)
            ts = (d + pd.to_timedelta(minutes, unit="m")).tz_localize(ET)
            bars.append(pd.DataFrame({"symbol": sym, "ts": ts, "open": o, "high": h, "low": l,
                                      "close": c, "volume": np.round(vol)}))
            daily.append({"date": d.date(), "symbol": sym, "prev_close": p0, "adv30": adv,
                          "float_shares": flt})

    bars = pd.concat(bars, ignore_index=True)
    news = pd.DataFrame(news)
    news["ts"] = pd.to_datetime(news["ts"]).dt.tz_localize(ET)
    return bars, pd.DataFrame(daily), news

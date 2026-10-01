"""
单只股票单日的逐 K 线模拟（核心状态机）。

成交假设：
- 入场：前一根收盘时挂"前一根高点 + 1 tick"的买入止损单；本根最高价触及即成交，
  开盘就高于触发价则按开盘价成交。再加滑点。
- 止损：开盘已低于止损 → 按开盘价成交（跳空）；否则最低价触及 → 按止损价成交。
- 同一根 K 线里的先后顺序（intrabar_path）：
    "ohlc"       阳线 开→低→高→收，阴线 开→高→低→收（业内常用近似）
    "worst_case" 止损永远先于目标；入场那根最低价破止损即视为入场后被打掉
- "ohlc" 下，阳线且开盘低于触发价时，最低点发生在突破之前：
  若这个最低点已经跌破回调低点，说明形态在触发前就坏了，撤单不入场。
- 止损上移（保本 / 加仓后上移）从下一根开始生效。
- 离场信号在收盘时判定，下一根开盘价离场。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
import json
import math

import pandas as pd

from .config import Config
from .features import hhmm


@dataclass
class _Pos:
    entry_idx: int
    episode: int
    trigger: float
    stop: float
    stop_init: float
    t1: float
    risk_ps: float
    init_shares: int
    shares: int
    avg_cost_raw: float
    entry_fill: float
    pct_at_entry: float
    rvol_at_entry: float
    cash: float = 0.0
    shares_max: int = 0
    t1_done: bool = False
    extras_done: int = 0
    be_pending: bool = False
    stop_kind: str = "stop"
    adds: int = 0
    last_add_episode: int = -1
    pending: str | None = None
    hi: float = -math.inf
    lo: float = math.inf
    fills: list = field(default_factory=list)


def _slip(px: float, cfg: Config) -> float:
    return max(cfg.slippage_min, px * cfg.slippage_bps / 1e4)


def _comm(sh: int, cfg: Config) -> float:
    return max(cfg.commission_min, cfg.commission_per_share * sh) if sh > 0 else 0.0


def simulate_symbol_day(sym: str, date, f: pd.DataFrame, cfg: Config,
                        float_shares: float, counters: Counter,
                        resolver: dict | None = None, amb_log: list | None = None) -> list[dict]:
    """
    resolver: {(symbol, 分钟起点UTC, 上方价位, 下方价位): "up" | "down"}——逐笔成交给出的
              "同一分钟里两个价位谁先被触及"。没有的就退回 intrabar_path 假设。
    amb_log : 传入列表时，把所有这种歧义事件记下来，供下载逐笔用。
    """
    m = f["minute"].tolist()
    ts = f["ts"].tolist()
    o = f["open"].tolist(); h = f["high"].tolist()
    l = f["low"].tolist(); c = f["close"].tolist(); v = f["volume"].tolist()
    vwap = f["vwap"].tolist(); hod = f["hod"].tolist()
    qual = f["qual"].astype(bool).tolist()
    rvol = f["rvol"].tolist(); pct = f["pct"].tolist()
    vavg = f["vol_avg_prev"].tolist(); pvol = f["part_vol"].tolist()
    bratio = f["bid_ratio"].tolist()
    n = len(m)

    E0, E1, FL = hhmm(cfg.entry_start), hhmm(cfg.entry_end), hhmm(cfg.flatten_time)
    blk = [(hhmm(a), hhmm(b)) for a, b in cfg.entry_blackouts]
    aggressive = cfg.mode == "aggressive"
    worst = cfg.intrabar_path == "worst_case"

    def in_window(t):
        return E0 <= t < E1 and not any(a <= t < b for a, b in blk)

    def first_touch(i, up, down, kind):
        """同一根 K 线里 up 和 down 两个价位都被触及：逐笔回答谁先，没有逐笔返回 None。"""
        key = (sym, pd.Timestamp(ts[i]).tz_convert("UTC").isoformat(), round(up, 4), round(down, 4))
        if amb_log is not None:
            amb_log.append({"symbol": key[0], "minute_utc": key[1], "up": key[2], "down": key[3],
                            "kind": kind})
        ans = resolver.get(key) if resolver else None
        counters[f"ambiguous_{kind}"] += 1
        if ans in ("up", "down"):
            counters[f"ambiguous_{kind}_resolved"] += 1
            return ans
        return None

    trades: list[dict] = []
    pos: _Pos | None = None
    n_trades = 0

    # ---- 形态状态 ----
    run_hod = -math.inf
    hod_val = math.nan
    impulse_low = math.nan
    last_hod_idx = -10
    push_vols: list[float] = []      # 每一轮冲高（连续创新高的 K 线）的峰值成交量
    pb_active = False
    pb_bars = 0
    pb_low = math.inf
    pb_red = 0
    cur_episode = None
    episode_no = 0
    traded_episodes: set[int] = set()
    armed: dict | None = None

    # ---------------- 成交工具 ----------------
    def record(side, i, sh, px, why):
        pos.fills.append({"t": str(ts[i]), "side": side, "sh": sh, "px": round(px, 4), "why": why})

    def sell(i, sh, raw, why):
        px = max(0.0001, raw - _slip(raw, cfg))
        pos.cash += sh * px - _comm(sh, cfg)
        pos.shares -= sh
        record("S", i, sh, px, why)

    def close_all(i, raw, why):
        nonlocal pos
        if pos.shares > 0:
            sell(i, pos.shares, raw, why)
        trades.append(_trade_row(sym, date, pos, ts[i], why, float_shares, ts, m))
        pos = None

    def take_targets(i):
        if aggressive or pos is None:
            return
        if not pos.t1_done and h[i] >= pos.t1:
            q = min(pos.shares, max(1, int(round(pos.init_shares * cfg.t1_fraction))))
            sell(i, q, pos.t1, "t1")
            pos.t1_done = True
            if cfg.move_stop_to_be_after_t1:
                pos.be_pending = True
        if pos.t1_done:
            while pos.shares > 0 and pos.extras_done < len(cfg.extra_targets):
                r, frac = cfg.extra_targets[pos.extras_done]
                lvl = pos.trigger + r * pos.risk_ps
                if h[i] < lvl:
                    break
                q = min(pos.shares, max(1, int(round(pos.init_shares * frac))))
                sell(i, q, lvl, f"t{pos.extras_done + 2}")
                pos.extras_done += 1
        if pos.shares <= 0:
            close_all(i, c[i], "targets_done")

    def stop_check(i, why):
        if pos is not None and l[i] <= pos.stop:
            close_all(i, pos.stop, why)

    def next_target():
        if aggressive or pos is None:
            return None
        if not pos.t1_done:
            return pos.t1
        if pos.extras_done < len(cfg.extra_targets):
            return pos.trigger + cfg.extra_targets[pos.extras_done][0] * pos.risk_ps
        return None

    def bar_exits(i, fresh: bool, at_open: bool, entry_first: bool = False):
        """本根 K 线内的止损 / 目标，按逐笔结论或 intrabar_path 决定先后。"""
        green = c[i] >= o[i]
        if fresh:
            if entry_first:                  # 逐笔：先到触发价，之后才跌破止损
                take_targets(i)
                stop_check(i, "stop_same_bar")
            elif worst:
                stop_check(i, "stop_same_bar")
                take_targets(i)
            elif green:                      # 开→低→高：开盘成交才会经历低点
                if at_open:
                    stop_check(i, "stop_same_bar")
                take_targets(i)
            else:                            # 开→高→低
                take_targets(i)
                stop_check(i, "stop_same_bar")
            return
        if o[i] <= pos.stop:
            close_all(i, o[i], pos.stop_kind + "_gap")
            return
        nxt = next_target()
        if nxt is not None and l[i] <= pos.stop and h[i] >= nxt and o[i] < nxt:
            ft = first_touch(i, nxt, pos.stop, "position")
            if ft is not None:
                stop_first = ft == "down"
            else:
                stop_first = worst or green
        else:
            stop_first = worst or green
        if stop_first:
            stop_check(i, pos.stop_kind)
            take_targets(i)
        else:
            take_targets(i)
            stop_check(i, pos.stop_kind)

    def signals(i) -> str | None:
        hi_ref = hod[i]
        near = h[i] >= hi_ref * (1 - cfg.near_hod_pct)
        avg = vavg[i]
        has_avg = avg is not None and not math.isnan(avg) and avg > 0
        rng = h[i] - l[i]
        if (not aggressive and cfg.exit_on_first_red_before_t1
                and not pos.t1_done and c[i] < o[i]):
            return "first_red_candle"
        if cfg.sig_stall and has_avg and near and v[i] >= cfg.stall_vol_mult * avg and c[i] <= o[i]:
            return "stall_heavy_volume"
        if cfg.sig_bid_pressure:
            br = bratio[i]
            if br is not None and not math.isnan(br):
                if br >= cfg.bid_ratio_threshold and has_avg and v[i] >= avg:
                    return "sell_at_bid"
            elif rng > 0 and has_avg and (c[i] - l[i]) / rng <= cfg.clv_threshold \
                    and v[i] >= cfg.pressure_vol_mult * avg:
                return "sell_pressure_proxy"
        if cfg.sig_topping_tail and rng > 0 and rng / c[i] >= cfg.min_tail_range_pct and near \
                and (h[i] - max(o[i], c[i])) / rng >= cfg.wick_ratio:
            return "topping_tail"
        if cfg.sig_rejections:
            j0 = max(pos.entry_idx, i - cfg.rejection_window + 1)
            k = sum(1 for j in range(j0, i + 1)
                    if h[j] >= hi_ref * (1 - cfg.near_hod_pct) and c[j] < (h[j] + l[j]) / 2)
            if k >= cfg.rejection_count:
                return "repeated_rejection"
        if (cfg.sig_volume_divergence and i > pos.entry_idx and last_hod_idx == i
                and len(push_vols) >= 2 and push_vols[-1] < cfg.divergence_ratio * push_vols[-2]):
            return "volume_divergence"
        return None

    # ---------------- 主循环 ----------------
    for i in range(n):
        t = m[i]
        if pos is None and t >= FL:
            break

        # 1) 已有持仓：挂起离场 / 清仓时间 / 止损 / 目标
        if pos is not None:
            if pos.pending:
                close_all(i, o[i], pos.pending)
            elif t >= FL:
                close_all(i, o[i], "flatten_time")
            else:
                pos.hi = max(pos.hi, h[i]); pos.lo = min(pos.lo, l[i])
                bar_exits(i, fresh=False, at_open=False)

        # 2) 入场 / 加仓（上一根收盘时挂好的单）
        if armed is not None and in_window(t) and h[i] >= armed["trigger"]:
            ep = armed["episode"]
            at_open = o[i] >= armed["trigger"]
            touches_stop = l[i] <= armed["stop"]
            entry_first = False
            if worst or at_open or not touches_stop:
                broken_first = False
            elif o[i] <= armed["stop"]:          # 开盘就在止损下方，先破位再拉上来
                broken_first = True
            else:
                broken_first = c[i] >= o[i]      # 默认：阳线先低后高 → 破位在先
                eligible = (pos is None and 1 <= ep <= cfg.max_pullback_number
                            and ep not in traded_episodes
                            and n_trades < cfg.max_trades_per_symbol_day) or \
                           (pos is not None and aggressive)
                if eligible:
                    ft = first_touch(i, armed["trigger"], armed["stop"], "entry")
                    if ft is not None:
                        broken_first = ft == "down"
                        entry_first = ft == "up"
            if pos is None:
                if not (1 <= ep <= cfg.max_pullback_number):
                    counters["blocked_pullback_number"] += 1
                elif ep in traded_episodes:
                    pass
                elif n_trades >= cfg.max_trades_per_symbol_day:
                    counters["blocked_max_trades"] += 1
                elif broken_first:
                    counters["setup_broken_before_trigger"] += 1
                else:
                    risk = armed["risk"]
                    sh = int(math.floor(cfg.risk_per_trade / risk + 1e-9))
                    sh = min(sh, int(cfg.max_notional // armed["trigger"]),
                             int(cfg.max_participation * pvol[i - 1]))
                    if sh < 1:
                        counters["skipped_size_zero"] += 1
                    else:
                        raw = o[i] if at_open else armed["trigger"]
                        fill = raw + _slip(raw, cfg)
                        pos = _Pos(entry_idx=i, episode=ep, trigger=armed["trigger"],
                                   stop=armed["stop"], stop_init=armed["stop"], t1=armed["t1"],
                                   risk_ps=risk, init_shares=sh, shares=sh, shares_max=sh,
                                   avg_cost_raw=raw, entry_fill=fill,
                                   pct_at_entry=pct[i - 1], rvol_at_entry=rvol[i - 1])
                        pos.cash -= sh * fill + _comm(sh, cfg)
                        record("B", i, sh, fill, "entry")
                        n_trades += 1
                        traded_episodes.add(ep)
                        counters["entries"] += 1
                        pos.hi, pos.lo = h[i], (l[i] if (worst or at_open or entry_first or c[i] < o[i])
                                                else raw)
                        bar_exits(i, fresh=True, at_open=at_open, entry_first=entry_first)
            elif (aggressive and ep >= 1 and ep != pos.episode and ep != pos.last_add_episode
                  and pos.adds < cfg.max_adds and rvol[i - 1] >= cfg.add_min_rvol
                  and armed["stop"] >= pos.avg_cost_raw and not broken_first):
                add = max(1, int(round(pos.init_shares * cfg.add_fraction)))
                add = min(add, int(cfg.max_participation * pvol[i - 1]))
                if add >= 1:
                    raw = o[i] if at_open else armed["trigger"]
                    fill = raw + _slip(raw, cfg)
                    pos.avg_cost_raw = (pos.avg_cost_raw * pos.shares + raw * add) / (pos.shares + add)
                    pos.shares += add
                    pos.shares_max = max(pos.shares_max, pos.shares)
                    pos.cash -= add * fill + _comm(add, cfg)
                    record("B", i, add, fill, f"add_{pos.adds + 1}")
                    pos.adds += 1
                    pos.last_add_episode = ep
                    pos.stop = max(pos.stop, armed["stop"])
                    pos.stop_kind = "raised_stop"
                    counters["adds"] += 1
                    if worst or at_open or entry_first or c[i] < o[i]:
                        stop_check(i, "raised_stop_same_bar")

        # 3) 用本根 K 线更新形态（收盘时判定，挂下一根的单）
        if h[i] > run_hod:
            if last_hod_idx == i - 1 and push_vols:
                push_vols[-1] = max(push_vols[-1], v[i])
            else:
                push_vols.append(v[i])
            last_hod_idx = i
            hod_val = h[i]
            j0 = max(0, i - cfg.impulse_lookback + 1)
            impulse_low = min(l[j0:i + 1])
            pb_active = False
            cur_episode = None
            armed = None
        elif last_hod_idx >= 0:
            if not pb_active:
                pb_active, pb_bars, pb_low, pb_red, cur_episode = True, 0, math.inf, 0, None
            pb_bars += 1
            pb_low = min(pb_low, l[i])
            pb_red += c[i] < o[i]
            armed = None
            ok = (cfg.min_pullback_bars <= pb_bars <= cfg.max_pullback_bars
                  and pb_red >= cfg.min_red_bars and qual[i]
                  and impulse_low > 0 and hod_val > impulse_low
                  and hod_val / impulse_low - 1 >= cfg.min_impulse_pct
                  and (hod_val - pb_low) / (hod_val - impulse_low) <= cfg.max_retrace)
            if ok and cfg.require_above_vwap:
                ok = not math.isnan(vwap[i]) and pb_low >= vwap[i]
            if ok:
                trigger = round(h[i] + cfg.tick, 4)
                stop = pb_low
                risk = round(trigger - stop, 6)
                ok = risk >= cfg.min_risk_per_share and risk / trigger <= cfg.max_risk_pct
                if ok:
                    t1 = trigger + cfg.t1_r * risk if cfg.target_mode == "r_multiple" else hod_val
                    if not aggressive and (t1 - trigger) / risk < cfg.min_rr:
                        ok = False
                        counters["rejected_rr"] += 1
            if ok:
                countable = cfg.count_pullbacks_from == "qualified" or t >= E0 - 1
                if cur_episode in (None, 0):
                    if countable:
                        episode_no += 1
                        cur_episode = episode_no
                        counters["episodes_armed"] += 1
                    else:
                        cur_episode = 0
                armed = {"trigger": trigger, "stop": stop, "risk": risk, "t1": t1,
                         "episode": cur_episode}
        run_hod = max(run_hod, h[i])

        # 4) 收盘：保本止损生效（下一根起）、离场信号
        if pos is not None:
            if pos.be_pending:
                pos.stop = max(pos.stop, pos.avg_cost_raw)
                pos.stop_kind = "breakeven_stop"
                pos.be_pending = False
            if pos.pending is None:
                pos.pending = signals(i)

    if pos is not None:  # 数据提前结束
        last = min(i, n - 1)
        close_all(last, c[last], "end_of_data")
    return trades


def _trade_row(sym, date, p: _Pos, exit_ts, why, float_shares, ts, m) -> dict:
    risk_dollars = p.init_shares * p.risk_ps
    return {
        "date": date, "symbol": sym, "episode": p.episode,
        "entry_time": ts[p.entry_idx], "entry_minute": m[p.entry_idx],
        "trigger": p.trigger, "entry_fill": round(p.entry_fill, 4),
        "stop_init": p.stop_init, "t1": round(p.t1, 4), "risk_ps": round(p.risk_ps, 4),
        "shares_init": p.init_shares, "shares_max": p.shares_max, "adds": p.adds,
        "exit_time": exit_ts, "exit_reason": why, "t1_hit": p.t1_done,
        "pnl": round(p.cash, 2), "r_multiple": round(p.cash / risk_dollars, 4),
        "mfe_r": round((p.hi - p.trigger) / p.risk_ps, 3),
        "mae_r": round((p.trigger - p.lo) / p.risk_ps, 3),
        "pct_at_entry": p.pct_at_entry, "rvol_at_entry": p.rvol_at_entry,
        "float_shares": float_shares,
        "fills": json.dumps(p.fills, ensure_ascii=False),
    }

"""
回测参数。每个字段都对应研究里核对过的一条规则，注释写明出处或取舍。

时间一律是美东时间 (America/New_York)，字符串 "HH:MM"。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict, replace
from typing import Optional, Tuple
import json


@dataclass(frozen=True)
class Config:
    # ------------------------------------------------------------------ #
    # 一、选股五条件（每根 1 分钟 K 线收盘时按当时已知数据判定，无未来函数）
    # ------------------------------------------------------------------ #
    min_pct_change: float = 0.10       # 较昨收涨幅 ≥10%
    min_rvol: float = 5.0              # 当日累计成交量 / 前 30 日日均量 ≥5（Ross 新版选股说明用 30 日）
    min_price: float = 2.0             # 现价下限（书：$2–$10 最佳，上限 $20）
    max_price: float = 20.0
    max_float: float = 20e6            # 流通股 ≤2000 万（书："preferably 20M or fewer"）
    require_catalyst: bool = True      # 需要新闻；没提供新闻数据时会自动关闭并在结果里标注
    news_lookback_hours: float = 24.0  # 新闻时间在 [t - 24h, t] 内算"当日催化"
    top_n_gainers: Optional[int] = None  # Ross 只看涨幅榜前 2-3 名；None = 不限
    missing_float: str = "exclude"     # 流通股缺失时："exclude" 剔除 / "include" 放行

    # ------------------------------------------------------------------ #
    # 二、交易时段
    # ------------------------------------------------------------------ #
    session_start: str = "04:00"       # 盘前起点，累计量 / VWAP / 当日高点从这里算
    entry_start: str = "07:00"
    entry_end: str = "10:00"           # 只在 [07:00, 10:00) 开新仓
    flatten_time: str = "10:30"        # 之后强制清仓（规则没写，给持仓留余地；可改）
    entry_blackouts: Tuple[Tuple[str, str], ...] = ()  # 例：(("09:30", "09:35"),) 避开开盘拥挤
    vwap_anchor: str = "04:00"

    # ------------------------------------------------------------------ #
    # 三、买点：冲高 → 回调 → 第一根突破前一根高点的 K 线
    # ------------------------------------------------------------------ #
    impulse_lookback: int = 5          # 冲高段：创日内新高那根往前 5 根内的最低点算起
    min_impulse_pct: float = 0.03      # 冲高幅度 ≥3% 才算一段有效拉升
    min_pullback_bars: int = 1         # 回调 K 线数（Ross 微回调 1-3 根，牛旗 2-3 根）
    max_pullback_bars: int = 5         # 超过就视为走弱，等下一次新高
    min_red_bars: int = 1              # 回调中至少 1 根阴线
    max_retrace: float = 0.5           # 回撤不超过拉升段的 50%（Ross 牛旗规则）
    require_above_vwap: bool = True    # 回调低点不破 VWAP（Ross 牛旗规则）
    max_pullback_number: int = 1       # 1 = 只做第一次回调；2 = 第一或第二次
    count_pullbacks_from: str = "qualified"  # 回调计数从何时起："qualified" 入选后 / "window" 进入时段后
    max_trades_per_symbol_day: int = 2
    tick: float = 0.01
    min_risk_per_share: float = 0.02   # 止损距离太小（≈价差）不做
    max_risk_pct: float = 0.08         # 止损距离超过入场价 8% 不做

    # ------------------------------------------------------------------ #
    # 四、卖出
    # ------------------------------------------------------------------ #
    mode: str = "conservative"         # "conservative" 先卖一半再边涨边卖 / "aggressive" 持有+加仓
    target_mode: str = "r_multiple"    # "r_multiple" 第一目标=入场+2R（Ross 选股说明）
                                       # "hod" 第一目标=回调前的当日高点（你的原文版本）
    t1_r: float = 2.0
    min_rr: float = 1.0                # 第一目标的盈亏比下限（hod 模式下才会真正起过滤作用）
    t1_fraction: float = 0.5           # 第一目标卖一半
    extra_targets: Tuple[Tuple[float, float], ...] = ((3.0, 0.25),)  # "边涨边卖"：+3R 再卖初始仓位的 25%
    move_stop_to_be_after_t1: bool = True   # 卖一半后剩余止损上移到成本（Ross）
    exit_on_first_red_before_t1: bool = True  # 未到第一目标前，第一根收阴 K 线即离场（Ross Exit #2）
    # 激进模式：不在第一目标卖，出现新一轮回调突破且行情极热时加仓
    max_adds: int = 2
    add_fraction: float = 0.5          # 每次加初始仓位的 50%
    add_min_rvol: float = 10.0         # "极热"：相对量 ≥10 倍
    # 加仓前提：新回调低点 ≥ 持仓均价，整笔止损上移到新回调低点 → 加仓不增加亏损风险

    # ------------------------------------------------------------------ #
    # 五、离场信号（K 线代理；有逐笔/盘口聚合列时自动用真实数据）
    # ------------------------------------------------------------------ #
    sig_stall: bool = True             # ①② 大卖单 / 隐藏卖家 → 高点附近放巨量但收阴（滞涨）
    sig_bid_pressure: bool = True      # ③ 按买一价成交激增 → 有 vol_at_bid/vol_at_ask 列用真实比例，否则用收盘位置代理
    sig_topping_tail: bool = True      # ④ 长上影
    sig_rejections: bool = True        # ④ 高位反复受阻
    sig_volume_divergence: bool = True  # ⑤ 价创新高但量能减弱
    vol_avg_window: int = 20
    stall_vol_mult: float = 3.0
    near_hod_pct: float = 0.01
    bid_ratio_threshold: float = 0.65
    clv_threshold: float = 0.25
    pressure_vol_mult: float = 1.5
    wick_ratio: float = 0.5
    min_tail_range_pct: float = 0.02
    rejection_window: int = 10
    rejection_count: int = 2
    divergence_ratio: float = 0.5      # 本轮冲高的峰值量 < 上一轮冲高峰值量 × 0.5

    # ------------------------------------------------------------------ #
    # K 线内部路径假设（只有 OHLC 时无法知道高低点先后）
    # ------------------------------------------------------------------ #
    intrabar_path: str = "ohlc"        # "ohlc": 阳线按 开→低→高→收、阴线按 开→高→低→收
                                       # "worst_case": 任何同根冲突都按最坏结果（稳健性检验用）

    # ------------------------------------------------------------------ #
    # 仓位与成本
    # ------------------------------------------------------------------ #
    risk_per_trade: float = 100.0      # 每笔计划亏损（1R）= $100
    max_notional: float = 50_000.0
    max_participation: float = 0.10    # 股数 ≤ 最近 5 根 K 线平均量的 10%
    participation_window: int = 5
    slippage_bps: float = 10.0         # 每边滑点 = max(0.01, 10bps × 价格)
    slippage_min: float = 0.01
    commission_per_share: float = 0.0035
    commission_min: float = 0.35

    # ------------------------------------------------------------------ #
    # 组合层
    # ------------------------------------------------------------------ #
    max_concurrent_positions: int = 1
    daily_max_loss: Optional[float] = 300.0  # 当日已实现亏损达 $300 停手；None = 不限

    # ---------------------------------------------------------------- #
    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_dict(d: dict) -> "Config":
        d = dict(d)
        for k in ("entry_blackouts", "extra_targets"):
            if k in d and d[k] is not None:
                d[k] = tuple(tuple(x) for x in d[k])
        return Config(**d)

    def with_(self, **kw) -> "Config":
        return replace(self, **kw)


PRESETS = {
    # Ross 自己的资料：2R 第一目标、卖一半、止损上移保本
    "ross": Config(),
    # 你的原文版本：第一目标 = 当日高点附近，盈亏比至少 1:1
    "summary": Config(target_mode="hod", min_rr=1.0),
    # 激进：持有 + 极热时加仓
    "aggressive": Config(mode="aggressive"),
}

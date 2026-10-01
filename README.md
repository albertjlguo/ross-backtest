# Ross Cameron 短线规则回测

把五条规则写成逐分钟、无未来函数的事件驱动回测。纯 Python（pandas + numpy）；真实数据来自 Alpaca 免费账户和 SEC 免费接口。

## 在 Replit 上跑真实数据

1. 在 Replit 里 **Import from GitHub**，选择这个仓库。
2. 在 **Secrets** 里加三项：

   | 名称 | 值 |
   |---|---|
   | `APCA_API_KEY_ID` | Alpaca key（模拟盘账户的就行） |
   | `APCA_API_SECRET_KEY` | Alpaca secret |
   | `SEC_USER_AGENT` | 你的名字和邮箱，例如 `Your Name you@example.com`（SEC 要求） |

3. 点 **Run**，会先装依赖，再跑 `python pipeline.py smoke`：最近约 3 周的小样本，几分钟就能跑完，用来确认 key、接口和数据都正常。
4. 冒烟测试没问题后，在 Shell 里跑全量：

   ```bash
   python pipeline.py all --push          # 默认 2024-10-01 至昨天；跑完把 results/ 推回 GitHub
   python pipeline.py all --start 2023-01-01 --push   # 拉长区间
   ```

   全量预计 1–2 小时（下载约 40 分钟，受 Alpaca 免费额度每分钟 200 次限制；10 组回测约 45 分钟）。
   每一步都会落盘，中断后重跑会自动跳过已完成的部分；改了预筛规则只会补下新增的股票。
   `--push` 需要先在 Replit 的 Git 面板里连好 GitHub 账户。

### 流水线步骤

| 步骤 | 做什么 | 产出 |
|---|---|---|
| `universe` | Alpaca 资产列表，含已停止交易的股票；去掉权证、单位、优先股 | `data/universe.parquet` |
| `daily` | 全市场日线，原始价和拆股调整价各一份 | `data/daily/` |
| `candidates` | 用日线预筛，只用必要条件：全天量 ≥ 0.9×5×ADV30、昨收 ≤ $18.18。不用日线最高价，因为 Alpaca 日线最高价不含盘前 | `data/candidates.parquet` |
| `minute` | 候选股票当天 04:00–12:00 的 1 分钟线（全市场合并数据），按日期批量请求 | `data/minute/日期.parquet` |
| `news` | 候选股票前一天 04:00 到当天 12:00 的新闻 | `data/news/` |
| `sec` | SEC 财报封面上的已发行股数，按申报日期对齐 | `data/float.parquet`、`results/sec_coverage.json` |
| `dq` | 数据质量检查，见下 | `results/data_quality.json` |
| `backtest` | 跑 10 组配置 | `results/<配置名>/`、`results/overview.csv` |

### 10 组配置

- 三套主预设：`ross`、`summary`、`aggressive`。
- 七组稳健性检验，各自只改一项：
  - 最坏成交假设
  - 滑点 30bps
  - 关掉"价涨量缩"信号
  - 只做涨幅榜前 3
  - 允许第二次回调
  - 回调从 07:00 起计数（而不是从入选那一刻起）
  - 不要求新闻

**解读结果前先看 `results/data_quality.json`**，它回答以下问题：
- **日线成交量是否包含盘前盘后？**
  - 如果不包含，相对量会被系统性高估。
- **日线最高价是否包含盘前高点？**
  - 如果不包含，预筛会漏掉只在盘前拉升又回落的票。这类票恰好多是亏损交易，漏掉会让结果偏乐观。
- **已停止交易的股票有没有数据？**
  - 判断幸存者偏差有多大。

## 本地 / 合成数据

```bash
pip install -r requirements.txt
python run_backtest.py --demo         # 合成数据跑通
python -m pytest -q tests             # 37 个测试：手算场景、无未来函数、模拟接口的流水线
```

> 合成数据只用来验证代码逻辑。它的拉升段是人为造的，所以 demo 的胜率没有任何参考意义。

## 规则 → 代码

| 规则 | 实现 | 参数 |
|---|---|---|
| 涨幅 ≥10% | 当根收盘价 / 昨收 − 1 | `min_pct_change` |
| 相对量 ≥5 倍 | 04:00 起累计量 / 前 30 日均量 | `min_rvol` |
| 新闻催化 | 新闻时间 ∈ [t−24h, t] | `require_catalyst` `news_lookback_hours` |
| 价格 $2–20 | 当根收盘价 | `min_price` `max_price` |
| 流通股 ≤2000 万 | 日表 `float_shares` | `max_float` |
| 涨幅榜前 N | 每分钟在满足其余条件的股票里排名 | `top_n_gainers` |
| 07:00–10:00 | 只在时段内开新仓；10:30 强平 | `entry_start` `entry_end` `flatten_time` |
| 避开拥挤时段 | 可设禁入区间，默认不设 | `entry_blackouts` |
| 第一次回调 | 冲高创日内新高 → 1–5 根回调（≥1 根阴线、回撤 ≤50%、不破 VWAP） | `max_pullback_number` 等 |
| 突破前一根高点 | 前一根收盘挂"前高 + 1 tick"买入止损单 | `tick` |
| 止损 = 回调低点 | 回调期间最低价 | `min_risk_per_share` `max_risk_pct` |
| 盈亏比 ≥1:1 | 第一目标 / 风险 ≥ `min_rr` | `min_rr` |
| 先卖一半，边涨边卖 | T1 卖 50%，止损上移保本，+3R 再卖 25% | `t1_r` `t1_fraction` `extra_targets` |
| 不下移止损、不摊平 | 止损只上移；加仓只在激进模式、且新止损 ≥ 均价时 | — |
| 激进：持有 + 加仓 | 新一轮回调突破、相对量 ≥10 倍时加 50% | `mode="aggressive"` |
| ①② 大卖单 / 隐藏卖家 | 代理：高点附近放量 ≥3 倍但收阴（滞涨） | `sig_stall` |
| ③ 按买一价成交激增 | 有 `vol_at_bid`/`vol_at_ask` 列时用真实比例 ≥65%；否则用"收在低位 + 放量"代理 | `sig_bid_pressure` |
| ④ 长上影 / 反复受阻 | 上影 ≥ K 线长度 50%；或入场后 10 根内 ≥2 次在高点收在下半部 | `sig_topping_tail` `sig_rejections` |
| ⑤ 价涨量缩 | 本轮冲高峰值量 < 上一轮的 50% | `sig_volume_divergence` |
| Ross：T1 前第一根阴线离场 | 收盘判定，下一根开盘出 | `exit_on_first_red_before_t1` |

### 三套预设

`--preset ross`（默认）
: 第一目标 2R，Ross 选股说明的版本。

`--preset summary`
: 第一目标是回调前的日内高点，盈亏比至少 1:1，即你原文的版本。
: 这条规则会过滤掉大量微回调，因为触发价离高点太近，到高点的空间不够 1R。

`--preset aggressive`
: 持有，并在行情极热时加仓。

## 数据格式

| 表 | 必需列 | 说明 |
|---|---|---|
| 分钟线 | `symbol, ts, open, high, low, close, volume` | 必须从 04:00 ET 开始，含盘前。可选列 `vol_at_bid, vol_at_ask`。 |
| 日表 | `date, symbol, prev_close, adv30, float_shares` | 必须是当时可知的值。可用 `prepare_daily()` 从日线生成。 |
| 新闻 | `symbol, ts` | 可选。没有时也可在日表加 `has_catalyst` 布尔列。 |

- 时间戳带时区就自动转成美东；不带时区则按美东解释。
- 日线的成交量口径要和分钟线一致，都含盘前盘后。否则相对量会被高估。
- 用 `candidate_days()` 先按日线预筛，只下载可能入选的 (日期, 股票)。

**真实数据运行示例：**

```bash
python run_backtest.py --bars bars.parquet --daily daily.csv --news news.csv \
  --preset ross --set top_n_gainers=3 --set intrabar_path=worst_case --out results/
```

**输出文件：**
- 交易明细：`trades.csv`，含每笔成交的 fills JSON
- 汇总：`summary.json`
- 分组统计：按时段、价格、流通股、离场原因、回调序号拆分的 `by_*.csv`
- 每日权益：`equity_daily.csv`

## 成交假设

- **入场**：止损单触及即按触发价成交；开盘就高于触发价时按开盘价成交。
- **跳空止损**：开盘已低于止损价时按开盘价成交。
- **同一根 K 线内的先后**（`intrabar_path`）：
  - `"ohlc"`（默认）：阳线按 开→低→高→收，阴线按 开→高→低→收。
  - `"worst_case"`：止损永远先于目标。
  - 跑完两种都看一遍。合成数据上两者差距很大，说明结论对这个假设敏感。
- **形态在触发前就破了**：阳线、开盘低于触发价、但低点先跌破回调低点时，视为形态在突破前已失效，撤单不入场。
- **止损上移**：保本、加仓后的上移都从下一根 K 线开始生效。
- **离场信号**：收盘时判定，下一根开盘价离场。
- **成本**：
  - 滑点每边 max($0.01, 10bps)。
  - 佣金 $0.0035/股，最低 $0.35。
  - 小盘股实际价差常比这个宽，建议用 `--set slippage_bps=30` 做敏感性测试。
- **仓位**：每笔 1R = $100，同时受两个上限约束：名义金额 $5 万，以及最近 5 根 K 线平均量的 10%。

## 已知偏差：解读结果前必看

1. **盘口代理不等于盘口。** 规则①②（大卖单、隐藏卖家）需要 Level 2 数据，K 线只能近似。
   - 规则③在提供 `vol_at_bid`/`vol_at_ask` 列时用的是真实数据。这两列由逐笔成交和 NBBO 报价逐笔匹配后按分钟聚合得到。
2. **离场信号⑤可能过于敏感。** 新闻刚出时的第一根 K 线通常是全天量峰，之后任何一轮冲高都容易不到它的一半。
   - 建议单独开关对比：`--set sig_volume_divergence=false`。
3. **幸存者偏差。** 分钟线和日表必须包含已退市、已被停牌的股票。2025 年 9 月以来被 SEC 停牌的那批小盘股，恰好是这套规则最常选中的类型。
4. **流通股要用"当时"的值。** 用今天的流通股去回测 2023 年，会把后来增发稀释过的股票错误地判为"大流通"而剔除，反过来也一样。
5. **熔断停牌没有建模。** LULD 停牌期间分钟线是空的，恢复交易时的跳空由"开盘价成交"处理，但无法模拟停牌期间无法下单的情况。
6. **组合层规则是近似。** 被"同时持仓上限"或"单日亏损上限"拦下的交易，不会让这只股票后续的信号重新出现。
7. **固定止损距离太小时成本会吞掉利润。** demo 里 3 美分止损的一笔，−1R 实际变成了 −1.85R。可以调高 `min_risk_per_share` 观察结果的变化。

## 文件

```
rossbt/config.py     全部参数与预设
rossbt/data.py       加载、校验、prepare_daily、candidate_days
rossbt/features.py   累计量 / 相对量 / VWAP / 五条件 / 涨幅榜排名
rossbt/engine.py     单票单日状态机（入场、止损、目标、加仓、离场信号）
rossbt/backtest.py   逐日驱动 + 组合层规则
rossbt/report.py     汇总与分组统计
rossbt/synth.py      合成数据（仅用于测试）
run_backtest.py      命令行
tests/test_engine.py 手算场景 + 无未来函数截断测试
```

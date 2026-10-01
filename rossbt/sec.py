"""
用 SEC 免费接口近似"当时的流通股"。

数据：每份 10-K / 10-Q 封面上的 dei:EntityCommonStockSharesOutstanding（已发行普通股数），
按"申报日期 filed"做 as-of 合并——某个交易日只用到当天及以前已经公开的数字，没有未来函数。

已发行股数 ≥ 流通股（多算了内部人持股），所以用它过"流通股 ≤2000 万"是偏保守的：
会多剔除一些票，不会把大流通的票误放进来。

局限：
- SEC 的代码对照表只有当前代码，已退市、改过代码的公司大多查不到 → float 缺失
  （回测里 missing_float="include" 放行并单独统计，避免重新引入幸存者偏差）。
- 外国私人发行人（20-F/6-K）通常不填这个字段 → 同样缺失。

SEC 要求请求头带联系方式：在 Replit Secrets 里设 SEC_USER_AGENT，例如 "Your Name your@email.com"。
"""
from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd
import requests

from .alpaca import RateLimiter

log = logging.getLogger("rossbt.sec")

TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"


def norm_ticker(t: str) -> str:
    return str(t).upper().replace(".", "-").replace("/", "-")


class SecClient:
    def __init__(self, user_agent: str | None = None, per_second: float = 8.0,
                 session=None, cache_dir: str | Path | None = None, sleep=time.sleep):
        self.ua = user_agent or os.environ.get("SEC_USER_AGENT")
        if not self.ua:
            raise RuntimeError("缺少 SEC_USER_AGENT（SEC 要求请求头带姓名和邮箱）")
        self.s = session or requests.Session()
        self.rl = RateLimiter(int(per_second * 60))
        self.cache = Path(cache_dir) if cache_dir else None
        if self.cache:
            self.cache.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep

    def _get_json(self, url: str):
        for attempt in range(5):
            self.rl.wait()
            r = self.s.get(url, headers={"User-Agent": self.ua, "Accept-Encoding": "gzip"},
                           timeout=60)
            if r.status_code == 404:
                return None
            if r.status_code == 429 or r.status_code >= 500:
                self._sleep(min(60, 2 ** attempt + 1))
                continue
            r.raise_for_status()
            return r.json()
        raise RuntimeError(f"SEC 请求失败: {url}")

    def ticker_map(self) -> dict[str, int]:
        js = self._get_json(TICKERS_URL) or {}
        return {norm_ticker(v["ticker"]): int(v["cik_str"]) for v in js.values()}

    def company_facts(self, cik: int):
        if self.cache:
            p = self.cache / f"{cik:010d}.json"
            if p.exists():
                return json.loads(p.read_text())
        js = self._get_json(FACTS_URL.format(cik=cik))
        if self.cache:
            (self.cache / f"{cik:010d}.json").write_text(json.dumps(js) if js else "null")
        return js


def shares_from_facts(js) -> pd.DataFrame:
    """companyfacts JSON → (filed, end, shares)，同一申报日取报告期最晚的一条。"""
    try:
        items = js["facts"]["dei"]["EntityCommonStockSharesOutstanding"]["units"]["shares"]
    except (KeyError, TypeError):
        return pd.DataFrame(columns=["filed", "end", "shares"])
    df = pd.DataFrame(items)
    if df.empty or "filed" not in df:
        return pd.DataFrame(columns=["filed", "end", "shares"])
    df = df.rename(columns={"val": "shares"})
    df["filed"] = pd.to_datetime(df["filed"])
    df["end"] = pd.to_datetime(df["end"])
    df = df[df["shares"] > 0].sort_values(["filed", "end"])
    return df.groupby("filed", as_index=False).last()[["filed", "end", "shares"]]


def shares_table(client: SecClient, symbols: Iterable[str]) -> tuple[pd.DataFrame, dict]:
    """返回 (date, symbol, float_shares) 表——date 为申报日，供 prepare_daily 做 as-of 合并。"""
    tmap = client.ticker_map()
    rows, stats = [], {"symbols": 0, "no_cik": 0, "no_shares": 0, "ok": 0}
    for sym in sorted(set(symbols)):
        stats["symbols"] += 1
        cik = tmap.get(norm_ticker(sym))
        if cik is None:
            stats["no_cik"] += 1
            continue
        df = shares_from_facts(client.company_facts(cik))
        if df.empty:
            stats["no_shares"] += 1
            continue
        stats["ok"] += 1
        for r in df.itertuples():
            rows.append((r.filed, sym, float(r.shares)))
    out = pd.DataFrame(rows, columns=["date", "symbol", "float_shares"])
    log.info("SEC 股本覆盖: %s", stats)
    return out, stats

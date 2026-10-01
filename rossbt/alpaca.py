"""
Alpaca 行情客户端（免费账户即可）。

免费账户可以查询全市场合并数据（feed=sip），条件是查询结束时间早于 15 分钟前；
历史从 2016 年起；每分钟 200 次调用（这里默认限速 180 次，留余量）。

环境变量：
  APCA_API_KEY_ID / APCA_API_SECRET_KEY   在 Replit Secrets 里设置
  APCA_API_BASE_URL                       交易接口地址，默认模拟盘 https://paper-api.alpaca.markets
"""
from __future__ import annotations

import logging
import os
import time
from typing import Iterable

import pandas as pd
import requests

log = logging.getLogger("rossbt.alpaca")

DATA_URL = "https://data.alpaca.markets"


class RateLimiter:
    def __init__(self, per_minute: int):
        self.interval = 60.0 / per_minute
        self._next = 0.0

    def wait(self):
        now = time.monotonic()
        if now < self._next:
            time.sleep(self._next - now)
        self._next = max(now, self._next) + self.interval


class AlpacaError(RuntimeError):
    pass


class AlpacaClient:
    def __init__(self, key_id: str | None = None, secret: str | None = None,
                 per_minute: int = 180, session=None, max_retries: int = 6,
                 trade_url: str | None = None, sleep=time.sleep):
        self.key_id = key_id or os.environ.get("APCA_API_KEY_ID")
        self.secret = secret or os.environ.get("APCA_API_SECRET_KEY")
        if not self.key_id or not self.secret:
            raise AlpacaError("缺少 APCA_API_KEY_ID / APCA_API_SECRET_KEY（在 Replit Secrets 里设置）")
        self.trade_url = (trade_url or os.environ.get("APCA_API_BASE_URL",
                                                      "https://paper-api.alpaca.markets")).rstrip("/")
        if self.trade_url.endswith("/v2"):          # 控制台显示的 Endpoint 带 /v2，这里统一去掉
            self.trade_url = self.trade_url[:-3]
        self.s = session or requests.Session()
        self.rl = RateLimiter(per_minute)
        self.max_retries = max_retries
        self._sleep = sleep
        self.calls = 0

    # ------------------------------------------------------------------ #
    def _get(self, url: str, params: dict) -> dict | list:
        headers = {"APCA-API-KEY-ID": self.key_id, "APCA-API-SECRET-KEY": self.secret}
        for attempt in range(self.max_retries):
            self.rl.wait()
            self.calls += 1
            try:
                r = self.s.get(url, params=params, headers=headers, timeout=60)
            except requests.RequestException as e:
                log.warning("网络错误 %s，重试 %d", e, attempt + 1)
                self._sleep(min(60, 2 ** attempt))
                continue
            if r.status_code == 429 or r.status_code >= 500:
                wait = min(60, 2 ** attempt)
                log.warning("HTTP %s，%ss 后重试", r.status_code, wait)
                self._sleep(wait)
                continue
            if r.status_code in (401, 403):
                raise AlpacaError(f"HTTP {r.status_code}: {getattr(r, 'text', '')[:300]} "
                                  "（检查 key 是否正确；免费账户查 SIP 时 end 必须早于 15 分钟前）")
            if r.status_code >= 400:
                raise AlpacaError(f"HTTP {r.status_code}: {getattr(r, 'text', '')[:300]} params={params}")
            return r.json()
        raise AlpacaError(f"重试 {self.max_retries} 次仍失败: {url}")

    # ------------------------------------------------------------------ #
    def assets(self, status: str = "active") -> pd.DataFrame:
        """us_equity 资产列表；status = active / inactive（已退市、停止交易的在 inactive 里）。"""
        js = self._get(f"{self.trade_url}/v2/assets",
                       {"status": status, "asset_class": "us_equity"})
        df = pd.DataFrame(js)
        if len(df) == 0:
            return pd.DataFrame(columns=["symbol", "exchange", "status", "name"])
        keep = [c for c in ["symbol", "exchange", "status", "name", "tradable"] if c in df.columns]
        return df[keep]

    def bars(self, symbols: Iterable[str], timeframe: str, start: str, end: str,
             adjustment: str = "raw", feed: str = "sip", limit: int = 10000) -> pd.DataFrame:
        symbols = list(symbols)
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start,
                  "end": end, "adjustment": adjustment, "feed": feed, "limit": limit,
                  "sort": "asc"}
        rows = []
        while True:
            js = self._get(f"{DATA_URL}/v2/stocks/bars", params)
            for sym, bars in (js.get("bars") or {}).items():
                for b in bars or []:
                    rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"],
                                 b.get("n"), b.get("vw")))
            tok = js.get("next_page_token")
            if not tok:
                break
            params = {**params, "page_token": tok}
        df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close",
                                         "volume", "trades", "vwap"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    def news(self, symbols: Iterable[str], start: str, end: str) -> pd.DataFrame:
        symbols = list(symbols)
        want = set(symbols)
        params = {"symbols": ",".join(symbols), "start": start, "end": end, "limit": 50,
                  "sort": "asc", "include_content": "false"}
        rows = []
        while True:
            js = self._get(f"{DATA_URL}/v1beta1/news", params)
            for a in js.get("news") or []:
                for s in a.get("symbols") or []:
                    if s in want:
                        rows.append((s, a.get("created_at"), a.get("headline"), a.get("source")))
            tok = js.get("next_page_token")
            if not tok:
                break
            params = {**params, "page_token": tok}
        df = pd.DataFrame(rows, columns=["symbol", "ts", "headline", "source"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    def trades(self, symbols: Iterable[str], start: str, end: str, feed: str = "sip",
               limit: int = 10000) -> pd.DataFrame:
        """逐笔成交：t 时间、p 价格、s 股数、c 成交条件、i 成交编号。"""
        symbols = list(symbols)
        params = {"symbols": ",".join(symbols), "start": start, "end": end, "feed": feed,
                  "limit": limit, "sort": "asc"}
        rows = []
        while True:
            js = self._get(f"{DATA_URL}/v2/stocks/trades", params)
            for sym, trs in (js.get("trades") or {}).items():
                for x in trs or []:
                    rows.append((sym, x["t"], x["p"], x.get("s"), ",".join(x.get("c") or []),
                                 x.get("i")))
            tok = js.get("next_page_token")
            if not tok:
                break
            params = {**params, "page_token": tok}
        df = pd.DataFrame(rows, columns=["symbol", "ts", "price", "size", "conditions", "id"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

    def crypto_bars(self, symbols: Iterable[str], timeframe: str, start: str, end: str,
                    limit: int = 10000) -> pd.DataFrame:
        """加密货币 K 线（v1beta3，美国交易场所），symbol 形如 BTC/USD。"""
        symbols = list(symbols)
        params = {"symbols": ",".join(symbols), "timeframe": timeframe, "start": start,
                  "end": end, "limit": limit, "sort": "asc"}
        rows = []
        while True:
            js = self._get(f"{DATA_URL}/v1beta3/crypto/us/bars", params)
            for sym, bars in (js.get("bars") or {}).items():
                for b in bars or []:
                    rows.append((sym, b["t"], b["o"], b["h"], b["l"], b["c"], b["v"]))
            tok = js.get("next_page_token")
            if not tok:
                break
            params = {**params, "page_token": tok}
        df = pd.DataFrame(rows, columns=["symbol", "ts", "open", "high", "low", "close", "volume"])
        df["ts"] = pd.to_datetime(df["ts"], utc=True, format="ISO8601")
        return df

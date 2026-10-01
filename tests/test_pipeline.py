"""
下载器和流水线的离线测试：用伪造的 Alpaca / SEC 响应，不访问网络。
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rossbt.alpaca import AlpacaClient, AlpacaError
from rossbt.data import prepare_daily_split_aware
from rossbt.sec import SecClient, shares_from_facts, shares_table
from rossbt.synth import generate_synthetic

ET = "America/New_York"


# ---------------------------------------------------------------- 假 HTTP ----
class Resp:
    def __init__(self, code, js=None, text=""):
        self.status_code, self._js, self.text = code, js, text

    def json(self):
        return self._js

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class SeqSession:
    """按顺序返回预设响应，并记录请求参数。"""
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def get(self, url, params=None, headers=None, timeout=None):
        self.calls.append((url, dict(params or {}), headers))
        return self.responses.pop(0)


def test_alpaca_bars_pagination_and_retry():
    page1 = {"bars": {"AAA": [{"t": "2026-03-10T11:00:00Z", "o": 1, "h": 2, "l": 1, "c": 2, "v": 100}]},
             "next_page_token": "tok"}
    page2 = {"bars": {"AAA": [{"t": "2026-03-10T11:01:00Z", "o": 2, "h": 3, "l": 2, "c": 3, "v": 50}],
                      "BBB": [{"t": "2026-03-10T11:00:00Z", "o": 5, "h": 5, "l": 5, "c": 5, "v": 7}]},
             "next_page_token": None}
    s = SeqSession([Resp(429), Resp(200, page1), Resp(503), Resp(200, page2)])
    c = AlpacaClient("k", "s", per_minute=60000, session=s, sleep=lambda x: None)
    df = c.bars(["AAA", "BBB"], "1Min", "2026-03-10", "2026-03-10")
    assert len(df) == 3 and set(df["symbol"]) == {"AAA", "BBB"}
    assert str(df["ts"].dt.tz) == "UTC"
    assert s.calls[-1][1]["page_token"] == "tok"
    assert s.calls[0][1]["feed"] == "sip" and s.calls[0][1]["adjustment"] == "raw"
    assert s.calls[0][2]["APCA-API-KEY-ID"] == "k"


def test_alpaca_auth_error_is_clear():
    s = SeqSession([Resp(403, {}, "forbidden")])
    c = AlpacaClient("k", "s", per_minute=60000, session=s, sleep=lambda x: None)
    with pytest.raises(AlpacaError, match="15 分钟"):
        c.bars(["AAA"], "1Min", "2026-03-10", "2026-03-10")


def test_alpaca_news_filters_symbols():
    js = {"news": [{"created_at": "2026-03-10T11:00:00Z", "symbols": ["AAA", "ZZZ"],
                    "headline": "h", "source": "benzinga"}], "next_page_token": None}
    c = AlpacaClient("k", "s", per_minute=60000, session=SeqSession([Resp(200, js)]),
                     sleep=lambda x: None)
    df = c.news(["AAA"], "2026-03-09", "2026-03-10")
    assert list(df["symbol"]) == ["AAA"]


def test_missing_keys(monkeypatch):
    monkeypatch.delenv("APCA_API_KEY_ID", raising=False)
    monkeypatch.delenv("APCA_API_SECRET_KEY", raising=False)
    with pytest.raises(AlpacaError):
        AlpacaClient()


# ---------------------------------------------------------- 拆股口径 ----
def test_split_aware_prev_close_and_adv():
    # 1 合 10 反向拆股发生在第 40 天：原始价 1.0 → 10.0，原始量 10000 → 1000
    dates = pd.bdate_range("2026-01-01", periods=45)
    raw_close = np.where(np.arange(45) < 40, 1.0, 10.0)
    raw_vol = np.where(np.arange(45) < 40, 10_000.0, 1_000.0)
    raw = pd.DataFrame({"symbol": "X", "date": dates.date, "close": raw_close, "volume": raw_vol})
    adj = raw.copy()
    adj["close"] = 10.0                                   # 调整后历史价都换算成拆股后口径
    out = prepare_daily_split_aware(raw, adj)
    day40 = out[out["date"] == dates[40].date()].iloc[0]
    assert day40["prev_close"] == pytest.approx(10.0)    # 不再出现假的 +900%
    assert day40["adv30"] == pytest.approx(1_000.0)      # 量也换算到拆股后口径
    day39 = out[out["date"] == dates[39].date()].iloc[0]
    assert day39["prev_close"] == pytest.approx(1.0)     # 拆股前的日子保持原始口径
    assert day39["adv30"] == pytest.approx(10_000.0)


def test_split_aware_merges_float_as_of_filed_date():
    dates = pd.bdate_range("2026-01-01", periods=40)
    raw = pd.DataFrame({"symbol": "X", "date": dates.date, "close": 5.0, "volume": 1000.0})
    ft = pd.DataFrame({"date": [pd.Timestamp(dates[30])], "symbol": ["X"], "float_shares": [8e6]})
    out = prepare_daily_split_aware(raw, raw.copy(), ft)
    assert pd.isna(out[out["date"] == dates[29].date()]["float_shares"].iloc[0])  # 申报前不可知
    assert out[out["date"] == dates[30].date()]["float_shares"].iloc[0] == 8e6


# ---------------------------------------------------------------- SEC ----
FACTS = {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
    {"end": "2025-03-31", "val": 9_000_000, "filed": "2025-05-10", "form": "10-Q"},
    {"end": "2025-06-30", "val": 12_000_000, "filed": "2025-08-12", "form": "10-Q"},
]}}}}}


def test_shares_from_facts():
    df = shares_from_facts(FACTS)
    assert list(df["shares"]) == [9e6, 12e6]
    assert shares_from_facts({"facts": {}}).empty


def test_shares_table_coverage(tmp_path):
    tick = {"0": {"cik_str": 123, "ticker": "AAA", "title": "A"},
            "1": {"cik_str": 456, "ticker": "BRK-B", "title": "B"}}
    s = SeqSession([Resp(200, tick), Resp(200, FACTS), Resp(404)])
    c = SecClient("t t@x.com", per_second=1000, session=s, cache_dir=tmp_path, sleep=lambda x: None)
    tbl, stats = shares_table(c, ["AAA", "BRK.B", "GONE"])
    assert stats == {"symbols": 3, "no_cik": 1, "no_shares": 1, "ok": 1}
    assert set(tbl["symbol"]) == {"AAA"} and len(tbl) == 2
    assert all("User-Agent" in h for _, _, h in s.calls)


# --------------------------------------------------------- 端到端流水线 ----
class FakeAlpaca:
    """用合成数据扮演 Alpaca：日线历史 + 分钟线 + 新闻。"""
    def __init__(self, n_days=12, n_symbols=10, seed=11):
        bars, daily, news = generate_synthetic(n_days=n_days, n_symbols=n_symbols,
                                               runner_prob=0.35, seed=seed)
        alpha = {f"SYN{i:02d}": "SY" + chr(65 + i // 26) + chr(65 + i % 26) for i in range(n_symbols)}
        for df in (bars, daily, news):
            df["symbol"] = df["symbol"].map(alpha)
        self.minute, self.news_df = bars, news
        self.days = sorted(daily["date"].unique())
        self.calls = Counter()
        # 日线：合成日之前 80 个交易日平稳；合成日用分钟线汇总；收盘衔接下一日昨收
        rows = []
        p0 = {(r.date, r.symbol): (r.prev_close, r.adv30) for r in daily.itertuples()}
        syms = sorted(daily["symbol"].unique())
        hist = pd.bdate_range(end=pd.Timestamp(self.days[0]) - pd.Timedelta(days=1), periods=80)
        for s in syms:
            pc0, adv0 = p0[(self.days[0], s)]
            for d in hist:
                rows.append((s, d.date(), pc0, pc0 * 1.01, pc0 * 0.99, pc0, adv0))
            for i, d in enumerate(self.days):
                m = bars[(bars["symbol"] == s) & (bars["ts"].dt.date == d)]
                nxt = p0.get((self.days[i + 1], s), (m["close"].iloc[-1], 0))[0] if i + 1 < len(self.days) \
                    else m["close"].iloc[-1]
                rows.append((s, d, m["open"].iloc[0], m["high"].max(), m["low"].min(), nxt,
                             m["volume"].sum()))
        self.daily = pd.DataFrame(rows, columns=["symbol", "date", "open", "high", "low",
                                                 "close", "volume"])
        self.daily["ts"] = pd.to_datetime(self.daily["date"].astype(str)).dt.tz_localize(ET).dt.tz_convert("UTC")
        self.syms = syms

    def assets(self, status):
        self.calls["assets"] += 1
        if status == "inactive":
            return pd.DataFrame({"symbol": ["DEADW", "OLD"], "exchange": ["NASDAQ", "NASDAQ"],
                                 "status": ["inactive"] * 2, "name": ["w", "o"]})
        return pd.DataFrame({"symbol": self.syms, "exchange": "NASDAQ", "status": "active",
                             "name": self.syms})

    def bars(self, symbols, timeframe, start, end, adjustment="raw", feed="sip"):
        self.calls[timeframe] += 1
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        if timeframe == "1Day":
            d = self.daily[self.daily["symbol"].isin(symbols)]
            d = d[(d["date"] >= s.date()) & (d["date"] <= e.date())]
            return d[["symbol", "ts", "open", "high", "low", "close", "volume"]].reset_index(drop=True)
        s = s if s.tzinfo else s.tz_localize(ET)
        e = e if e.tzinfo else e.tz_localize(ET)
        m = self.minute[self.minute["symbol"].isin(symbols)]
        m = m[(m["ts"] >= s) & (m["ts"] <= e)].copy()
        m["ts"] = m["ts"].dt.tz_convert("UTC")
        return m.reset_index(drop=True)

    def trades(self, symbols, start, end, feed="sip"):
        """把那根分钟线拆成逐笔：故意和 ohlc 假设反着来（阳线先高后低、阴线先低后高）。"""
        self.calls["trades"] += 1
        s0 = pd.Timestamp(start)
        m = self.minute[(self.minute["symbol"].isin(symbols))
                        & (self.minute["ts"].dt.tz_convert("UTC") == s0)]
        rows = []
        for r in m.itertuples():
            path = [r.open, r.high, r.low, r.close] if r.close >= r.open else \
                   [r.open, r.low, r.high, r.close]
            for k, px in enumerate(path):
                rows.append((r.symbol, s0 + pd.Timedelta(seconds=10 * k), px, 100, "@", k))
        rows.append((symbols[0], s0 + pd.Timedelta(seconds=5), 999.0, 100, "Z", 99))  # 乱序成交应被剔除
        return pd.DataFrame(rows, columns=["symbol", "ts", "price", "size", "conditions", "id"])

    def news(self, symbols, start, end):
        self.calls["news"] += 1
        s, e = pd.Timestamp(start), pd.Timestamp(end)
        n = self.news_df[self.news_df["symbol"].isin(symbols)]
        return n[(n["ts"] >= s) & (n["ts"] <= e)].assign(headline="h", source="t").reset_index(drop=True)


class FakeSec:
    def __init__(self, syms):
        self.syms = syms

    def ticker_map(self):
        return {s: i + 1 for i, s in enumerate(self.syms[:-2])}   # 最后两只查不到

    def company_facts(self, cik):
        return {"facts": {"dei": {"EntityCommonStockSharesOutstanding": {"units": {"shares": [
            {"end": "2025-12-31", "val": 4e6 + cik * 1e6, "filed": "2026-01-15"}]}}}}}


from collections import Counter  # noqa: E402


def test_pipeline_end_to_end_and_resume(tmp_path):
    import pipeline as pl
    fa = FakeAlpaca()
    start, end = str(fa.days[0]), str(fa.days[-1])
    p = pl.Pipeline(tmp_path / "data", tmp_path / "results", start, end, alpaca=fa,
                    sec=FakeSec(fa.syms), dq_samples=3, batch_symbols=4)
    ov = p.run_all()
    assert set(ov["run"]) == set(pl.RUNS)
    assert (tmp_path / "results" / "ross" / "trades.csv").exists()
    assert (tmp_path / "results" / "data_quality.json").exists()
    dq = json.loads((tmp_path / "results" / "data_quality.json").read_text())
    assert dq["inactive_symbols"] == 1                       # DEADW（权证）被过滤，只剩 OLD
    cands = pd.read_parquet(tmp_path / "data" / "candidates.parquet")
    assert len(cands) > 0
    minute_files = list((tmp_path / "data" / "minute").glob("*.parquet"))
    assert len(minute_files) == cands["date"].nunique()
    ross = ov.set_index("run").loc["ross"]
    assert ross["trades"] > 0, ov
    info = json.loads((tmp_path / "results" / "run_info.json").read_text())
    assert info["ambiguous_events"] > 0
    assert info["ambiguous_resolved_by_ticks"] == info["ambiguous_events"]
    assert fa.calls["trades"] > 0

    # 断点续传：再跑一次不应再打行情接口
    before = dict(fa.calls)
    p.run_all()
    assert dict(fa.calls) == before


def test_volume_prescreen_catches_premarket_only_spike():
    from rossbt.data import candidate_days
    d = pd.Timestamp("2026-03-10").date()
    daily = pd.DataFrame({"date": [d, d, d], "symbol": ["PM", "QUIET", "BIG"],
                          "prev_close": [3.0, 3.0, 25.0], "adv30": [1e5, 1e5, 1e5]})
    # PM：盘前冲到 +60% 后回落，日线最高价（不含盘前）只 +3%，但全天量 8 倍
    bars = pd.DataFrame({"date": [d, d, d], "symbol": ["PM", "QUIET", "BIG"],
                         "high": [3.09, 3.05, 30.0], "low": [2.9, 2.95, 24.0],
                         "volume": [8e5, 1.5e5, 9e5]})
    assert set(candidate_days(bars, daily, mode="price")["symbol"]) == set()
    assert set(candidate_days(bars, daily, mode="volume")["symbol"]) == {"PM"}   # BIG 昨收太高


def test_incremental_download_after_prescreen_change(tmp_path):
    import pipeline as pl
    fa = FakeAlpaca(seed=5)
    start, end = str(fa.days[0]), str(fa.days[-1])
    kw = dict(alpaca=fa, sec=FakeSec(fa.syms), dq_samples=2, batch_symbols=4)
    pl.Pipeline(tmp_path / "d", tmp_path / "r", start, end, prescreen="price", **kw).run_all()
    p2 = pl.Pipeline(tmp_path / "d", tmp_path / "r", start, end, prescreen="volume", **kw)
    p2.run_all()
    cands = pd.read_parquet(tmp_path / "d" / "candidates.parquet")
    for date, g in cands.groupby("date"):
        man = json.loads((tmp_path / "d" / "minute" / f"{date}.json").read_text())
        assert set(g["symbol"]) <= set(man)
    dq = json.loads((tmp_path / "r" / "data_quality.json").read_text())
    assert dq["signature"]["mode"] == "volume"


def test_extending_start_redownloads_daily_but_reuses_minutes(tmp_path):
    import pipeline as pl
    fa = FakeAlpaca(seed=9)
    kw = dict(alpaca=fa, sec=FakeSec(fa.syms), dq_samples=1, batch_symbols=4)
    pl.Pipeline(tmp_path / "d", tmp_path / "r", str(fa.days[3]), str(fa.days[-1]), **kw).run_all()
    m_before, d_before = fa.calls["1Min"], fa.calls["1Day"]
    pl.Pipeline(tmp_path / "d", tmp_path / "r", str(fa.days[0]), str(fa.days[-1]), **kw).run_all()
    assert fa.calls["1Day"] > d_before                          # 新区间重下日线
    cands = pd.read_parquet(tmp_path / "d" / "candidates.parquet")
    assert cands["date"].min() == fa.days[0]
    files = {f.stem for f in (tmp_path / "d" / "minute").glob("*.parquet")}
    assert {str(d) for d in cands["date"].unique()} <= files

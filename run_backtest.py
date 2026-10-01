"""
命令行：

  # 合成数据演示（验证能跑通）
  python run_backtest.py --demo

  # 真实数据
  python run_backtest.py --bars bars.parquet --daily daily.csv --news news.csv \
      --preset ross --set top_n_gainers=3 --set max_pullback_number=2 --out results/

  --preset   ross | summary | aggressive
  --config   JSON 文件，覆盖任意参数
  --set      单个参数覆盖，可重复（值按 JSON 解析，如 --set entry_blackouts='[["09:30","09:35"]]'）
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from rossbt import (PRESETS, Config, load_bars, load_daily, load_news, run_backtest,
                    summarize, print_summary)


def build_cfg(args) -> Config:
    d = asdict(PRESETS[args.preset])
    if args.config:
        d.update(json.loads(Path(args.config).read_text()))
    for kv in args.set or []:
        k, v = kv.split("=", 1)
        if k not in d:
            raise SystemExit(f"未知参数: {k}")
        try:
            d[k] = json.loads(v)
        except json.JSONDecodeError:
            d[k] = v
    return Config.from_dict(d)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bars"); ap.add_argument("--daily"); ap.add_argument("--news")
    ap.add_argument("--preset", default="ross", choices=list(PRESETS))
    ap.add_argument("--config"); ap.add_argument("--set", action="append")
    ap.add_argument("--out", default="results")
    ap.add_argument("--demo", action="store_true")
    ap.add_argument("--progress", action="store_true")
    args = ap.parse_args()
    cfg = build_cfg(args)

    if args.demo:
        from rossbt.synth import generate_synthetic
        b, d, n = generate_synthetic()
        bars, daily, news = load_bars(b), load_daily(d), load_news(n)
    else:
        if not (args.bars and args.daily):
            raise SystemExit("需要 --bars 和 --daily（或 --demo）")
        bars, daily = load_bars(args.bars), load_daily(args.daily)
        news = load_news(args.news) if args.news else None

    res = run_backtest(bars, daily, cfg, news=news, progress=args.progress)
    s = summarize(res)
    print_summary(s, res.notes)

    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    res.trades.to_csv(out / "trades.csv", index=False)
    res.trades_all.to_csv(out / "trades_all_signals.csv", index=False)
    res.dropped.to_csv(out / "dropped_by_portfolio.csv", index=False)
    if "equity" in s:
        s["equity"].rename("cum_pnl").to_csv(out / "equity_daily.csv")
    (out / "config_used.json").write_text(cfg.to_json())
    summary_json = {k: v for k, v in s.items() if not k.startswith("by_") and k != "equity"}
    (out / "summary.json").write_text(json.dumps(summary_json, ensure_ascii=False, indent=2,
                                                 default=str))
    for k, v in s.items():
        if k.startswith("by_") and len(v):
            v.to_csv(out / f"{k}.csv")
    print(f"\n结果已写入 {out.resolve()}")


if __name__ == "__main__":
    main()

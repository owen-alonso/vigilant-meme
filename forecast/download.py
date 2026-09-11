"""Pull OHLCV into the canonical parquet cache.

Alpha Vantage free daily history is the latest ~100 bars.
Weekly/monthly from the same key, or Stooq/Yahoo daily, return much longer history.

Usage:
    python -m forecast.download --symbols AAPL,MSFT
    python -m forecast.download --symbols AAPL,MSFT --interval weekly --source yahoo --replace
    python -m forecast.download --universe liquid --source yahoo --replace
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from forecast.alphavantage import (
    AlphaVantageClient,
    AlphaVantageError,
    download_symbol,
    symbol_parquet_path,
    write_bars,
)
from forecast.config import DataConfig
from forecast.generate import parse_symbols
from forecast.stooq import fetch_stooq_bars
from forecast.universe import download_symbols as universe_download_symbols
from forecast.yahoo import fetch_yahoo_bars


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Download Alpha Vantage / Yahoo / Stooq bars into data/<SYMBOL>_<interval>.parquet."
    )
    p.add_argument(
        "--symbols",
        default=None,
        help="comma-separated tickers, e.g. AAPL,MSFT. Omit with --universe liquid.",
    )
    p.add_argument(
        "--universe",
        default=None,
        choices=("liquid", "liquid_wide"),
        help="train-era-locked liquid (~85) or liquid_wide (~175) names (+ SPY).",
    )
    p.add_argument("--data-dir", default=DataConfig().data_dir)
    p.add_argument(
        "--interval",
        default="daily",
        choices=("daily", "weekly", "monthly", "1min", "5min", "15min", "30min", "60min"),
        help="daily compact is ~100 bars on a free Alpha Vantage key; "
        "weekly/monthly return 20+ years; yahoo/stooq daily is full history",
    )
    p.add_argument(
        "--source",
        default="alphavantage",
        choices=("alphavantage", "stooq", "yahoo"),
        help="alphavantage (~100 daily bars on a free key), yahoo (full daily history, no key), "
        "or stooq",
    )
    p.add_argument(
        "--history-months",
        type=int,
        default=0,
        help="intraday only: fetch that many YYYY-MM slices instead of the latest window",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="outputsize=full (premium on TIME_SERIES_DAILY). Default is compact (~100 bars).",
    )
    p.add_argument(
        "--extended-hours",
        action="store_true",
        help="include pre/post-market bars (the session grid still drops them)",
    )
    p.add_argument(
        "--adjusted",
        action="store_true",
        help="TIME_SERIES_*_ADJUSTED (split-adjusted close). "
        "Weekly/monthly default to this; daily adjusted is premium.",
    )
    p.add_argument(
        "--no-adjusted",
        action="store_true",
        help="keep raw unadjusted close (split jumps become returns)",
    )
    p.add_argument(
        "--replace",
        action="store_true",
        help="overwrite existing parquets instead of upserting (use when the cache is mixed-scale)",
    )
    p.add_argument(
        "--min-interval-sec",
        type=float,
        default=None,
        help="sleep between Alpha Vantage API calls (default: ALPHA_VANTAGE_MIN_INTERVAL_SEC or 13)",
    )
    p.add_argument(
        "--sleep-sec",
        type=float,
        default=0.35,
        help="pause between Yahoo/Stooq symbol fetches (be kind to the public API)",
    )
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="do not fetch a symbol that already has a parquet in --data-dir",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    if args.universe:
        symbols = universe_download_symbols(
            include_benchmark=True, universe=str(args.universe)
        )
    else:
        symbols = parse_symbols(args.symbols)
    if not symbols:
        print("no symbols given (pass --symbols or --universe liquid)", file=sys.stderr)
        return 2
    if args.adjusted and args.no_adjusted:
        print("use only one of --adjusted / --no-adjusted", file=sys.stderr)
        return 2
    if args.no_adjusted:
        adjusted = False
    elif args.adjusted:
        adjusted = True
    else:
        adjusted = args.interval in ("weekly", "monthly")
    failed: list[str] = []
    try:
        client = None
        if args.source == "alphavantage":
            client = AlphaVantageClient(min_interval_sec=args.min_interval_sec)
        for i, symbol in enumerate(symbols):
            dest = symbol_parquet_path(args.data_dir, symbol, interval=args.interval)
            if args.skip_existing and dest.exists() and not args.replace:
                print(f"skip {symbol.upper()}: already cached")
                continue
            try:
                if args.source == "stooq":
                    print(f"fetching {symbol.upper()} {args.interval} via stooq")
                    bars = fetch_stooq_bars(symbol, interval=args.interval)
                    path = symbol_parquet_path(args.data_dir, symbol, interval=args.interval)
                    merged = write_bars(path, bars, replace=args.replace)
                    print(
                        f"  {len(bars)} bars  {bars['datetime'].min()} -> {bars['datetime'].max()}"
                    )
                    print(f"wrote {path} ({len(merged)} bars)")
                elif args.source == "yahoo":
                    print(f"fetching {symbol.upper()} {args.interval} via yahoo")
                    bars = fetch_yahoo_bars(symbol, interval=args.interval)
                    path = symbol_parquet_path(args.data_dir, symbol, interval=args.interval)
                    merged = write_bars(path, bars, replace=args.replace)
                    print(
                        f"  {len(bars)} bars  {bars['datetime'].min()} -> {bars['datetime'].max()}"
                    )
                    print(f"wrote {path} ({len(merged)} bars)")
                else:
                    download_symbol(
                        symbol,
                        args.data_dir,
                        client=client,
                        history_months=args.history_months,
                        outputsize="full" if args.full else "compact",
                        interval=args.interval,
                        extended_hours=args.extended_hours,
                        adjusted=adjusted,
                        replace=args.replace,
                    )
            except (AlphaVantageError, OSError, ValueError) as exc:
                print(f"FAILED {symbol.upper()}: {exc}", file=sys.stderr)
                failed.append(symbol.upper())
            if (
                args.source in ("yahoo", "stooq")
                and float(args.sleep_sec) > 0
                and i + 1 < len(symbols)
            ):
                time.sleep(float(args.sleep_sec))
    except AlphaVantageError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if failed:
        print(f"{len(failed)} symbol(s) failed: {', '.join(failed)}", file=sys.stderr)
        return 1 if len(failed) == len(symbols) else 0
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

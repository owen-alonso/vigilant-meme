"""Alpha Vantage client and canonical OHLCV store.

Vendor dumps are no longer the source of truth. Training and generate read a
local cache of Alpha Vantage bars:

    data/<SYMBOL>_daily.parquet     (default, free TIME_SERIES_DAILY compact)
    data/<SYMBOL>_1min.parquet      (premium TIME_SERIES_INTRADAY)

Columns: datetime, open, high, low, close, volume, source, interval.

Free keys are typically 5 requests/minute and 25/day. Daily ``outputsize=full``
and 1-minute history are premium.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path
from typing import Any, Callable

import pandas as pd

from mamba_lm.paths import REPO_ROOT, anchor_to_repo

BASE_URL = "https://www.alphavantage.co/query"
SOURCE_NAME = "alphavantage"
DEFAULT_INTERVAL = "daily"
DEFAULT_OUTPUTSIZE = "compact"
CANONICAL_COLUMNS = (
    "datetime",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "interval",
)
_TIME_SERIES_PREFIX = "Time Series"


def _time_series_key(payload: dict[str, Any]) -> str | None:
    """Weekly/monthly payloads use 'Weekly Time Series', not 'Time Series (...)'."""
    for key, value in payload.items():
        if not isinstance(value, dict) or not value:
            continue
        name = str(key).lower()
        if name.startswith("meta"):
            continue
        if "time series" in name:
            return str(key)
    return None


class AlphaVantageError(RuntimeError):
    """HTTP, parse, or vendor-note failure from Alpha Vantage."""


def load_dotenv(path: str | Path | None = None) -> None:
    """Load KEY=VALUE pairs from a .env file without overriding the process env."""
    candidates: list[Path] = []
    if path is not None:
        candidates.append(Path(path))
    else:
        candidates.extend((Path.cwd() / ".env", REPO_ROOT / ".env"))
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        for raw in candidate.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'").strip('"')
            if key and key not in os.environ:
                os.environ[key] = value


def api_key_from_env() -> str:
    load_dotenv()
    key = os.environ.get("ALPHA_VANTAGE_API_KEY", "").strip()
    if not key:
        raise AlphaVantageError(
            "ALPHA_VANTAGE_API_KEY is not set. Copy .env.example to .env "
            "and put your Alpha Vantage key there."
        )
    return key


def min_interval_from_env(default: float = 13.0) -> float:
    load_dotenv()
    raw = os.environ.get("ALPHA_VANTAGE_MIN_INTERVAL_SEC", "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError as exc:
        raise AlphaVantageError(
            f"ALPHA_VANTAGE_MIN_INTERVAL_SEC must be a number, got {raw!r}"
        ) from exc
    if value < 0:
        raise AlphaVantageError("ALPHA_VANTAGE_MIN_INTERVAL_SEC must be >= 0")
    return value


def symbol_parquet_path(
    data_dir: str | Path, symbol: str, interval: str = DEFAULT_INTERVAL
) -> Path:
    root = Path(data_dir)
    if not root.is_absolute():
        root = anchor_to_repo(root)
    suffix = "daily" if interval == "daily" else interval
    return root / f"{str(symbol).upper()}_{suffix}.parquet"


def bars_to_canonical(
    df: pd.DataFrame,
    *,
    interval: str = DEFAULT_INTERVAL,
    source: str = SOURCE_NAME,
) -> pd.DataFrame:
    """Normalize any OHLCV-like frame to the on-disk schema."""
    if df.empty:
        return pd.DataFrame(columns=list(CANONICAL_COLUMNS))
    out = df.copy()
    renamed: dict[Any, str] = {}
    for col in out.columns:
        key = str(col).lower().replace(" ", "")
        if key in {"date", "datetime", "timestamp", "index"}:
            renamed[col] = "datetime"
        else:
            renamed[col] = key
    out = out.rename(columns=renamed)
    if "datetime" not in out.columns:
        if isinstance(out.index, pd.DatetimeIndex):
            out = out.reset_index().rename(columns={out.columns[0]: "datetime"})
        else:
            raise ValueError("expected a datetime column or DatetimeIndex")
    adj_col = next((c for c in out.columns if str(c).replace(" ", "") == "adjustedclose"), None)
    if adj_col is not None and "close" in out.columns:
        close = pd.to_numeric(out["close"], errors="coerce")
        adj = pd.to_numeric(out[adj_col], errors="coerce")
        factor = adj / close.replace(0, float("nan"))
        for col in ("open", "high", "low"):
            if col in out.columns:
                out[col] = pd.to_numeric(out[col], errors="coerce") * factor
        out["close"] = adj
    missing = [c for c in ("open", "high", "low", "close", "volume") if c not in out.columns]
    if missing:
        raise ValueError(f"missing columns {missing}")
    out["datetime"] = pd.to_datetime(out["datetime"])
    if getattr(out["datetime"].dtype, "tz", None) is None:
        out["datetime"] = out["datetime"].dt.tz_localize("America/New_York")
    else:
        out["datetime"] = out["datetime"].dt.tz_convert("America/New_York")
    for col in ("open", "high", "low", "close", "volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ("open", "high", "low"):
        if col in out.columns:
            out[col] = out[col].where(out[col] > 0)
    out["source"] = source
    out["interval"] = interval
    out = out.dropna(subset=["datetime", "close"])
    out = out.loc[out["close"] > 0]
    out = out.sort_values("datetime").drop_duplicates("datetime", keep="last")
    return out.loc[:, list(CANONICAL_COLUMNS)].reset_index(drop=True)


def parse_intraday_payload(payload: dict[str, Any], *, interval: str = "1min") -> pd.DataFrame:
    """Turn a TIME_SERIES_* JSON object into canonical bars."""
    if not isinstance(payload, dict):
        raise AlphaVantageError(f"expected a JSON object, got {type(payload).__name__}")
    series_key = _time_series_key(payload)
    if series_key is None:
        for banner_key in ("Error Message", "Note", "Information"):
            message = payload.get(banner_key)
            if message:
                raise AlphaVantageError(f"Alpha Vantage {banner_key}: {message}")
        keys = ", ".join(sorted(payload)) or "(empty)"
        raise AlphaVantageError(f"no time series in Alpha Vantage payload; keys: {keys}")
    series = payload[series_key]
    if not isinstance(series, dict) or not series:
        raise AlphaVantageError("Alpha Vantage time series was empty")
    rows = []
    for stamp, bar in series.items():
        if not isinstance(bar, dict):
            continue
        flat = {str(k).split(". ", 1)[-1].lower(): v for k, v in bar.items()}
        rows.append({"datetime": stamp, **flat})
    if not rows:
        raise AlphaVantageError("Alpha Vantage time series had no bars")
    return bars_to_canonical(pd.DataFrame(rows), interval=interval)


def is_premium_feature_error(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "premium" in text


def month_labels(n_months: int, *, end: date | None = None) -> list[str]:
    """YYYY-MM labels, newest first, covering ``n_months`` calendar months."""
    if n_months < 1:
        raise ValueError("n_months must be >= 1")
    end = end or date.today()
    year, month = end.year, end.month
    labels: list[str] = []
    for _ in range(n_months):
        labels.append(f"{year:04d}-{month:02d}")
        month -= 1
        if month == 0:
            month = 12
            year -= 1
    return labels


def upsert_bars(path: str | Path, incoming: pd.DataFrame) -> pd.DataFrame:
    """Merge ``incoming`` into an existing parquet, keeping the latest stamp."""
    path = Path(path)
    frames = [incoming]
    if path.exists():
        frames.insert(0, pd.read_parquet(path))
    merged = pd.concat(frames, ignore_index=True)
    interval = str(incoming["interval"].iloc[0]) if len(incoming) else DEFAULT_INTERVAL
    merged = bars_to_canonical(merged, interval=interval)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(path, index=False)
    return merged


def write_bars(
    path: str | Path, incoming: pd.DataFrame, *, replace: bool = False
) -> pd.DataFrame:
    """Write bars. ``replace`` overwrites an existing parquet instead of merging."""
    path = Path(path)
    if replace and path.exists():
        path.unlink()
    if replace or not path.exists():
        interval = str(incoming["interval"].iloc[0]) if len(incoming) else DEFAULT_INTERVAL
        out = bars_to_canonical(incoming, interval=interval)
        path.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(path, index=False)
        return out
    return upsert_bars(path, incoming)


class AlphaVantageClient:
    def __init__(
        self,
        api_key: str | None = None,
        *,
        min_interval_sec: float | None = None,
        opener: Callable[[str], Any] | None = None,
        timeout_sec: float = 30.0,
    ) -> None:
        self.api_key = api_key if api_key is not None else api_key_from_env()
        self.min_interval_sec = (
            min_interval_from_env() if min_interval_sec is None else float(min_interval_sec)
        )
        self._opener = opener
        self.timeout_sec = timeout_sec
        self._last_call = 0.0

    def _throttle(self) -> None:
        if self.min_interval_sec <= 0:
            return
        wait = self.min_interval_sec - (time.monotonic() - self._last_call)
        if wait > 0:
            time.sleep(wait)

    def get_json(self, params: dict[str, str]) -> dict[str, Any]:
        query = dict(params)
        query["apikey"] = self.api_key
        url = f"{BASE_URL}?{urllib.parse.urlencode(query)}"
        self._throttle()
        self._last_call = time.monotonic()
        try:
            if self._opener is not None:
                raw = self._opener(url)
                if isinstance(raw, (bytes, bytearray)):
                    text = bytes(raw).decode("utf-8")
                else:
                    text = str(raw)
            else:
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "vigilant-meme/forecast"},
                )
                with urllib.request.urlopen(request, timeout=self.timeout_sec) as response:
                    text = response.read().decode("utf-8")
        except urllib.error.URLError as exc:
            raise AlphaVantageError(f"Alpha Vantage request failed: {exc}") from exc
        try:
            payload = json.loads(text)
        except json.JSONDecodeError as exc:
            raise AlphaVantageError("Alpha Vantage returned non-JSON") from exc
        if not isinstance(payload, dict):
            raise AlphaVantageError("Alpha Vantage JSON was not an object")
        return payload

    def fetch_bars(
        self,
        symbol: str,
        *,
        interval: str = DEFAULT_INTERVAL,
        outputsize: str = DEFAULT_OUTPUTSIZE,
        month: str | None = None,
        extended_hours: bool = False,
        adjusted: bool = False,
    ) -> pd.DataFrame:
        ticker = str(symbol).upper()
        if interval == "daily":
            params = {
                "function": "TIME_SERIES_DAILY_ADJUSTED" if adjusted else "TIME_SERIES_DAILY",
                "symbol": ticker,
                "outputsize": outputsize,
                "datatype": "json",
            }
        elif interval == "weekly":
            params = {
                "function": "TIME_SERIES_WEEKLY_ADJUSTED" if adjusted else "TIME_SERIES_WEEKLY",
                "symbol": ticker,
                "datatype": "json",
            }
        elif interval == "monthly":
            params = {
                "function": "TIME_SERIES_MONTHLY_ADJUSTED" if adjusted else "TIME_SERIES_MONTHLY",
                "symbol": ticker,
                "datatype": "json",
            }
        else:
            params = {
                "function": "TIME_SERIES_INTRADAY",
                "symbol": ticker,
                "interval": interval,
                "outputsize": outputsize,
                "extended_hours": "true" if extended_hours else "false",
                "adjusted": "true" if adjusted else "false",
                "datatype": "json",
            }
            if month:
                params["month"] = month
        payload = self.get_json(params)
        try:
            return parse_intraday_payload(payload, interval=interval)
        except AlphaVantageError as exc:
            if outputsize != "full" or not is_premium_feature_error(exc):
                raise
            compact = dict(params)
            compact["outputsize"] = "compact"
            payload = self.get_json(compact)
            return parse_intraday_payload(payload, interval=interval)

    def fetch_intraday(self, symbol: str, **kwargs: Any) -> pd.DataFrame:
        return self.fetch_bars(symbol, **kwargs)


def download_symbol(
    symbol: str,
    data_dir: str | Path,
    *,
    client: AlphaVantageClient | None = None,
    history_months: int = 0,
    outputsize: str = DEFAULT_OUTPUTSIZE,
    interval: str = DEFAULT_INTERVAL,
    extended_hours: bool = False,
    adjusted: bool = False,
    replace: bool = False,
    log_fn: Callable[[str], None] | None = print,
) -> pd.DataFrame:
    """Fetch latest (and optional historical months) and write the parquet cache."""
    client = client or AlphaVantageClient()
    ticker = str(symbol).upper()
    path = symbol_parquet_path(data_dir, ticker, interval=interval)
    chunks: list[pd.DataFrame] = []

    months: list[str | None] = [None]
    if interval not in {"daily", "weekly", "monthly"} and history_months > 0:
        months = month_labels(history_months)

    for month in months:
        label = month or "latest"
        if log_fn:
            log_fn(f"fetching {ticker} {interval} {label} via alphavantage")
        bars = client.fetch_bars(
            ticker,
            interval=interval,
            outputsize=outputsize,
            month=month,
            extended_hours=extended_hours,
            adjusted=adjusted,
        )
        if log_fn and outputsize == "full" and len(bars) <= 100:
            log_fn(
                "  note: Alpha Vantage returned compact history (~100 bars); "
                "outputsize=full is premium on this key"
            )
        chunks.append(bars)
        if log_fn:
            log_fn(f"  {len(bars)} bars  {bars['datetime'].min()} -> {bars['datetime'].max()}")

    incoming = pd.concat(chunks, ignore_index=True)
    merged = write_bars(path, incoming, replace=replace)
    if log_fn:
        log_fn(f"wrote {path} ({len(merged)} bars)")
    return merged

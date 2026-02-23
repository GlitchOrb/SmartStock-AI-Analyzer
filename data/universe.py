from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from io import StringIO
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
import yfinance as yf


CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)

NASDAQ_LISTED_URL = "https://ftp.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
WIKI_NDX_URL = "https://en.wikipedia.org/wiki/Nasdaq-100"

DEFAULT_FALLBACK = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AVGO", "AMD", "NFLX",
    "QCOM", "INTC", "MU", "ARM", "SMCI", "CRWD", "NET", "PLTR", "COIN", "MSTR",
]


def _http_get_with_retry(url: str, timeout: int = 20, attempts: int = 3) -> requests.Response:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            resp = requests.get(url, timeout=timeout)
            resp.raise_for_status()
            return resp
        except Exception as exc:
            last_exc = exc
            if attempt < attempts:
                time.sleep(0.8 * attempt)
    if last_exc:
        raise last_exc
    raise RuntimeError(f"Failed to GET {url}")


@dataclass
class UniverseConfig:
    kind: str = "nasdaq_all"
    max_size: int = 0
    ttl_seconds: int = 86400
    include_etf: bool = False


@dataclass
class UniversePreFilterConfig:
    liquidity_threshold: int = 100_000
    min_dollar_volume_20d: float = 5_000_000.0
    min_price: float = 1.0
    min_history_days: int = 30
    target_min_size: int = 500
    target_max_size: int = 1200
    ttl_seconds: int = 21600
    chunk_size: int = 250
    history_period: str = "3mo"


def _universe_cache_path(cfg: UniverseConfig) -> Path:
    return CACHE_DIR / f"universe_{cfg.kind}_{cfg.max_size}_{int(cfg.include_etf)}.json"


def _tickers_fingerprint(tickers: list[str]) -> str:
    joined = ",".join(sorted(tickers))
    return hashlib.md5(joined.encode("utf-8")).hexdigest()[:12]


def _prefilter_cache_path(cfg: UniversePreFilterConfig, tickers: list[str]) -> Path:
    universe_fingerprint = _tickers_fingerprint(tickers)
    return CACHE_DIR / (
        "nasdaq_prefilter_"
        f"{cfg.liquidity_threshold}_{cfg.min_dollar_volume_20d}_{cfg.min_price}_{cfg.min_history_days}_{cfg.target_min_size}_{cfg.target_max_size}_{universe_fingerprint}.json"
    )


def _load_cache(path: Path, ttl_seconds: int) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        saved_at = float(payload.get("saved_at", 0))
        if (time.time() - saved_at) <= ttl_seconds:
            return payload
    except Exception:
        return None
    return None


def _save_cache(path: Path, payload: dict[str, Any]) -> None:
    payload = dict(payload)
    payload["saved_at"] = time.time()
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _is_common_stock(row: pd.Series, include_etf: bool) -> bool:
    etf = str(row.get("ETF", "")).strip().upper()
    test_issue = str(row.get("Test Issue", "")).strip().upper()
    symbol = str(row.get("Symbol", "")).strip().upper()

    if not symbol or symbol == "SYMBOL":
        return False
    if test_issue == "Y":
        return False
    if not include_etf and etf == "Y":
        return False
    return True


def _fetch_nasdaq_listed(include_etf: bool = False) -> list[str]:
    response = _http_get_with_retry(NASDAQ_LISTED_URL, timeout=20, attempts=3)

    lines = response.text.splitlines()
    body = [ln for ln in lines if "|" in ln and not ln.startswith("File Creation Time")]
    df = pd.read_csv(StringIO("\n".join(body)), sep="|", dtype=str)
    df = df[df["Symbol"].notna()].copy()
    df = df[df.apply(lambda row: _is_common_stock(row, include_etf), axis=1)]

    tickers = df["Symbol"].astype(str).str.upper().str.strip().tolist()
    return sorted(list(dict.fromkeys(tickers)))


def _fetch_nasdaq100_from_wikipedia() -> list[str]:
    response = _http_get_with_retry(WIKI_NDX_URL, timeout=20, attempts=3)
    tables = pd.read_html(response.text)

    for table in tables:
        lowered = [str(c).lower() for c in table.columns]
        if "ticker" in lowered or "ticker symbol" in lowered:
            col = next((c for c in table.columns if str(c).lower() in {"ticker", "ticker symbol"}), None)
            if col is None:
                continue
            tickers = (
                table[col]
                .astype(str)
                .str.upper()
                .str.replace(r"\.", "-", regex=True)
                .str.strip()
                .tolist()
            )
            tickers = [t for t in tickers if t and t != "NAN"]
            return sorted(list(dict.fromkeys(tickers)))
    return []


def build_universe(cfg: UniverseConfig) -> list[str]:
    cached = _load_cache(_universe_cache_path(cfg), cfg.ttl_seconds)
    if cached and isinstance(cached.get("tickers"), list):
        return list(cached["tickers"])

    tickers: list[str]
    try:
        if cfg.kind == "nasdaq100":
            tickers = _fetch_nasdaq100_from_wikipedia()
            if not tickers:
                tickers = _fetch_nasdaq_listed(include_etf=cfg.include_etf)
        elif cfg.kind == "nasdaq300":
            ndx = _fetch_nasdaq100_from_wikipedia()
            listed = _fetch_nasdaq_listed(include_etf=cfg.include_etf)
            extra = [t for t in listed if t not in set(ndx)]
            tickers = ndx + extra
        else:
            tickers = _fetch_nasdaq_listed(include_etf=cfg.include_etf)
    except Exception:
        tickers = DEFAULT_FALLBACK[:]

    if cfg.max_size and cfg.max_size > 0:
        tickers = tickers[: cfg.max_size]

    tickers = [t for t in tickers if t and t.isascii()]
    if not tickers:
        tickers = DEFAULT_FALLBACK[:]

    _save_cache(_universe_cache_path(cfg), {"tickers": tickers})
    return tickers


def _normalize_batch_columns(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data
    lvl0 = list(map(str, data.columns.levels[0]))
    if "Close" in lvl0:
        return data.swaplevel(0, 1, axis=1)
    return data


def _extract_ticker_df(batch: pd.DataFrame, ticker: str, is_single: bool) -> pd.DataFrame | None:
    if is_single:
        df = batch.copy()
    elif isinstance(batch.columns, pd.MultiIndex) and ticker in batch.columns.get_level_values(0):
        df = batch[ticker].copy()
    else:
        return None

    cols = [c for c in ("Open", "High", "Low", "Close", "Volume") if c in df.columns]
    if not cols:
        return None
    df = df[cols].dropna(how="any")
    if df.empty:
        return None
    return df


def build_prefiltered_universe(
    tickers: list[str],
    cfg: UniversePreFilterConfig,
) -> tuple[list[str], dict[str, Any]]:
    cache_path = _prefilter_cache_path(cfg, tickers)
    cached = _load_cache(cache_path, cfg.ttl_seconds)
    if cached and isinstance(cached.get("tickers"), list):
        stats = dict(cached.get("stats", {}))
        stats["from_cache"] = True
        return list(cached["tickers"]), stats

    kept: list[tuple[str, float]] = []
    eligible_history = 0
    scanned = 0
    price_pass_count = 0

    for i in range(0, len(tickers), cfg.chunk_size):
        chunk = tickers[i : i + cfg.chunk_size]
        if not chunk:
            continue
        scanned += len(chunk)

        raw = pd.DataFrame()
        for attempt in range(1, 4):
            try:
                raw = yf.download(
                    chunk,
                    period=cfg.history_period,
                    interval="1d",
                    group_by="ticker",
                    threads=True,
                    progress=False,
                    auto_adjust=False,
                )
                if isinstance(raw, pd.DataFrame) and not raw.empty:
                    break
            except Exception:
                pass
            time.sleep(0.8 * attempt)
        if raw.empty:
            continue

        batch = _normalize_batch_columns(raw)
        single = not isinstance(batch.columns, pd.MultiIndex)

        for ticker in chunk:
            df = _extract_ticker_df(batch, ticker, is_single=(single and len(chunk) == 1))
            if df is None:
                continue
            if len(df) < cfg.min_history_days:
                continue
            eligible_history += 1

            last_price = float(df["Close"].iloc[-1])
            if pd.isna(last_price) or last_price < cfg.min_price:
                continue
            price_pass_count += 1

            avg_vol_20 = float(df["Volume"].tail(20).mean())
            avg_dollar_vol_20 = avg_vol_20 * last_price
            if pd.isna(avg_vol_20) or avg_vol_20 < cfg.liquidity_threshold:
                continue
            if pd.isna(avg_dollar_vol_20) or avg_dollar_vol_20 < cfg.min_dollar_volume_20d:
                continue
            kept.append((ticker, avg_dollar_vol_20))

    kept.sort(key=lambda item: item[1], reverse=True)
    if cfg.target_max_size > 0:
        kept = kept[: cfg.target_max_size]

    selected = [t for t, _ in kept]

    stats = {
        "source_universe_size": len(tickers),
        "scanned_tickers": scanned,
        "eligible_history_count": eligible_history,
        "price_pass_count": price_pass_count,
        "liquidity_pass_count": len(selected),
        "liquidity_threshold": cfg.liquidity_threshold,
        "min_dollar_volume_20d": cfg.min_dollar_volume_20d,
        "min_price": cfg.min_price,
        "min_history_days": cfg.min_history_days,
        "target_range": f"{cfg.target_min_size}-{cfg.target_max_size}",
        "from_cache": False,
    }

    _save_cache(cache_path, {"tickers": selected, "stats": stats})
    return selected, stats

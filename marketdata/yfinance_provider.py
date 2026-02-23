from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import pandas as pd
import yfinance as yf

from data.universe import UniverseConfig, build_universe
from marketdata.provider_base import MarketDataProvider, Snapshot


US_EASTERN = ZoneInfo("America/New_York")


def _session_label(ts: datetime | None = None) -> str:
    now = ts.astimezone(US_EASTERN) if ts else datetime.now(US_EASTERN)
    minutes = now.hour * 60 + now.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "PRE"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "REG"
    if 16 * 60 <= minutes < 20 * 60:
        return "AFTER"
    return "CLOSED"


def _normalize_download(df: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(df.columns, pd.MultiIndex):
        return df
    lvl0 = [str(x) for x in df.columns.levels[0]]
    if "Close" in lvl0:
        return df.swaplevel(0, 1, axis=1)
    return df


class YFinanceProvider(MarketDataProvider):
    """Fallback delayed provider for development."""

    name = "yfinance"

    def get_universe(self, include_etf: bool = False) -> list[str]:
        return build_universe(UniverseConfig(kind="nasdaq_all", max_size=0, include_etf=include_etf))

    def get_snapshot(self, tickers: list[str]) -> dict[str, Snapshot]:
        out: dict[str, Snapshot] = {}
        if not tickers:
            return out
        try:
            raw = yf.download(
                tickers,
                period="2d",
                interval="1m",
                prepost=True,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception:
            return out

        data = _normalize_download(raw)
        single = not isinstance(data.columns, pd.MultiIndex)
        for ticker in tickers:
            try:
                df = data if (single and len(tickers) == 1) else data[ticker]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
                if df.empty:
                    continue
                last = float(df["Close"].iloc[-1])
                ts = df.index[-1].to_pydatetime().replace(tzinfo=US_EASTERN) if df.index.tz is None else df.index[-1].to_pydatetime()
                chg_1m = 0.0
                chg_5m = 0.0
                if len(df) >= 2 and df["Close"].iloc[-2] != 0:
                    chg_1m = ((last - float(df["Close"].iloc[-2])) / float(df["Close"].iloc[-2])) * 100
                if len(df) >= 6 and df["Close"].iloc[-6] != 0:
                    chg_5m = ((last - float(df["Close"].iloc[-6])) / float(df["Close"].iloc[-6])) * 100
                vwap = ((df["Close"] * df["Volume"]).cumsum() / df["Volume"].replace(0, pd.NA).cumsum()).iloc[-1]
                out[ticker] = Snapshot(
                    ticker=ticker,
                    price=last,
                    timestamp=ts,
                    session=_session_label(ts),
                    volume=float(df["Volume"].iloc[-1]),
                    change_1m_pct=float(chg_1m),
                    change_5m_pct=float(chg_5m),
                    vwap=float(vwap) if pd.notna(vwap) else None,
                )
            except Exception:
                continue
        return out

    def get_bars(
        self,
        tickers: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
        extended_hours: bool = True,
    ) -> dict[str, pd.DataFrame]:
        if not tickers:
            return {}

        tf_map = {
            "1m": ("1m", "7d"),
            "5m": ("5m", "60d"),
            "15m": ("15m", "60d"),
            "1h": ("60m", "730d"),
            "1d": ("1d", "5y"),
        }
        interval, period = tf_map.get(timeframe, ("1m", "7d"))
        try:
            raw = yf.download(
                tickers,
                period=period,
                interval=interval,
                prepost=extended_hours,
                group_by="ticker",
                threads=True,
                progress=False,
                auto_adjust=False,
            )
        except Exception:
            return {}

        data = _normalize_download(raw)
        single = not isinstance(data.columns, pd.MultiIndex)
        out: dict[str, pd.DataFrame] = {}
        for ticker in tickers:
            try:
                df = data if (single and len(tickers) == 1) else data[ticker]
                df = df[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any").tail(limit)
                out[ticker] = df
            except Exception:
                continue
        return out

    async def websocket_stream(self, tickers: list[str]) -> AsyncIterator[dict[str, Any]]:
        while True:
            snapshots = self.get_snapshot(tickers)
            now = datetime.now(US_EASTERN)
            for ticker, snap in snapshots.items():
                yield {
                    "event": "tick",
                    "ticker": ticker,
                    "price": snap.price,
                    "volume": snap.volume,
                    "timestamp": snap.timestamp.isoformat() if snap.timestamp else now.isoformat(),
                    "session": snap.session,
                    "provider": self.name,
                }
            await asyncio.sleep(1.0)

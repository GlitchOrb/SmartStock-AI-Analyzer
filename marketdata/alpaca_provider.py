from __future__ import annotations
from datetime import datetime, timezone
import json
from typing import Any, AsyncIterator
from zoneinfo import ZoneInfo

import pandas as pd
import requests
import websockets

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


def _parse_ts(ts: str | None) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt
    except Exception:
        return datetime.now(timezone.utc)


class AlpacaProvider(MarketDataProvider):
    """Alpaca market data provider (real-time capable)."""

    name = "alpaca"

    def __init__(
        self,
        api_key: str | None = None,
        secret_key: str | None = None,
        data_feed: str | None = None,
    ) -> None:
        import os

        self.api_key = (
            api_key
            or os.environ.get("ALPACA_API_KEY")
            or os.environ.get("API_KEY")
            or os.environ.get("APCA_API_KEY_ID")
            or ""
        )
        self.secret_key = (
            secret_key
            or os.environ.get("ALPACA_SECRET_KEY")
            or os.environ.get("API_SECRET_KEY")
            or os.environ.get("APCA_API_SECRET_KEY")
            or ""
        )
        self.data_feed = (data_feed or os.environ.get("ALPACA_DATA_FEED", "iex")).lower()
        self.rest_base = "https://data.alpaca.markets/v2"
        self.ws_base = f"wss://stream.data.alpaca.markets/v2/{self.data_feed}"

    @property
    def configured(self) -> bool:
        return bool(self.api_key and self.secret_key)

    def _headers(self) -> dict[str, str]:
        if not self.configured:
            raise RuntimeError("Alpaca keys are not configured.")
        return {
            "APCA-API-KEY-ID": self.api_key,
            "APCA-API-SECRET-KEY": self.secret_key,
        }

    def get_universe(self, include_etf: bool = False) -> list[str]:
        return build_universe(UniverseConfig(kind="nasdaq_all", max_size=0, include_etf=include_etf))

    def get_snapshot(self, tickers: list[str]) -> dict[str, Snapshot]:
        out: dict[str, Snapshot] = {}
        if not tickers:
            return out
        if not self.configured:
            raise RuntimeError("AlpacaProvider is not configured with API keys.")

        for i in range(0, len(tickers), 200):
            chunk = tickers[i : i + 200]
            symbols = ",".join(chunk)
            try:
                resp = requests.get(
                    f"{self.rest_base}/stocks/snapshots",
                    params={"symbols": symbols, "feed": self.data_feed},
                    headers=self._headers(),
                    timeout=15,
                )
                resp.raise_for_status()
                payload = resp.json() or {}
                snapshots = payload.get("snapshots", {})
            except Exception:
                continue

            for ticker, obj in snapshots.items():
                try:
                    last_trade = obj.get("latestTrade") or {}
                    minute_bar = obj.get("minuteBar") or {}
                    prev_minute = obj.get("prevMinuteBar") or {}
                    price = float(last_trade.get("p", minute_bar.get("c", 0.0)))
                    ts = _parse_ts(last_trade.get("t"))
                    vwap = minute_bar.get("vw")
                    prev_close = float(prev_minute.get("c", 0.0) or 0.0)
                    chg_1m = ((price - prev_close) / prev_close * 100.0) if prev_close > 0 else 0.0
                    out[ticker] = Snapshot(
                        ticker=ticker,
                        price=price,
                        timestamp=ts,
                        session=_session_label(ts),
                        volume=float(minute_bar.get("v", 0.0) or 0.0),
                        bid=(obj.get("latestQuote") or {}).get("bp"),
                        ask=(obj.get("latestQuote") or {}).get("ap"),
                        change_1m_pct=float(chg_1m),
                        change_5m_pct=0.0,
                        vwap=float(vwap) if vwap is not None else None,
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
        if not self.configured:
            raise RuntimeError("AlpacaProvider is not configured with API keys.")

        tf_map = {
            "1m": "1Min",
            "5m": "5Min",
            "15m": "15Min",
            "1h": "1Hour",
            "1d": "1Day",
        }
        alpaca_tf = tf_map.get(timeframe, "1Min")
        end = end or datetime.now(timezone.utc)
        if start is None:
            start = end - pd.Timedelta(days=30)

        out: dict[str, pd.DataFrame] = {}
        for i in range(0, len(tickers), 80):
            chunk = tickers[i : i + 80]
            params = {
                "symbols": ",".join(chunk),
                "timeframe": alpaca_tf,
                "start": start.isoformat(),
                "end": end.isoformat(),
                "limit": limit,
                "feed": self.data_feed,
                "adjustment": "raw",
            }
            if timeframe != "1d":
                params["asof"] = datetime.now(timezone.utc).date().isoformat()

            try:
                resp = requests.get(
                    f"{self.rest_base}/stocks/bars",
                    params=params,
                    headers=self._headers(),
                    timeout=20,
                )
                resp.raise_for_status()
                payload = resp.json() or {}
                bars_map = payload.get("bars", {})
            except Exception:
                continue

            for ticker, bars in bars_map.items():
                if not bars:
                    continue
                try:
                    df = pd.DataFrame(bars)
                    df = df.rename(columns={"o": "Open", "h": "High", "l": "Low", "c": "Close", "v": "Volume", "t": "Timestamp"})
                    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True)
                    df = df.set_index("Timestamp")[["Open", "High", "Low", "Close", "Volume"]].dropna(how="any")
                    out[ticker] = df
                except Exception:
                    continue
        return out

    async def websocket_stream(self, tickers: list[str]) -> AsyncIterator[dict[str, Any]]:
        if not self.configured:
            raise RuntimeError("AlpacaProvider is not configured with API keys.")
        if not tickers:
            return

        async with websockets.connect(self.ws_base, ping_interval=20, ping_timeout=20, max_size=2_000_000) as ws:
            await ws.send(json.dumps({"action": "auth", "key": self.api_key, "secret": self.secret_key}))
            _ = await ws.recv()
            await ws.send(json.dumps({"action": "subscribe", "trades": tickers, "bars": tickers}))

            while True:
                raw = await ws.recv()
                try:
                    messages = json.loads(raw)
                except Exception:
                    continue
                if not isinstance(messages, list):
                    continue
                for msg in messages:
                    ev = msg.get("T")
                    if ev in {"t", "b"}:
                        ticker = msg.get("S", "")
                        ts = _parse_ts(msg.get("t"))
                        price = float(msg.get("p", msg.get("c", 0.0)) or 0.0)
                        volume = float(msg.get("s", msg.get("v", 0.0)) or 0.0)
                        yield {
                            "event": "tick" if ev == "t" else "bar",
                            "ticker": ticker,
                            "price": price,
                            "volume": volume,
                            "timestamp": ts.isoformat(),
                            "session": _session_label(ts),
                            "provider": self.name,
                        }

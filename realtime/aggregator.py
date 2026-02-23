from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any


@dataclass
class OHLCV:
    ts: datetime
    o: float
    h: float
    l: float
    c: float
    v: float

    def as_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.ts.isoformat(),
            "open": self.o,
            "high": self.h,
            "low": self.l,
            "close": self.c,
            "volume": self.v,
        }


class TickBarAggregator:
    """Maintains rolling tick tape and 1s/1m bars per ticker."""

    def __init__(self, max_ticks: int = 600, max_bars: int = 2000) -> None:
        self.max_ticks = max_ticks
        self.max_bars = max_bars
        self.ticks: dict[str, deque[dict[str, Any]]] = defaultdict(lambda: deque(maxlen=self.max_ticks))
        self.bars_1s: dict[str, deque[OHLCV]] = defaultdict(lambda: deque(maxlen=self.max_bars))
        self.bars_1m: dict[str, deque[OHLCV]] = defaultdict(lambda: deque(maxlen=self.max_bars))

    @staticmethod
    def _floor_to_second(ts: datetime) -> datetime:
        return ts.replace(microsecond=0)

    @staticmethod
    def _floor_to_minute(ts: datetime) -> datetime:
        return ts.replace(second=0, microsecond=0)

    def _upsert_bar(self, bars: deque[OHLCV], bucket_ts: datetime, price: float, volume: float) -> None:
        if not bars or bars[-1].ts != bucket_ts:
            bars.append(OHLCV(ts=bucket_ts, o=price, h=price, l=price, c=price, v=volume))
            return
        bar = bars[-1]
        bar.h = max(bar.h, price)
        bar.l = min(bar.l, price)
        bar.c = price
        bar.v += volume

    def update(self, event: dict[str, Any]) -> None:
        ticker = str(event.get("ticker", "")).upper()
        if not ticker:
            return
        price = float(event.get("price", 0.0) or 0.0)
        volume = float(event.get("volume", 0.0) or 0.0)
        if price <= 0:
            return

        ts_raw = event.get("timestamp")
        try:
            ts = datetime.fromisoformat(str(ts_raw).replace("Z", "+00:00"))
        except Exception:
            ts = datetime.now(timezone.utc)

        self.ticks[ticker].append(
            {
                "timestamp": ts.isoformat(),
                "price": price,
                "volume": volume,
                "session": event.get("session", "UNKNOWN"),
                "event": event.get("event", "tick"),
            }
        )

        sec_bucket = self._floor_to_second(ts)
        min_bucket = self._floor_to_minute(ts)
        self._upsert_bar(self.bars_1s[ticker], sec_bucket, price, volume)
        self._upsert_bar(self.bars_1m[ticker], min_bucket, price, volume)

    def get_tape(self, ticker: str, n: int = 100) -> list[dict[str, Any]]:
        ticker = ticker.upper()
        return list(self.ticks[ticker])[-n:]

    def get_bars(self, ticker: str, timeframe: str = "1s", n: int = 300) -> list[dict[str, Any]]:
        ticker = ticker.upper()
        bars = self.bars_1s[ticker] if timeframe == "1s" else self.bars_1m[ticker]
        return [b.as_dict() for b in list(bars)[-n:]]


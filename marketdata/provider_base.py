from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import Any, AsyncIterator

import pandas as pd


@dataclass
class Snapshot:
    ticker: str
    price: float
    timestamp: datetime
    session: str
    volume: float = 0.0
    bid: float | None = None
    ask: float | None = None
    change_1m_pct: float = 0.0
    change_5m_pct: float = 0.0
    vwap: float | None = None


class MarketDataProvider(ABC):
    """Abstract market data provider interface."""

    name: str = "base"

    @abstractmethod
    def get_universe(self, include_etf: bool = False) -> list[str]:
        """Return provider-supported universe."""

    @abstractmethod
    def get_snapshot(self, tickers: list[str]) -> dict[str, Snapshot]:
        """Fetch latest snapshot metrics for symbols."""

    @abstractmethod
    def get_bars(
        self,
        tickers: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
        extended_hours: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """Fetch OHLCV bars by ticker."""

    @abstractmethod
    async def websocket_stream(self, tickers: list[str]) -> AsyncIterator[dict[str, Any]]:
        """Yield real-time events from websocket."""

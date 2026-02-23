from __future__ import annotations

from datetime import datetime
from typing import Any, AsyncIterator

import pandas as pd

from marketdata.provider_base import MarketDataProvider, Snapshot


class PolygonProvider(MarketDataProvider):
    """Polygon provider stub."""

    name = "polygon"

    def get_universe(self, include_etf: bool = False) -> list[str]:
        raise NotImplementedError("PolygonProvider is optional and not configured in this build.")

    def get_snapshot(self, tickers: list[str]) -> dict[str, Snapshot]:
        raise NotImplementedError("PolygonProvider is optional and not configured in this build.")

    def get_bars(
        self,
        tickers: list[str],
        timeframe: str,
        start: datetime | None = None,
        end: datetime | None = None,
        limit: int = 500,
        extended_hours: bool = True,
    ) -> dict[str, pd.DataFrame]:
        raise NotImplementedError("PolygonProvider is optional and not configured in this build.")

    async def websocket_stream(self, tickers: list[str]) -> AsyncIterator[dict[str, Any]]:
        raise NotImplementedError("PolygonProvider is optional and not configured in this build.")

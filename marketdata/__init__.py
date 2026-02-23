from __future__ import annotations

import os

try:
    from dotenv import load_dotenv
    load_dotenv(override=False)
except Exception:
    pass

from marketdata.alpaca_provider import AlpacaProvider
from marketdata.polygon_provider import PolygonProvider
from marketdata.provider_base import MarketDataProvider
from marketdata.yfinance_provider import YFinanceProvider


def get_provider() -> MarketDataProvider:
    provider_name = os.environ.get("MARKET_DATA_PROVIDER", "alpaca").lower()
    if provider_name == "alpaca":
        ap = AlpacaProvider()
        if ap.configured:
            return ap
        return YFinanceProvider()
    if provider_name == "yfinance":
        return YFinanceProvider()
    if provider_name == "polygon":
        try:
            return PolygonProvider()
        except Exception:
            return YFinanceProvider()
    return YFinanceProvider()

"""
SmartStock AI Analyzer — DataAgent
Fetches market data from yfinance. No Gemini calls.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import yfinance as yf
import pandas as pd

from agents.base import BaseAgent
from schemas.agents import (
    DataAgentOutput,
    Fundamentals,
    PriceSnapshot,
    VolumeInfo,
)
from schemas.enums import Sector
from schemas.config import settings
from utils.cache import cache_get, cache_set
from utils.helpers import retry
from utils.logger import log_agent


_SECTOR_MAP: dict[str, Sector] = {
    "technology": Sector.TECHNOLOGY,
    "healthcare": Sector.HEALTHCARE,
    "financial services": Sector.FINANCE,
    "financial": Sector.FINANCE,
    "energy": Sector.ENERGY,
    "consumer cyclical": Sector.CONSUMER,
    "consumer defensive": Sector.CONSUMER,
    "industrials": Sector.INDUSTRIAL,
    "real estate": Sector.REAL_ESTATE,
    "utilities": Sector.UTILITIES,
    "basic materials": Sector.MATERIALS,
    "communication services": Sector.TELECOM,
}


class DataAgent(BaseAgent):
    """Fetches price, volume, fundamentals, and history via yfinance."""

    name = "DataAgent"

    @retry(max_attempts=3, delay=1.0)
    def run(self, ticker: str) -> DataAgentOutput:
        log_agent(self.name, f"Fetching data for [bold]{ticker}[/bold]")

        # Check cache
        cache_key = f"data_{ticker}"
        cached = cache_get(cache_key)
        if cached:
            log_agent(self.name, "Cache hit ✓")
            return DataAgentOutput.model_validate(cached)

        # Fetch from yfinance
        stock = yf.Ticker(ticker)
        info = stock.info or {}

        # Price
        price = PriceSnapshot(
            current=info.get("currentPrice") or info.get("regularMarketPrice", 0),
            open=info.get("regularMarketOpen", 0),
            high_52w=info.get("fiftyTwoWeekHigh", 0),
            low_52w=info.get("fiftyTwoWeekLow", 0),
            change_pct=self._calc_change_pct(info),
        )

        # Volume
        cur_vol = info.get("regularMarketVolume") or info.get("volume", 0)
        avg_10 = info.get("averageDailyVolume10Day", cur_vol or 1)
        avg_3m = info.get("averageVolume", cur_vol or 1)
        volume = VolumeInfo(
            current=cur_vol,
            avg_10d=avg_10,
            avg_3m=avg_3m,
            relative_volume=round(cur_vol / avg_10, 2) if avg_10 else 1.0,
        )

        # Fundamentals
        fundamentals = Fundamentals(
            market_cap=info.get("marketCap"),
            pe_ratio=info.get("trailingPE"),
            forward_pe=info.get("forwardPE"),
            peg_ratio=info.get("pegRatio"),
            ps_ratio=info.get("priceToSalesTrailing12Months"),
            pb_ratio=info.get("priceToBook"),
            dividend_yield=info.get("dividendYield"),
            eps=info.get("trailingEps"),
            revenue=info.get("totalRevenue"),
            profit_margin=info.get("profitMargins"),
            debt_to_equity=info.get("debtToEquity"),
            roe=info.get("returnOnEquity"),
            free_cash_flow=info.get("freeCashflow"),
            beta=info.get("beta"),
        )

        # History → CSV
        hist = stock.history(period="1y")
        csv_path = settings.cache_dir / f"{ticker}_history.csv"
        hist.to_csv(csv_path)

        # Sector mapping
        raw_sector = (info.get("sector") or "").lower()
        sector = _SECTOR_MAP.get(raw_sector, Sector.OTHER)

        output = DataAgentOutput(
            ticker=ticker.upper(),
            company_name=info.get("shortName") or info.get("longName", ticker),
            sector=sector,
            exchange=info.get("exchange", ""),
            currency=info.get("currency", "USD"),
            price=price,
            volume=volume,
            fundamentals=fundamentals,
            history_df_path=str(csv_path),
            fetched_at=datetime.now(),
        )

        # Cache the result
        cache_set(cache_key, output.model_dump(mode="json"))

        log_agent(self.name, f"[green]{ticker} data fetched ✓[/green]")
        return output

    @staticmethod
    def _calc_change_pct(info: dict) -> float:
        prev = info.get("regularMarketPreviousClose", 0)
        curr = info.get("currentPrice") or info.get("regularMarketPrice", 0)
        if prev and curr:
            return round(((curr - prev) / prev) * 100, 2)
        return 0.0

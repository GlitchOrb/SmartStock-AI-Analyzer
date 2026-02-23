from __future__ import annotations

from data.screener import MarketScreener
from schemas.agents import ScreenerResult


class AgentScreener:
    """Agent wrapper around the NASDAQ_ALL screener."""

    def __init__(self) -> None:
        self.engine = MarketScreener(
            tickers=None,
            universe_kind="nasdaq_all",
            universe_size=0,
            liquidity_threshold=100_000,
            min_dollar_volume_20d=5_000_000.0,
            min_price=1.0,
            target_universe_min=500,
            target_universe_max=1200,
        )

    def run(self) -> list[ScreenerResult]:
        return self.engine.run(send_telegram=False)

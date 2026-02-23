"""Data agent.

Fetches market/fundamental/technical data and returns DataAgentOutput.
"""

from __future__ import annotations

from agents.base import BaseAgent
from data.fetcher import fetch_all
from schemas.agents import DataAgentOutput
from utils.cache import cache_get, cache_set
from utils.helpers import retry
from utils.logger import log_agent


class DataAgent(BaseAgent):
    """Fetches market data using the unified data fetch pipeline."""

    name = "DataAgent"

    @retry(max_attempts=3, delay=1.0)
    def run(self, ticker: str) -> DataAgentOutput:
        normalized = ticker.strip().upper()
        if not normalized:
            raise ValueError("Ticker must not be empty.")

        cache_key = f"data_{normalized}"
        cached = cache_get(cache_key)
        if cached:
            log_agent(self.name, f"Cache hit: {normalized}")
            return DataAgentOutput.model_validate(cached)

        output = fetch_all(normalized, period="1y")
        cache_set(cache_key, output.model_dump(mode="json"))

        log_agent(self.name, f"[green]Data fetched: {normalized}[/green]")
        return output

"""
SmartStock AI Analyzer — Shared Helpers
"""

from __future__ import annotations

import functools
import time
from datetime import datetime
from typing import Callable, TypeVar

from utils.logger import log_agent

T = TypeVar("T")


def retry(max_attempts: int = 3, delay: float = 1.0):
    """Decorator: retry a function up to max_attempts with exponential backoff."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args, **kwargs) -> T:
            last_error: Exception | None = None
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_error = e
                    if attempt < max_attempts:
                        wait = delay * (2 ** (attempt - 1))
                        log_agent("Retry", f"{func.__name__} attempt {attempt} failed, waiting {wait:.1f}s")
                        time.sleep(wait)
            raise last_error  # type: ignore[misc]

        return wrapper
    return decorator


def fmt_number(value: float | int | None, decimals: int = 2) -> str:
    """Format a number with commas. Returns 'N/A' for None."""
    if value is None:
        return "N/A"
    if abs(value) >= 1_000_000_000_000:
        return f"${value / 1_000_000_000_000:,.{decimals}f}T"
    if abs(value) >= 1_000_000_000:
        return f"${value / 1_000_000_000:,.{decimals}f}B"
    if abs(value) >= 1_000_000:
        return f"${value / 1_000_000:,.{decimals}f}M"
    return f"{value:,.{decimals}f}"


def fmt_pct(value: float | None, decimals: int = 2) -> str:
    """Format a percentage. Returns 'N/A' for None."""
    if value is None:
        return "N/A"
    sign = "+" if value > 0 else ""
    return f"{sign}{value:.{decimals}f}%"


def fmt_date(dt: datetime | None, fmt: str = "%Y-%m-%d %H:%M") -> str:
    """Format a datetime. Returns 'N/A' for None."""
    if dt is None:
        return "N/A"
    return dt.strftime(fmt)


def signal_color(signal_value: str) -> str:
    """Map a Signal enum value to a hex color for UI rendering."""
    colors = {
        "Strong Buy": "#00C853",
        "Buy": "#4CAF50",
        "Hold": "#FFC107",
        "Sell": "#FF5722",
        "Strong Sell": "#D32F2F",
    }
    return colors.get(signal_value, "#9E9E9E")


def sentiment_color(sentiment_value: str) -> str:
    """Map a SentimentLabel value to a hex color."""
    colors = {
        "Very Positive": "#00C853",
        "Positive": "#66BB6A",
        "Neutral": "#90A4AE",
        "Negative": "#FF7043",
        "Very Negative": "#D32F2F",
    }
    return colors.get(sentiment_value, "#9E9E9E")

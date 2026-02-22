"""
SmartStock AI Analyzer — Chart Renderers
Generates PNG charts for embedding in PDF reports.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np

from schemas.config import settings
from utils.logger import log_agent


def _chart_dir() -> Path:
    d = settings.cache_dir / "charts"
    d.mkdir(parents=True, exist_ok=True)
    return d


def render_price_chart(
    csv_path: str,
    ticker: str,
    support_levels: list[float] | None = None,
    resistance_levels: list[float] | None = None,
) -> str | None:
    """Render a price chart with volume bars and S/R levels. Returns PNG path."""
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"], index_col="Date")
    except Exception:
        log_agent("Charts", "Failed to read history CSV")
        return None

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(10, 6), gridspec_kw={"height_ratios": [3, 1]}, sharex=True
    )
    fig.patch.set_facecolor("#1a1a2e")

    # Price line
    ax1.plot(df.index, df["Close"], color="#448AFF", linewidth=1.5, label="Close")
    ax1.fill_between(df.index, df["Close"], alpha=0.1, color="#448AFF")

    # Support / Resistance lines
    if support_levels:
        for s in support_levels[:3]:
            ax1.axhline(y=s, color="#00C853", linestyle="--", alpha=0.6, linewidth=0.8)
    if resistance_levels:
        for r in resistance_levels[:3]:
            ax1.axhline(y=r, color="#D32F2F", linestyle="--", alpha=0.6, linewidth=0.8)

    ax1.set_facecolor("#1a1a2e")
    ax1.set_ylabel("Price ($)", color="#aaa", fontsize=9)
    ax1.tick_params(colors="#aaa", labelsize=8)
    ax1.legend(loc="upper left", fontsize=8, facecolor="#1a1a2e", edgecolor="#333", labelcolor="#aaa")
    ax1.set_title(f"{ticker} — 1 Year Price Chart", color="#fff", fontsize=12, pad=10)
    ax1.grid(True, alpha=0.15, color="#444")

    # Volume bars
    colors = ["#00C853" if c >= o else "#D32F2F" for c, o in zip(df["Close"], df["Open"])]
    ax2.bar(df.index, df["Volume"], color=colors, alpha=0.7, width=1.5)
    ax2.set_facecolor("#1a1a2e")
    ax2.set_ylabel("Volume", color="#aaa", fontsize=9)
    ax2.tick_params(colors="#aaa", labelsize=8)
    ax2.grid(True, alpha=0.15, color="#444")

    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    fig.autofmt_xdate()
    plt.tight_layout()

    out_path = _chart_dir() / f"{ticker}_price.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    log_agent("Charts", f"Price chart saved → {out_path.name}")
    return str(out_path)


def render_signal_gauge(signal: str, ticker: str) -> str | None:
    """Render a simple signal gauge chart. Returns PNG path."""
    signal_map = {"Strong Sell": 0, "Sell": 1, "Hold": 2, "Buy": 3, "Strong Buy": 4}
    colors_map = ["#D32F2F", "#FF5722", "#FFC107", "#4CAF50", "#00C853"]

    val = signal_map.get(signal, 2)

    fig, ax = plt.subplots(figsize=(6, 1.5))
    fig.patch.set_facecolor("#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw the gauge bar
    for i in range(5):
        alpha = 1.0 if i == val else 0.25
        ax.barh(0, 1, left=i, height=0.6, color=colors_map[i], alpha=alpha, edgecolor="#333")

    labels = ["Strong\nSell", "Sell", "Hold", "Buy", "Strong\nBuy"]
    for i, label in enumerate(labels):
        color = "#fff" if i == val else "#666"
        ax.text(i + 0.5, -0.6, label, ha="center", va="top", fontsize=7, color=color, fontweight="bold")

    ax.set_xlim(0, 5)
    ax.set_ylim(-1.2, 0.8)
    ax.axis("off")
    ax.set_title(f"{ticker} Signal", color="#fff", fontsize=10, pad=8)
    plt.tight_layout()

    out_path = _chart_dir() / f"{ticker}_gauge.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#1a1a2e")
    plt.close(fig)
    log_agent("Charts", f"Signal gauge saved → {out_path.name}")
    return str(out_path)

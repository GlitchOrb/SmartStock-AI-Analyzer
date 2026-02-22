"""
SmartStock AI Analyzer — Agent Output Schemas (Pydantic v2)
Stage 3: Added RAG-backed ResearchAgentOutput + citation-based SentimentAgentOutput.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

from pydantic import BaseModel, Field

from schemas.enums import ReportDepth, Sector, SentimentLabel, Signal


# ──────────────────────────────────────────────
# DataAgent sub-models
# ──────────────────────────────────────────────

class PriceSnapshot(BaseModel):
    """Current price metrics."""
    current: float = 0.0
    open: float = 0.0
    high_52w: float = 0.0
    low_52w: float = 0.0
    change_pct: float = Field(default=0.0, description="Daily percentage change")


class VolumeInfo(BaseModel):
    """Volume statistics."""
    current: int = 0
    avg_10d: int = 0
    avg_3m: int = 0
    relative_volume: float = Field(default=1.0, description="current / avg_10d")


class Fundamentals(BaseModel):
    """Key fundamental metrics."""
    market_cap: float | None = None
    per: float | None = None
    pbr: float | None = None
    roe: float | None = None
    debt_ratio: float | None = None
    revenue: float | None = None
    operating_profit: float | None = None
    eps: float | None = None
    dividend_yield: float | None = None
    profit_margin: float | None = None
    beta: float | None = None


class Technicals(BaseModel):
    """Computed technical indicators."""
    rsi_14: float | None = None
    macd: float | None = None
    macd_signal: float | None = None
    macd_histogram: float | None = None
    bb_upper: float | None = None
    bb_middle: float | None = None
    bb_lower: float | None = None
    ma_20: float | None = None
    ma_60: float | None = None
    mdd: float | None = None
    volatility: float | None = None


class DataAgentOutput(BaseModel):
    """Output from data fetcher — pure yfinance, no Gemini call."""
    ticker: str
    company_name: str = ""
    sector: Sector = Sector.OTHER
    exchange: str = ""
    currency: str = "USD"
    price: PriceSnapshot = Field(default_factory=PriceSnapshot)
    volume: VolumeInfo = Field(default_factory=VolumeInfo)
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)
    technicals: Technicals = Field(default_factory=Technicals)
    history_df_path: str = Field(default="", description="Path to cached OHLCV CSV")
    anomalies: list[str] = Field(default_factory=list, description="Data quality flags")
    fetched_at: datetime = Field(default_factory=datetime.now)


# ──────────────────────────────────────────────
# ResearchAgent (RAG-backed)
# ──────────────────────────────────────────────

class TimelineEvent(BaseModel):
    """A dated event from research."""
    date: str = ""
    event: str = ""


class ResearchAgentOutput(BaseModel):
    """Output from ResearchAgent — RAG + Gemini."""
    ticker: str
    key_themes: list[str] = Field(default_factory=list, description="Key research themes")
    timeline: list[TimelineEvent] = Field(default_factory=list, description="Event timeline")
    data_quality: str = Field(default="insufficient", description="sufficient|insufficient")
    news_count: int = Field(default=0, description="Number of news articles found")
    retrieved_chunks: list[dict] = Field(default_factory=list, description="Top-k RAG chunks")
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


# ──────────────────────────────────────────────
# SentimentAgent (citation-based)
# ──────────────────────────────────────────────

class Citation(BaseModel):
    """A news citation backing a sentiment claim."""
    text: str = ""
    source: str = ""
    url: str = ""
    timestamp: str = ""


class SentimentAgentOutput(BaseModel):
    """Output from SentimentAgent — Standard & Deep only."""
    ticker: str
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_label: str = Field(default="neutral", description="positive|neutral|negative")
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


# ──────────────────────────────────────────────
# AnalysisAgent — Scenario-based (bull/base/bear)
# ──────────────────────────────────────────────

class BullCase(BaseModel):
    """Bull case scenario."""
    thesis: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class BaseCase(BaseModel):
    """Base case scenario."""
    thesis: str = ""
    drivers: list[str] = Field(default_factory=list)


class BearCase(BaseModel):
    """Bear case scenario."""
    thesis: str = ""
    risks: list[str] = Field(default_factory=list)
    warning: str = ""


class AnalysisAgentOutput(BaseModel):
    """Output from AnalysisAgent — scenario-based analysis."""
    ticker: str
    bull_case: BullCase = Field(default_factory=BullCase)
    base_case: BaseCase = Field(default_factory=BaseCase)
    bear_case: BearCase = Field(default_factory=BearCase)
    key_drivers: list[str] = Field(default_factory=list)
    rating: str = Field(default="", description="Buy|Hold|Sell (populated in Quick mode)")
    confidence: float = Field(default=0.0, description="0-100 (populated in Quick mode)")
    rationale: list[str] = Field(default_factory=list, description="Populated in Quick mode")
    invalidation_triggers: list[str] = Field(default_factory=list, description="Populated in Quick mode")
    generated_at: datetime = Field(default_factory=datetime.now)


# ──────────────────────────────────────────────
# RecommendationAgent
# ──────────────────────────────────────────────

class RecommendationAgentOutput(BaseModel):
    """Output from RecommendationAgent — Standard/Deep mode only."""
    ticker: str
    rating: str = Field(default="Hold", description="Buy|Hold|Sell")
    confidence: float = Field(default=0.0, description="0-100")
    rationale: list[str] = Field(default_factory=list)
    invalidation_triggers: list[str] = Field(default_factory=list)
    risk_notes: str = Field(default="")
    generated_at: datetime = Field(default_factory=datetime.now)


# ──────────────────────────────────────────────
# ReportAgent
# ──────────────────────────────────────────────

class ReportSection(BaseModel):
    """A section of the final report."""
    title: str
    content: str = Field(default="", description="Markdown-formatted section body")


class ReportAgentOutput(BaseModel):
    """Output from ReportAgent — orchestrator + PDF generation."""
    ticker: str
    report_depth: ReportDepth = ReportDepth.STANDARD
    executive_summary: str = Field(default="", description="5-7 sentence verdict")
    sections: list[ReportSection] = Field(default_factory=list)
    pdf_bytes: bytes | None = Field(default=None, description="Generated PDF bytes")
    markdown_report: str = Field(default="", description="Full markdown report text")
    gemini_calls_used: int = 0
    total_tokens_used: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)
    generation_time_s: float = 0.0

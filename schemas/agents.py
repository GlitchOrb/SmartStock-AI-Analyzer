"""Pydantic schemas shared across the SmartStock AI Analyzer project."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from schemas.enums import ReportDepth, Sector, Signal


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
    """Output from the market data fetcher."""

    ticker: str
    company_name: str = ""
    sector: Sector = Sector.OTHER
    exchange: str = ""
    currency: str = "USD"
    price: PriceSnapshot = Field(default_factory=PriceSnapshot)
    volume: VolumeInfo = Field(default_factory=VolumeInfo)
    fundamentals: Fundamentals = Field(default_factory=Fundamentals)
    technicals: Technicals = Field(default_factory=Technicals)
    history_df_path: str | None = Field(default=None, description="Path to cached OHLCV CSV")
    anomalies: list[str] = Field(default_factory=list, description="Data quality flags")
    fetched_at: datetime = Field(default_factory=datetime.now)


class TimelineEvent(BaseModel):
    """A dated event from research."""

    date: str = ""
    event: str = ""


class ResearchAgentOutput(BaseModel):
    """Output from ResearchAgent."""

    ticker: str
    key_themes: list[str] = Field(default_factory=list)
    timeline: list[TimelineEvent] = Field(default_factory=list)
    data_quality: str = Field(default="insufficient", description="sufficient|insufficient")
    news_count: int = 0
    retrieved_chunks: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class Citation(BaseModel):
    """A news citation backing a sentiment claim."""

    text: str = ""
    source: str = ""
    url: str = ""
    timestamp: str = ""


class SentimentAgentOutput(BaseModel):
    """Output from SentimentAgent."""

    ticker: str
    sentiment_score: float = Field(default=0.0, ge=-1.0, le=1.0)
    sentiment_label: str = Field(default="neutral", description="positive|neutral|negative")
    pros: list[str] = Field(default_factory=list)
    cons: list[str] = Field(default_factory=list)
    citations: list[Citation] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class BullCase(BaseModel):
    thesis: str = ""
    catalysts: list[str] = Field(default_factory=list)
    risks: list[str] = Field(default_factory=list)


class BaseCase(BaseModel):
    thesis: str = ""
    drivers: list[str] = Field(default_factory=list)


class BearCase(BaseModel):
    thesis: str = ""
    risks: list[str] = Field(default_factory=list)
    warning: str = ""


class AnalysisAgentOutput(BaseModel):
    """Output from AnalysisAgent."""

    ticker: str
    bull_case: BullCase = Field(default_factory=BullCase)
    base_case: BaseCase = Field(default_factory=BaseCase)
    bear_case: BearCase = Field(default_factory=BearCase)
    key_drivers: list[str] = Field(default_factory=list)
    rating: str = Field(default="", description="Buy|Hold|Sell")
    confidence: float = Field(default=0.0, description="0-100")
    rationale: list[str] = Field(default_factory=list)
    invalidation_triggers: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class RecommendationAgentOutput(BaseModel):
    """Output from RecommendationAgent."""

    ticker: str
    rating: str = Field(default="Hold", description="Buy|Hold|Sell")
    confidence: float = 0.0
    rationale: list[str] = Field(default_factory=list)
    invalidation_triggers: list[str] = Field(default_factory=list)
    risk_notes: str = ""
    generated_at: datetime = Field(default_factory=datetime.now)


class ReportSection(BaseModel):
    title: str
    content: str = ""


class ReportAgentOutput(BaseModel):
    """Output from ReportAgent orchestrator."""

    ticker: str
    report_depth: ReportDepth = ReportDepth.STANDARD
    executive_summary: str = ""
    sections: list[ReportSection] = Field(default_factory=list)
    pdf_bytes: bytes | None = None
    markdown_report: str = ""
    gemini_calls_used: int = 0
    total_tokens_used: int = 0
    generated_at: datetime = Field(default_factory=datetime.now)
    generation_time_s: float = 0.0


class TradePlan(BaseModel):
    """Rule-based trade setup (Entry / SL / TP)."""

    strategy_name: str = "ATR Momentum"
    entry_price: float
    stop_loss: float
    target_price_1: float
    target_price_2: float
    risk_reward_ratio: float
    suggested_position_size_pct: float | None = None
    rationale: str
    invalidation_triggers: list[str] = Field(default_factory=list)
    generated_at: datetime = Field(default_factory=datetime.now)


class CatalystEvent(BaseModel):
    """Upcoming / recent catalyst event."""

    event_type: str
    date: str
    description: str
    impact_score: float = 0.0
    credibility_score: float = 0.0
    source_name: str | None = None
    source_url: str | None = None
    duplicate_mentions: int = 1


class ScreenerAlert(BaseModel):
    """Automated alert for significant score changes."""

    ticker: str = ""
    old_score: float = 0.0
    new_score: float = 0.0
    score_change: float = 0.0
    key_signals_changed: list[str] = Field(default_factory=list)
    alert_triggered: bool = False
    alert_text: str = ""
    triggered: bool = False
    message: str = ""
    timestamp: datetime = Field(default_factory=datetime.now)


class UICard(BaseModel):
    """Data payload for dashboard cards."""

    title: str
    subtitle: str
    key_metrics: dict[str, Any] = Field(default_factory=dict)
    detail_link: str = ""


class ScreenerResult(BaseModel):
    """Result payload for momentum scanner rows/cards/details."""

    ticker: str
    company_name: str = ""
    current_price: float
    change_pct_1d: float
    volume_relative: float
    average_volume_20d: float = 0.0
    breakout_resistance_20d: float = 0.0
    rsi_14: float
    macd_line: float = 0.0
    macd_signal: float = 0.0
    macd_histogram: float = 0.0
    adx_14: float = 0.0
    obv_slope_10d: float = 0.0
    stochastic_k: float = 0.0
    stochastic_d: float = 0.0
    ema_20: float = 0.0
    ema_50: float = 0.0
    ema_120: float = 0.0
    atr_14: float = 0.0
    signal: Signal = Signal.HOLD
    momentum_category: str = "Neutral"
    indicators: list[str] = Field(default_factory=list)
    indicator_values: dict[str, Any] = Field(default_factory=dict)
    signals: dict[str, Any] = Field(default_factory=dict)

    trade_plan: TradePlan | None = None
    catalysts: list[CatalystEvent] = Field(default_factory=list)

    score: float = 0.0
    composite_score: float = 0.0
    key_positive_signals: list[str] = Field(default_factory=list)
    rationale: str = ""
    signal_flags: list[dict[str, Any]] = Field(default_factory=list)
    strategy_label: str | None = None
    strategy_rationale: str | None = None
    alert: ScreenerAlert | None = None
    telegram_message: str = ""
    ui_card: UICard | None = None
    generated_at: datetime = Field(default_factory=datetime.now)

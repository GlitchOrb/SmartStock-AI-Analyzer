"""
SmartStock AI Analyzer — Reusable UI Components
Cards, gauges, charts, and metric displays for the Streamlit dashboard.
"""

from __future__ import annotations

import streamlit as st

from schemas.agents import (
    AnalysisAgentOutput,
    DataAgentOutput,
    RecommendationAgentOutput,
    ReportAgentOutput,
    SentimentAgentOutput,
)
from utils.helpers import fmt_number, fmt_pct, signal_color, sentiment_color


def render_header(data: DataAgentOutput) -> None:
    """Render the stock header with price and basic info."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(
            f"""
            <h2 style="margin:0;">{data.company_name}</h2>
            <span style="color:#888; font-size:0.9rem;">
                {data.ticker} · {data.sector.value} · {data.exchange}
            </span>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        change_color = "#00C853" if data.price.change_pct >= 0 else "#D32F2F"
        st.metric(
            "Price",
            f"${data.price.current:.2f}",
            f"{data.price.change_pct:+.2f}%",
        )

    with col3:
        st.metric("Market Cap", fmt_number(data.fundamentals.market_cap))

    with col4:
        st.metric("Volume", f"{data.volume.current:,}", f"{data.volume.relative_volume:.1f}x avg")


def render_signal_card(recommendation: RecommendationAgentOutput) -> None:
    """Render the main signal card."""
    color = signal_color(recommendation.signal.value)
    st.markdown(
        f"""
        <div style="
            background: linear-gradient(135deg, {color}22, {color}11);
            border: 2px solid {color}44;
            border-radius: 16px;
            padding: 1.5rem;
            text-align: center;
            margin: 1rem 0;
        ">
            <div style="font-size: 0.85rem; color: #888; margin-bottom: 0.3rem;">
                AI Recommendation
            </div>
            <div style="font-size: 2.2rem; font-weight: 700; color: {color};">
                {recommendation.signal.value}
            </div>
            <div style="font-size: 0.95rem; color: #aaa; margin-top: 0.3rem;">
                Confidence: {recommendation.confidence:.0%}
                {f' · Target: ${recommendation.target_price:.2f}' if recommendation.target_price else ''}
            </div>
            <div style="font-size: 0.85rem; color: #999; margin-top: 0.5rem;">
                {recommendation.time_horizon}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_analysis_section(analysis: AnalysisAgentOutput) -> None:
    """Render technical and fundamental analysis."""
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Technical Indicators")
        for ind in analysis.technical_indicators:
            color = signal_color(ind.signal.value)
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:4px 0; border-bottom:1px solid #333;'>"
                f"<span>{ind.name}</span>"
                f"<span style='color:{color}; font-weight:600;'>{ind.signal.value}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with col2:
        st.markdown("#### 📋 Fundamental Verdicts")
        for v in analysis.fundamental_verdicts:
            assessment_color = {
                "Undervalued": "#00C853",
                "Fairly Valued": "#FFC107",
                "Overvalued": "#D32F2F",
            }.get(v.assessment, "#888")
            st.markdown(
                f"<div style='display:flex; justify-content:space-between; "
                f"padding:4px 0; border-bottom:1px solid #333;'>"
                f"<span>{v.metric}</span>"
                f"<span style='color:{assessment_color}; font-weight:600;'>{v.assessment}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown(f"**Combined Signal:** {analysis.combined_signal.value}")
    if analysis.summary:
        st.info(analysis.summary)


def render_sentiment_section(sentiment: SentimentAgentOutput) -> None:
    """Render sentiment analysis results."""
    color = sentiment_color(sentiment.overall_sentiment.value)
    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0;">
            <span style="font-size:1.5rem; color:{color}; font-weight:700;">
                {sentiment.overall_sentiment.value}
            </span>
            <span style="color:#888;">
                Score: {sentiment.score:+.2f} · Confidence: {sentiment.confidence:.0%}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if sentiment.summary:
        st.write(sentiment.summary)

    if sentiment.news_items:
        st.markdown("**Recent News:**")
        for item in sentiment.news_items[:5]:
            icon = {"Very Positive": "🟢", "Positive": "🟢", "Neutral": "⚪", "Negative": "🔴", "Very Negative": "🔴"}
            st.markdown(f"{icon.get(item.sentiment.value, '⚪')} {item.headline}")


def render_report_meta(report: ReportAgentOutput) -> None:
    """Render report metadata and download button."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gemini Calls", report.gemini_calls_used)
    with col2:
        st.metric("Tokens Used", f"{report.total_tokens_used:,}")
    with col3:
        st.metric("Generation Time", f"{report.generation_time_s:.1f}s")

    if report.pdf_path:
        try:
            with open(report.pdf_path, "rb") as f:
                st.download_button(
                    "📥 Download PDF Report",
                    data=f.read(),
                    file_name=f"{report.ticker}_{report.report_depth.value}_report.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
        except FileNotFoundError:
            st.warning("PDF file not found.")

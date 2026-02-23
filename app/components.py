"""Reusable Streamlit UI components."""

from __future__ import annotations

import streamlit as st

from schemas.agents import (
    AnalysisAgentOutput,
    DataAgentOutput,
    RecommendationAgentOutput,
    ReportAgentOutput,
    SentimentAgentOutput,
)
from utils.helpers import fmt_number, sentiment_color, signal_color


def render_header(data: DataAgentOutput) -> None:
    """Render the stock header with price and basic info."""
    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

    with col1:
        st.markdown(
            f"""
            <h2 style="margin:0;">{data.company_name}</h2>
            <span style="color:#888; font-size:0.9rem;">
                {data.ticker} 쨌 {data.sector.value} 쨌 {data.exchange}
            </span>
            """,
            unsafe_allow_html=True,
        )

    with col2:
        st.metric("Price", f"${data.price.current:.2f}", f"{data.price.change_pct:+.2f}%")

    with col3:
        st.metric("Market Cap", fmt_number(data.fundamentals.market_cap))

    with col4:
        st.metric("Volume", f"{data.volume.current:,}", f"{data.volume.relative_volume:.1f}x avg")


def render_signal_card(recommendation: RecommendationAgentOutput) -> None:
    """Render the main recommendation card."""
    color = signal_color(recommendation.rating)
    confidence_pct = _confidence_to_percent(recommendation.confidence)

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
                {recommendation.rating}
            </div>
            <div style="font-size: 0.95rem; color: #aaa; margin-top: 0.3rem;">
                Confidence: {confidence_pct:.0f}%
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if recommendation.rationale:
        st.caption(" / ".join(recommendation.rationale[:3]))


def render_analysis_section(analysis: AnalysisAgentOutput) -> None:
    """Render scenario-based analysis."""
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### Bull")
        st.write(analysis.bull_case.thesis or "N/A")
        if analysis.bull_case.catalysts:
            st.markdown("**Catalysts**")
            for item in analysis.bull_case.catalysts[:5]:
                st.markdown(f"- {item}")
        if analysis.bull_case.risks:
            st.markdown("**Risks**")
            for item in analysis.bull_case.risks[:5]:
                st.markdown(f"- {item}")

    with col2:
        st.markdown("#### Base")
        st.write(analysis.base_case.thesis or "N/A")
        if analysis.base_case.drivers:
            st.markdown("**Drivers**")
            for item in analysis.base_case.drivers[:6]:
                st.markdown(f"- {item}")

    with col3:
        st.markdown("#### Bear")
        st.write(analysis.bear_case.thesis or "N/A")
        if analysis.bear_case.risks:
            st.markdown("**Risks**")
            for item in analysis.bear_case.risks[:5]:
                st.markdown(f"- {item}")
        if analysis.bear_case.warning:
            st.warning(analysis.bear_case.warning)

    st.markdown("#### Key Drivers")
    if analysis.key_drivers:
        for item in analysis.key_drivers:
            st.markdown(f"- {item}")
    else:
        st.info("No key drivers generated.")


def render_sentiment_section(sentiment: SentimentAgentOutput) -> None:
    """Render sentiment analysis results."""
    normalized_label = sentiment.sentiment_label.capitalize()
    color = sentiment_color(normalized_label)

    st.markdown(
        f"""
        <div style="display:flex; align-items:center; gap:1rem; margin:0.5rem 0;">
            <span style="font-size:1.5rem; color:{color}; font-weight:700;">
                {normalized_label}
            </span>
            <span style="color:#888;">
                Score: {sentiment.sentiment_score:+.2f}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Pros**")
        if sentiment.pros:
            for item in sentiment.pros[:8]:
                st.markdown(f"- {item}")
        else:
            st.caption("N/A")

    with col2:
        st.markdown("**Cons**")
        if sentiment.cons:
            for item in sentiment.cons[:8]:
                st.markdown(f"- {item}")
        else:
            st.caption("N/A")

    if sentiment.citations:
        st.markdown("**Citations**")
        for c in sentiment.citations[:5]:
            source = c.source or "Unknown source"
            stamp = f" ({c.timestamp})" if c.timestamp else ""
            if c.url:
                st.markdown(f"- [{source}{stamp}]({c.url})")
            else:
                st.markdown(f"- {source}{stamp}")


def render_report_meta(report: ReportAgentOutput) -> None:
    """Render report metadata and download controls."""
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Gemini Calls", report.gemini_calls_used)
    with col2:
        st.metric("Tokens Used", f"{report.total_tokens_used:,}")
    with col3:
        st.metric("Generation Time", f"{report.generation_time_s:.1f}s")

    if report.pdf_bytes:
        st.download_button(
            "Download PDF Report",
            data=report.pdf_bytes,
            file_name=f"{report.ticker}_{report.report_depth.value}_report.pdf",
            mime="application/pdf",
            width="stretch",
        )
    else:
        st.info("PDF is not available for this run.")

    if report.markdown_report:
        st.download_button(
            "Download Markdown Report",
            data=report.markdown_report.encode("utf-8"),
            file_name=f"{report.ticker}_{report.report_depth.value}_report.md",
            mime="text/markdown",
            width="stretch",
        )


def _confidence_to_percent(value: float | int | None) -> float:
    if value is None:
        return 0.0
    try:
        conf = float(value)
    except (TypeError, ValueError):
        return 0.0
    if conf <= 1.0:
        conf *= 100.0
    return max(0.0, min(100.0, conf))


"""
SmartStock AI Analyzer — Streamlit Sidebar
Ticker input, depth selector, and run button.
"""

from __future__ import annotations

import streamlit as st
from schemas.enums import ReportDepth


def render_sidebar() -> tuple[str, ReportDepth, bool]:
    """
    Render the sidebar and return (ticker, depth, run_clicked).
    """
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding: 1rem 0;">
                <h1 style="margin:0; font-size:1.6rem;">
                    📈 SmartStock AI
                </h1>
                <p style="color:#888; font-size:0.85rem; margin-top:0.3rem;">
                    AI-Powered Stock Analysis
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.divider()

        # Ticker input
        ticker = st.text_input(
            "🔍 Stock Ticker",
            value="AAPL",
            max_chars=10,
            placeholder="e.g. AAPL, MSFT, GOOGL",
            help="Enter a valid stock ticker symbol",
            key="sidebar_ticker",
        ).upper().strip()

        st.markdown("")

        # Depth selector
        depth_options = {
            "⚡ Quick (2 AI calls)": ReportDepth.QUICK,
            "📊 Standard (4 AI calls)": ReportDepth.STANDARD,
            "🔬 Deep (6 AI calls)": ReportDepth.DEEP,
        }
        depth_label = st.radio(
            "📋 Analysis Depth",
            options=list(depth_options.keys()),
            index=1,
            help="Quick: fast overview · Standard: balanced · Deep: comprehensive",
            key="sidebar_depth",
        )
        depth = depth_options[depth_label]

        st.markdown("")

        # Run button
        run_clicked = st.button(
            "🚀 Analyze Stock",
            use_container_width=True,
            type="primary",
            key="sidebar_analyze_btn",
        )

        # Info box
        st.divider()
        with st.expander("ℹ️ About", expanded=False):
            st.markdown(
                """
                **SmartStock AI Analyzer** uses Google Gemini
                to perform multi-agent stock analysis:

                1. 📊 **Data** — Market data via yfinance
                2. 🔬 **Research** — Company deep-dive
                3. 💬 **Sentiment** — News sentiment
                4. 📈 **Analysis** — Technical & fundamental
                5. 🎯 **Recommendation** — Buy/Hold/Sell
                6. 📄 **Report** — PDF generation

                All processing is **local & free**.
                """
            )

        st.divider()
        st.caption("Built with Streamlit · Gemini · LangChain")

    return ticker, depth, run_clicked

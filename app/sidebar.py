"""
SmartStock AI Analyzer ??Streamlit Sidebar
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
                    ?뱢 SmartStock AI
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
            "?뵇 Stock Ticker",
            value="AAPL",
            max_chars=10,
            placeholder="e.g. AAPL, MSFT, GOOGL",
            help="Enter a valid stock ticker symbol",
            key="sidebar_ticker",
        ).upper().strip()

        st.markdown("")

        # Depth selector
        depth_options = {
            "??Quick (2 AI calls)": ReportDepth.QUICK,
            "?뱤 Standard (4 AI calls)": ReportDepth.STANDARD,
            "?뵮 Deep (6 AI calls)": ReportDepth.DEEP,
        }
        depth_label = st.radio(
            "?뱥 Analysis Depth",
            options=list(depth_options.keys()),
            index=1,
            help="Quick: fast overview 쨌 Standard: balanced 쨌 Deep: comprehensive",
            key="sidebar_depth",
        )
        depth = depth_options[depth_label]

        st.markdown("")

        # Run button
        run_clicked = st.button(
            "?? Analyze Stock",
            width="stretch",
            type="primary",
            key="sidebar_analyze_btn",
        )

        # Info box
        st.divider()
        with st.expander("?뱄툘 About", expanded=False):
            st.markdown(
                """
                **SmartStock AI Analyzer** uses Google Gemini
                to perform multi-agent stock analysis:

                1. ?뱤 **Data** ??Market data via yfinance
                2. ?뵮 **Research** ??Company deep-dive
                3. ?뮠 **Sentiment** ??News sentiment
                4. ?뱢 **Analysis** ??Technical & fundamental
                5. ?렞 **Recommendation** ??Buy/Hold/Sell
                6. ?뱞 **Report** ??PDF generation

                All processing is **local & free**.
                """
            )

        st.divider()
        st.caption("Built with Streamlit 쨌 Gemini 쨌 LangChain")

    return ticker, depth, run_clicked


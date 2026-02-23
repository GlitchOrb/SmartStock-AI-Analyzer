
from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

st.set_page_config(
    page_title="SmartStock AI Analyzer",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(180deg, #0a0a1a 0%, #111128 50%, #0d0d20 100%);
    }

    section[data-testid="stSidebar"] {
        background: #0f0f25;
        border-right: 1px solid #1a1a3e;
    }

    .stMetric {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        padding: 0.8rem;
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 8px;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(68, 138, 255, 0.15);
        border-bottom: 2px solid #448AFF;
    }

    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1A237E, #448AFF);
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }

    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(68, 138, 255, 0.35);
    }

    div[data-testid="stExpander"] {
        border: 1px solid rgba(255, 255, 255, 0.06);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.02);
    }

    .stDownloadButton > button {
        background: linear-gradient(135deg, #00C853, #00E676) !important;
        border: none;
        border-radius: 12px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from app.sidebar import render_sidebar
from app.components import (
    render_header,
    render_signal_card,
    render_analysis_section,
    render_sentiment_section,
    render_report_meta,
)


def main(ticker, depth, run_clicked):

    # Hero area
    if "report" not in st.session_state:
        st.markdown(
            """
            <div style="text-align:center; padding:4rem 0;">
                <h1 style="font-size:2.5rem; background: linear-gradient(135deg, #448AFF, #00E676);
                    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
                    font-weight:700;">
                    SmartStock AI Analyzer
                </h1>
                <p style="color:#666; font-size:1.1rem; max-width:600px; margin:1rem auto;">
                    AI-powered stock analysis using Google Gemini.
                    Enter a ticker in the sidebar and click <b>Analyze Stock</b> to get started.
                </p>
                <div style="display:flex; justify-content:center; gap:2rem; margin-top:2rem;">
                    <div style="text-align:center;">
                        <div style="font-size:2rem;">📊</div>
                        <div style="color:#888; font-size:0.85rem;">Technical Analysis</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem;">🧠</div>
                        <div style="color:#888; font-size:0.85rem;">AI Insights</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2rem;">📄</div>
                        <div style="color:#888; font-size:0.85rem;">PDF Reports</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    # Display existing report
    report = st.session_state["report"]
    data = st.session_state["data"]
    analysis = st.session_state["analysis"]
    recommendation = st.session_state["recommendation"]
    sentiment = st.session_state.get("sentiment")

    render_header(data)
    render_signal_card(recommendation)

    # Tabs
    tabs = st.tabs(["📊 Analysis", "💬 Sentiment", "📋 Report", "📥 Export"])

    with tabs[0]:
        render_analysis_section(analysis)

    with tabs[1]:
        if sentiment:
            render_sentiment_section(sentiment)
        else:
            st.info("Sentiment analysis not available in Quick mode.")

    with tabs[2]:
        st.markdown(f"### Executive Summary")
        st.write(report.executive_summary)
        st.divider()
        for section in report.sections:
            with st.expander(section.title, expanded=True):
                st.markdown(section.content)

    with tabs[3]:
        render_report_meta(report)


# ── Run analysis ──
if "run_triggered" not in st.session_state:
    st.session_state["run_triggered"] = False

ticker, depth, run_clicked = render_sidebar()

if run_clicked:
    st.session_state["run_triggered"] = True

if st.session_state.get("run_triggered"):
    st.session_state["run_triggered"] = False

    if not ticker:
        st.error("Please enter a valid ticker symbol.")
    else:
        from schemas.config import settings

        if not settings.gemini_api_key:
            st.error("⚠️ GEMINI_API_KEY not found. Please set it in your `.env` file.")
        else:
            with st.spinner(f"🔄 Running {depth.value} analysis for **{ticker}**..."):
                try:
                    from agents.report_agent import ReportAgent
                    from agents.data_agent import DataAgent
                    from agents.research_agent import ResearchAgent
                    from agents.sentiment_agent import SentimentAgent
                    from agents.analysis_agent import AnalysisAgent
                    from agents.recommendation_agent import RecommendationAgent

                    # Run the full pipeline via ReportAgent
                    report_agent = ReportAgent()
                    report = report_agent.run(ticker, depth)

                    # We need individual outputs for the UI
                    data = DataAgent().run(ticker)
                    research = ResearchAgent().run(data, depth)
                    sentiment = SentimentAgent().run(data, research, depth)
                    analysis = AnalysisAgent().run(data, sentiment, depth)
                    recommendation = RecommendationAgent().run(data, analysis, sentiment, depth)

                    # Store in session state
                    st.session_state["report"] = report
                    st.session_state["data"] = data
                    st.session_state["analysis"] = analysis
                    st.session_state["recommendation"] = recommendation
                    st.session_state["sentiment"] = sentiment

                    st.rerun()

                except Exception as e:
                    st.error(f"❌ Analysis failed: {e}")
                    st.exception(e)
else:
    main(ticker, depth, run_clicked)

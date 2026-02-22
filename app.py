"""
SmartStock AI Analyzer — Streamlit App (app.py)
Stage 3: Full 6-agent pipeline with RAG, news, sentiment, citations.
"""

from __future__ import annotations

import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

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
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 0.8rem;
    }
    .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1A237E, #448AFF);
        border: none; border-radius: 12px;
        padding: 0.6rem 1.5rem; font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(68,138,255,0.35);
    }
    .stDownloadButton > button {
        background: linear-gradient(135deg, #00C853, #00E676) !important;
        border: none; border-radius: 12px; font-weight: 600;
    }
    div.stAlert { border-radius: 12px; }
    .budget-display {
        background: rgba(68,138,255,0.08);
        border: 1px solid rgba(68,138,255,0.2);
        border-radius: 8px; padding: 0.5rem 0.8rem;
        text-align: center; margin: 0.5rem 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

from schemas.enums import ReportDepth
from schemas.config import settings
from utils.gemini import gemini_client, CALL_BUDGET


# ──────────────────────────────────────────────
# Cached data fetch (15 min TTL)
# ──────────────────────────────────────────────

@st.cache_data(ttl=900, show_spinner=False)
def cached_fetch_data(ticker: str, period: str):
    from data.fetcher import fetch_all
    return fetch_all(ticker, period)


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────

def render_sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style="text-align:center; padding:1rem 0;">
                <h1 style="margin:0; font-size:1.6rem;">📈 SmartStock AI</h1>
                <p style="color:#888; font-size:0.82rem; margin-top:0.3rem;">
                    AI 기반 주식 분석 시스템 v3
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.divider()

        tickers_input = st.text_input(
            "🔍 종목 티커 (쉼표로 구분)",
            value="AAPL",
            placeholder="예: AAPL, MSFT, GOOGL",
            help="하나 이상의 종목 티커를 쉼표로 구분하여 입력하세요.",
        )

        period = st.selectbox(
            "📅 분석 기간",
            options=["3mo", "6mo", "1y", "2y", "5y"],
            index=2,
        )

        depth_options = {
            "⚡ Quick (AI 2회)": ReportDepth.QUICK,
            "📊 Standard (AI 4회)": ReportDepth.STANDARD,
            "🔬 Deep (AI 6회)": ReportDepth.DEEP,
        }
        depth_label = st.radio("📋 분석 깊이", options=list(depth_options.keys()), index=1)
        depth = depth_options[depth_label]

        # Budget display
        budget = CALL_BUDGET[depth]
        st.markdown(
            f'<div class="budget-display">'
            f'🤖 API 호출 예산: <strong>0 / {budget}</strong> ({depth.value})'
            f'</div>',
            unsafe_allow_html=True,
        )

        st.markdown("")
        run_clicked = st.button("🚀 분석 실행", use_container_width=True, type="primary")

        st.divider()
        with st.expander("ℹ️ 정보", expanded=False):
            st.markdown(
                """
                **6-Agent Pipeline:**

                1. 📊 **DataAgent** — yfinance 데이터
                2. 🔍 **ResearchAgent** — 뉴스 RAG
                3. 🎭 **SentimentAgent** — 감성 분석
                4. 📈 **AnalysisAgent** — 시나리오 분석
                5. 🎯 **RecommendationAgent** — 투자 추천
                6. 📄 **ReportAgent** — PDF 리포트
                """
            )

        tickers = [t.strip().upper() for t in tickers_input.split(",") if t.strip()]
        return tickers, period, depth, run_clicked


# ──────────────────────────────────────────────
# 6-Agent Pipeline
# ──────────────────────────────────────────────

def run_analysis(ticker: str, period: str, depth: ReportDepth):
    """Run the full 6-agent pipeline with st.status blocks."""
    from schemas.agents import (
        AnalysisAgentOutput,
        RecommendationAgentOutput,
        ResearchAgentOutput,
        SentimentAgentOutput,
    )

    results = {
        "data": None,
        "research": None,
        "sentiment": None,
        "analysis": None,
        "recommendation": None,
        "markdown": None,
        "pdf_bytes": None,
        "errors": [],
    }

    gemini_client.reset_counters()
    budget = CALL_BUDGET[depth]
    start_time = time.time()

    # ── Step 1: DataAgent ──
    with st.status("📊 [1/6] 데이터 수집 중...", expanded=True) as status:
        try:
            data = cached_fetch_data(ticker, period)
            results["data"] = data

            if not data.company_name or data.company_name == ticker:
                st.error(f"❌ 티커를 찾을 수 없습니다: {ticker}")
                status.update(label="📊 [1/6] 티커를 찾을 수 없습니다", state="error")
                return results

            if data.anomalies:
                for a in data.anomalies:
                    st.warning(f"⚠️ {a}")

            st.write(f"✅ {data.company_name} ({data.ticker}) 데이터 수집 완료")
            st.write(f"   현재가: ${data.price.current:.2f} ({data.price.change_pct:+.2f}%)")
            status.update(label="📊 [1/6] 데이터 수집 완료", state="complete", expanded=False)

        except Exception as e:
            st.error(f"❌ 데이터 수집 실패: {e}")
            results["errors"].append(f"데이터 수집 실패: {e}")
            status.update(label="📊 [1/6] 데이터 수집 실패", state="error")
            return results

    # ── Step 2: ResearchAgent (RAG) ──
    with st.status("🔍 [2/6] 뉴스 리서치 중...", expanded=True) as status:
        try:
            from agents.research_agent import run_research_agent
            research = run_research_agent(ticker, period, depth)
            results["research"] = research

            st.write(f"✅ 뉴스 {research.news_count}건 수집 | 품질: {research.data_quality}")
            if research.key_themes:
                st.write(f"   핵심 테마: {', '.join(research.key_themes[:3])}")
            if research.warnings:
                for w in research.warnings:
                    st.warning(f"⚠️ {w}")

            _update_budget_display(depth)
            status.update(label=f"🔍 [2/6] 리서치 완료 ({research.news_count}건)", state="complete", expanded=False)

        except Exception as e:
            st.warning(f"⚠️ 리서치 실패: {e}. 건너뜁니다.")
            results["errors"].append(f"리서치 실패: {e}")
            research = ResearchAgentOutput(ticker=ticker, warnings=[f"리서치 실패: {e}"])
            results["research"] = research
            status.update(label="🔍 [2/6] 리서치 실패 (건너뜀)", state="error", expanded=False)

    # ── Step 3: SentimentAgent ──
    with st.status("🎭 [3/6] 센티멘트 분석 중...", expanded=True) as status:
        try:
            from agents.sentiment_agent import run_sentiment_agent
            sentiment = run_sentiment_agent(research, depth)
            results["sentiment"] = sentiment

            label_emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(
                sentiment.sentiment_label, "⚪"
            )
            st.write(f"✅ 센티멘트: {label_emoji} {sentiment.sentiment_label} ({sentiment.sentiment_score:.2f})")
            if sentiment.warnings:
                for w in sentiment.warnings:
                    if "건너뜀" not in w and "Quick" not in w:
                        st.warning(f"⚠️ {w}")

            _update_budget_display(depth)
            status.update(label=f"🎭 [3/6] 센티멘트 완료 ({sentiment.sentiment_label})", state="complete", expanded=False)

        except Exception as e:
            st.warning(f"⚠️ 센티멘트 분석 실패: {e}. 중립으로 기본 설정됩니다.")
            results["errors"].append(f"센티멘트 실패: {e}")
            sentiment = SentimentAgentOutput(ticker=ticker, warnings=[f"분석 실패: {e}"])
            results["sentiment"] = sentiment
            status.update(label="🎭 [3/6] 센티멘트 실패 (중립)", state="error", expanded=False)

    # ── Step 4: AnalysisAgent ──
    with st.status("📈 [4/6] AI 시나리오 분석 중...", expanded=True) as status:
        try:
            from agents.analysis_agent import run_analysis_agent
            analysis = run_analysis_agent(data, depth, sentiment)
            results["analysis"] = analysis

            mode_label = "통합 분석+추천" if depth == ReportDepth.QUICK else "시나리오 분석"
            st.write(f"✅ {mode_label} 완료")
            if analysis.bull_case.thesis:
                st.write(f"   🟢 Bull: {analysis.bull_case.thesis[:80]}...")

            _update_budget_display(depth)
            status.update(label="📈 [4/6] AI 분석 완료", state="complete", expanded=False)

        except Exception as e:
            st.warning(f"⚠️ AI 분석 실패: {e}. 기본값으로 계속합니다.")
            results["errors"].append(f"AI 분석 실패: {e}")
            analysis = AnalysisAgentOutput(ticker=ticker)
            results["analysis"] = analysis
            status.update(label="📈 [4/6] AI 분석 실패", state="error", expanded=False)

    # ── Step 5: RecommendationAgent ──
    with st.status("🎯 [5/6] 투자 추천 생성 중...", expanded=True) as status:
        try:
            from agents.recommendation_agent import run_recommendation_agent
            recommendation = run_recommendation_agent(analysis, depth)
            results["recommendation"] = recommendation

            rating_emoji = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(recommendation.rating, "⚪")
            st.write(f"✅ 등급: {rating_emoji} {recommendation.rating} (신뢰도 {recommendation.confidence}%)")

            _update_budget_display(depth)
            status.update(label=f"🎯 [5/6] 추천 완료 ({recommendation.rating})", state="complete", expanded=False)

        except Exception as e:
            st.warning(f"⚠️ 추천 실패: {e}. 기본값으로 계속합니다.")
            results["errors"].append(f"추천 실패: {e}")
            recommendation = RecommendationAgentOutput(ticker=ticker)
            results["recommendation"] = recommendation
            status.update(label="🎯 [5/6] 추천 실패 (Hold)", state="error", expanded=False)

    # ── Step 6: ReportAgent (PDF + Markdown) ──
    with st.status("📄 [6/6] 리포트 생성 중...", expanded=True) as status:
        try:
            from reporting.pdf_generator import build_markdown_report, generate_pdf

            markdown_report = build_markdown_report(
                data, research, sentiment, analysis, recommendation, depth,
            )
            results["markdown"] = markdown_report
            st.write("✅ 마크다운 리포트 생성 완료")

            # PDF generation
            try:
                pdf_bytes = generate_pdf(
                    data, research, sentiment, analysis, recommendation, depth,
                )
                results["pdf_bytes"] = pdf_bytes
                st.write(f"✅ PDF 생성 완료 ({len(pdf_bytes):,} bytes)")
            except FileNotFoundError as fe:
                st.error(
                    "❌ PDF 생성 실패: NanumGothic.ttf 폰트 파일이 필요합니다.\n"
                    "assets/fonts/NanumGothic.ttf 경로에 폰트 파일을 배치해주세요.\n"
                    "다운로드: https://fonts.google.com/specimen/Nanum+Gothic"
                )
                results["errors"].append(f"PDF 폰트 누락: {fe}")
            except Exception as pe:
                st.warning(f"⚠️ PDF 생성 실패: {pe}. 마크다운 파일을 대신 제공합니다.")
                results["errors"].append(f"PDF 생성 실패: {pe}")

            status.update(label="📄 [6/6] 리포트 완료", state="complete", expanded=False)

        except Exception as e:
            st.warning(f"⚠️ 리포트 생성 실패: {e}")
            results["errors"].append(f"리포트 생성 실패: {e}")
            status.update(label="📄 [6/6] 리포트 실패", state="error", expanded=False)

    elapsed = time.time() - start_time
    calls_used = gemini_client.call_count
    st.success(
        f"✅ {ticker} 분석 완료! "
        f"(Gemini {calls_used}/{budget}회 호출, {elapsed:.1f}초 소요)"
    )

    return results


# ──────────────────────────────────────────────
# Budget display helper
# ──────────────────────────────────────────────

def _update_budget_display(depth: ReportDepth):
    """Show current Gemini call count in sidebar."""
    budget = CALL_BUDGET[depth]
    used = gemini_client.call_count
    pct = min(100, int((used / budget) * 100)) if budget > 0 else 0
    color = "#00C853" if pct < 70 else ("#FFC107" if pct < 100 else "#D32F2F")
    st.sidebar.markdown(
        f'<div class="budget-display">'
        f'🤖 API 호출: <strong style="color:{color}">{used} / {budget}</strong> ({depth.value})'
        f'</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────
# Display results
# ──────────────────────────────────────────────

def display_results(ticker: str, results: dict):
    """Display full analysis results for a single ticker."""
    data = results.get("data")
    research = results.get("research")
    sentiment = results.get("sentiment")
    analysis = results.get("analysis")
    recommendation = results.get("recommendation")
    markdown = results.get("markdown")
    pdf_bytes = results.get("pdf_bytes")
    errors = results.get("errors", [])

    if not data:
        st.error(f"❌ {ticker}: 데이터 없음")
        return

    # Header metrics
    st.markdown(f"### {data.company_name} ({data.ticker})")
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("현재가", f"${data.price.current:.2f}", f"{data.price.change_pct:+.2f}%")
    with col2:
        mc = data.fundamentals.market_cap
        mc_str = f"${mc/1e9:.1f}B" if mc and mc >= 1e9 else (f"${mc/1e6:.1f}M" if mc else "N/A")
        st.metric("시가총액", mc_str)
    with col3:
        if recommendation:
            emoji = {"Buy": "🟢", "Hold": "🟡", "Sell": "🔴"}.get(recommendation.rating, "⚪")
            st.metric("등급", f"{emoji} {recommendation.rating}")
        else:
            st.metric("등급", "N/A")
    with col4:
        if recommendation:
            st.metric("신뢰도", f"{recommendation.confidence}%")
        else:
            st.metric("신뢰도", "N/A")
    with col5:
        if sentiment:
            emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(sentiment.sentiment_label, "⚪")
            st.metric("센티멘트", f"{emoji} {sentiment.sentiment_score:.2f}")
        else:
            st.metric("센티멘트", "N/A")

    # Tabs
    tab_report, tab_sentiment, tab_export = st.tabs([
        "📋 분석 리포트", "🎭 뉴스 & 센티멘트", "📥 내보내기",
    ])

    with tab_report:
        if markdown:
            st.markdown(markdown)
        else:
            st.info("마크다운 리포트를 생성하지 못했습니다.")

    with tab_sentiment:
        if sentiment:
            _display_sentiment(sentiment)
        else:
            st.info("센티멘트 분석 결과가 없습니다.")

        if research and research.news_count > 0:
            _display_research(research)

    with tab_export:
        col_pdf, col_md = st.columns(2)
        with col_pdf:
            if pdf_bytes:
                st.download_button(
                    "📥 PDF 리포트 다운로드",
                    data=pdf_bytes,
                    file_name=f"{ticker}_report_{datetime.now():%Y%m%d_%H%M}.pdf",
                    mime="application/pdf",
                    use_container_width=True,
                )
            else:
                st.info(
                    "PDF를 생성하지 못했습니다.\n"
                    "assets/fonts/NanumGothic.ttf 파일이 필요합니다."
                )
        with col_md:
            if markdown:
                st.download_button(
                    "📄 마크다운 리포트 다운로드",
                    data=markdown.encode("utf-8"),
                    file_name=f"{ticker}_report_{datetime.now():%Y%m%d_%H%M}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

    if errors:
        with st.expander("⚠️ 오류 로그", expanded=False):
            for err in errors:
                st.warning(err)


def _display_sentiment(sentiment):
    """Display sentiment analysis with citations."""
    label_emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(
        sentiment.sentiment_label, "⚪"
    )
    st.markdown(
        f"#### {label_emoji} 센티멘트: {sentiment.sentiment_label.upper()} "
        f"(점수: {sentiment.sentiment_score:.2f})"
    )

    col_pro, col_con = st.columns(2)
    with col_pro:
        st.markdown("**✅ 긍정 요인**")
        if sentiment.pros:
            for p in sentiment.pros:
                st.markdown(f"- {p}")
        else:
            st.markdown("*없음*")
    with col_con:
        st.markdown("**❌ 부정 요인**")
        if sentiment.cons:
            for c in sentiment.cons:
                st.markdown(f"- {c}")
        else:
            st.markdown("*없음*")

    if sentiment.citations:
        with st.expander("📎 출처 보기", expanded=False):
            for i, c in enumerate(sentiment.citations, 1):
                url_link = f"[🔗 링크]({c.url})" if c.url else ""
                st.markdown(
                    f"**{i}.** \"{c.text}\"\n\n"
                    f"   _{c.source}_ ({c.timestamp}) {url_link}"
                )
                if i < len(sentiment.citations):
                    st.divider()


def _display_research(research):
    """Display research themes and timeline."""
    st.markdown(f"#### 📰 뉴스 리서치 ({research.news_count}건)")

    if research.key_themes:
        st.markdown("**핵심 테마:**")
        for theme in research.key_themes:
            st.markdown(f"- {theme}")

    if research.timeline:
        with st.expander("📅 타임라인", expanded=False):
            for evt in research.timeline:
                st.markdown(f"- **{evt.date}**: {evt.event}")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    st.warning("⚠️ 본 리포트는 투자 조언이 아닌 정보 제공 목적입니다.", icon="📢")

    tickers, period, depth, run_clicked = render_sidebar()

    if run_clicked:
        if not tickers:
            st.error("하나 이상의 종목 티커를 입력해주세요.")
            return

        if not settings.gemini_api_key:
            st.error("⚠️ GEMINI_API_KEY가 설정되지 않았습니다. `.env` 파일을 확인하세요.")
            return

        st.divider()
        budget = CALL_BUDGET[depth]
        st.markdown(
            f"### 🔄 분석 실행: {', '.join(tickers)} "
            f"({depth.value} 모드, API {budget}회)"
        )

        all_results = {}
        for i, ticker in enumerate(tickers):
            if i > 0:
                st.divider()
            st.markdown(f"## 📊 {ticker}")
            results = run_analysis(ticker, period, depth)
            all_results[ticker] = results

        st.divider()
        for ticker, res in all_results.items():
            display_results(ticker, res)

    else:
        # Landing page
        st.markdown(
            """
            <div style="text-align:center; padding:4rem 0;">
                <h1 style="font-size:2.5rem; background:linear-gradient(135deg, #448AFF, #00E676);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
                    font-weight:700;">
                    SmartStock AI Analyzer
                </h1>
                <p style="color:#666; font-size:1.1rem; max-width:600px; margin:1rem auto;">
                    Google Gemini AI + RAG 기반 주식 분석 시스템.<br>
                    6-Agent 파이프라인으로 종합 분석을 수행합니다.
                </p>
                <div style="display:flex; justify-content:center; gap:2rem; margin-top:2.5rem; flex-wrap:wrap;">
                    <div style="text-align:center;">
                        <div style="font-size:2.2rem;">📊</div>
                        <div style="color:#888; font-size:0.8rem;">데이터 수집</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.2rem;">🔍</div>
                        <div style="color:#888; font-size:0.8rem;">뉴스 리서치</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.2rem;">🎭</div>
                        <div style="color:#888; font-size:0.8rem;">센티멘트</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.2rem;">🧠</div>
                        <div style="color:#888; font-size:0.8rem;">AI 시나리오</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.2rem;">🎯</div>
                        <div style="color:#888; font-size:0.8rem;">투자 추천</div>
                    </div>
                    <div style="text-align:center;">
                        <div style="font-size:2.2rem;">📄</div>
                        <div style="color:#888; font-size:0.8rem;">PDF 리포트</div>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


if __name__ == "__main__":
    main()
else:
    main()

"""
SmartStock AI Analyzer — PDF Report Generator
Stage 3: Added sentiment/news section + citations + .md fallback.
Korean markdown report → matplotlib chart → ReportLab PDF with NanumGothic font.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import HexColor
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.agents import (
    AnalysisAgentOutput,
    DataAgentOutput,
    RecommendationAgentOutput,
    ResearchAgentOutput,
    SentimentAgentOutput,
)
from schemas.enums import ReportDepth
from utils.logger import log_agent


# ──────────────────────────────────────────────
# Color constants
# ──────────────────────────────────────────────

C_PRIMARY = HexColor("#1A237E")
C_ACCENT = HexColor("#448AFF")
C_SUCCESS = HexColor("#00C853")
C_WARNING = HexColor("#FFC107")
C_DANGER = HexColor("#D32F2F")
C_TEXT = HexColor("#212121")
C_TEXT_LIGHT = HexColor("#757575")
C_BG_LIGHT = HexColor("#F5F5F5")
C_BORDER = HexColor("#E0E0E0")

RATING_COLORS = {
    "Buy": C_SUCCESS,
    "Hold": C_WARNING,
    "Sell": C_DANGER,
}

SENTIMENT_COLORS = {
    "positive": C_SUCCESS,
    "neutral": C_WARNING,
    "negative": C_DANGER,
}

DISCLAIMER_KR = "본 리포트는 투자 조언이 아닌 정보 제공 목적입니다."


# ──────────────────────────────────────────────
# Font setup
# ──────────────────────────────────────────────

def _register_font() -> str:
    """Register NanumGothic font. Raises FileNotFoundError if missing."""
    font_path = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "NanumGothic.ttf"
    if not font_path.exists():
        raise FileNotFoundError(
            f"NanumGothic.ttf 폰트 파일을 찾을 수 없습니다: {font_path}\n"
            f"assets/fonts/NanumGothic.ttf 경로에 폰트 파일을 배치해주세요.\n"
            f"다운로드: https://fonts.google.com/specimen/Nanum+Gothic"
        )
    pdfmetrics.registerFont(TTFont("NanumGothic", str(font_path)))
    log_agent("PDF", "NanumGothic 폰트 등록 완료")
    return "NanumGothic"


def _get_styles(font_name: str) -> dict[str, ParagraphStyle]:
    """Create custom paragraph styles using NanumGothic."""
    return {
        "title": ParagraphStyle(
            "Title", fontName=font_name, fontSize=24, textColor=C_PRIMARY,
            alignment=TA_CENTER, spaceAfter=4 * mm, leading=30,
        ),
        "subtitle": ParagraphStyle(
            "Subtitle", fontName=font_name, fontSize=12, textColor=C_TEXT_LIGHT,
            alignment=TA_CENTER, spaceAfter=6 * mm, leading=16,
        ),
        "heading": ParagraphStyle(
            "Heading", fontName=font_name, fontSize=16, textColor=C_PRIMARY,
            spaceBefore=6 * mm, spaceAfter=3 * mm, leading=20,
        ),
        "subheading": ParagraphStyle(
            "SubHeading", fontName=font_name, fontSize=13, textColor=HexColor("#283593"),
            spaceBefore=4 * mm, spaceAfter=2 * mm, leading=17,
        ),
        "body": ParagraphStyle(
            "Body", fontName=font_name, fontSize=10, textColor=C_TEXT,
            leading=15, alignment=TA_JUSTIFY, spaceAfter=2 * mm,
        ),
        "body_bold": ParagraphStyle(
            "BodyBold", fontName=font_name, fontSize=10, textColor=C_TEXT,
            leading=15,
        ),
        "small": ParagraphStyle(
            "Small", fontName=font_name, fontSize=8, textColor=C_TEXT_LIGHT,
            leading=11,
        ),
        "citation": ParagraphStyle(
            "Citation", fontName=font_name, fontSize=8, textColor=HexColor("#546E7A"),
            leading=11, leftIndent=10 * mm,
        ),
        "rating": ParagraphStyle(
            "Rating", fontName=font_name, fontSize=22, alignment=TA_CENTER,
            spaceAfter=3 * mm, leading=28,
        ),
        "disclaimer": ParagraphStyle(
            "Disclaimer", fontName=font_name, fontSize=7, textColor=C_TEXT_LIGHT,
            alignment=TA_CENTER, leading=10,
        ),
    }


# ──────────────────────────────────────────────
# Chart generation
# ──────────────────────────────────────────────

def _render_price_chart(csv_path: str, ticker: str) -> str | None:
    """Render price chart and save to temp file. Returns path."""
    try:
        df = pd.read_csv(csv_path, parse_dates=["Date"])
    except Exception:
        log_agent("PDF", "차트 생성 실패: CSV 읽기 오류")
        return None

    if df.empty or "Close" not in df.columns:
        return None

    close = df["Close"].astype(float)
    ma_20 = close.rolling(window=20).mean()
    ma_60 = close.rolling(window=60).mean()

    fig, ax = plt.subplots(figsize=(10, 4.5))
    fig.patch.set_facecolor("#0d1117")
    ax.set_facecolor("#0d1117")

    ax.plot(df["Date"], close, color="#448AFF", linewidth=1.4, label="종가")
    ax.plot(df["Date"], ma_20, color="#FFC107", linewidth=1.0, linestyle="--", label="20일 이평")
    ax.plot(df["Date"], ma_60, color="#FF5722", linewidth=1.0, linestyle="--", label="60일 이평")
    ax.fill_between(df["Date"], close, alpha=0.08, color="#448AFF")

    ax.set_title(f"{ticker} 가격 차트", color="#fff", fontsize=13, pad=10)
    ax.set_ylabel("가격 ($)", color="#aaa", fontsize=9)
    ax.tick_params(colors="#aaa", labelsize=8)
    ax.legend(loc="upper left", fontsize=8, facecolor="#0d1117", edgecolor="#333", labelcolor="#aaa")
    ax.grid(True, alpha=0.12, color="#444")

    for spine in ax.spines.values():
        spine.set_color("#333")

    plt.tight_layout()

    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    fig.savefig(tmp.name, dpi=150, bbox_inches="tight", facecolor="#0d1117")
    plt.close(fig)
    return tmp.name


# ──────────────────────────────────────────────
# Markdown report builder
# ──────────────────────────────────────────────

def build_markdown_report(
    data: DataAgentOutput,
    research: ResearchAgentOutput | None,
    sentiment: SentimentAgentOutput | None,
    analysis: AnalysisAgentOutput,
    recommendation: RecommendationAgentOutput,
    depth: ReportDepth,
) -> str:
    """Build a full Korean markdown report with all 6 agent outputs."""
    p = data.price
    f = data.fundamentals
    t = data.technicals
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    rating = recommendation.rating
    confidence = recommendation.confidence

    md = f"""# 📊 {data.company_name} ({data.ticker}) AI 분석 리포트

> 분석 깊이: **{depth.value}** | 생성 시각: {now_str}

---

## 📋 Executive Summary

**투자 등급: {rating}** (신뢰도: {confidence}%)

"""
    if recommendation.rationale:
        for r in recommendation.rationale:
            md += f"- {r}\n"
    md += "\n---\n\n"

    # Price & Technicals
    md += f"""## 📈 가격 동향 & 기술적 분석

| 항목 | 값 |
|---|---|
| 현재가 | ${p.current:.2f} |
| 일간 변동 | {p.change_pct:+.2f}% |
| 52주 최고 | ${p.high_52w:.2f} |
| 52주 최저 | ${p.low_52w:.2f} |
| RSI(14) | {t.rsi_14 if t.rsi_14 is not None else 'N/A'} |
| MACD | {t.macd if t.macd is not None else 'N/A'} |
| 볼린저 밴드 | {t.bb_lower or 'N/A'} ~ {t.bb_upper or 'N/A'} |
| 20일 이평 | {t.ma_20 if t.ma_20 is not None else 'N/A'} |
| 60일 이평 | {t.ma_60 if t.ma_60 is not None else 'N/A'} |
| MDD | {t.mdd if t.mdd is not None else 'N/A'}% |
| 변동성(연환산) | {t.volatility if t.volatility is not None else 'N/A'}% |

"""

    # Fundamentals
    md += f"""## 💰 펀더멘탈 분석

| 항목 | 값 |
|---|---|
| 시가총액 | {_fmt_num(f.market_cap)} |
| PER | {f.per or 'N/A'} |
| PBR | {f.pbr or 'N/A'} |
| ROE | {_fmt_pct(f.roe)} |
| 부채비율 | {f.debt_ratio or 'N/A'} |
| 매출 | {_fmt_num(f.revenue)} |
| 영업이익률 | {_fmt_pct(f.operating_profit)} |
| EPS | {f.eps or 'N/A'} |

"""

    # Research / News
    if research is not None:
        md += "## 📰 뉴스 & 리서치\n\n"
        md += f"수집 기사: **{research.news_count}건** | 데이터 품질: **{research.data_quality}**\n\n"
        if research.key_themes:
            md += "**핵심 테마:**\n"
            for theme in research.key_themes:
                md += f"- {theme}\n"
        if research.timeline:
            md += "\n**타임라인:**\n"
            for event in research.timeline:
                md += f"- **{event.date}**: {event.event}\n"
        if research.warnings:
            md += "\n**경고:**\n"
            for w in research.warnings:
                md += f"- ⚠️ {w}\n"
        md += "\n"

    # Sentiment
    if sentiment is not None:
        label_emoji = {"positive": "🟢", "neutral": "🟡", "negative": "🔴"}.get(sentiment.sentiment_label, "⚪")
        md += f"## 🎭 센티멘트 분석\n\n"
        md += f"**센티멘트: {label_emoji} {sentiment.sentiment_label}** (점수: {sentiment.sentiment_score:.2f})\n\n"
        if sentiment.pros:
            md += "**긍정 요인:**\n"
            for pro in sentiment.pros:
                md += f"- ✅ {pro}\n"
        if sentiment.cons:
            md += "\n**부정 요인:**\n"
            for con in sentiment.cons:
                md += f"- ❌ {con}\n"
        if sentiment.citations:
            md += "\n**출처:**\n"
            for c in sentiment.citations:
                url_text = f"[링크]({c.url})" if c.url else ""
                md += f"- 📎 \"{c.text}\" — {c.source} {url_text} ({c.timestamp})\n"
        if sentiment.warnings:
            md += "\n"
            for w in sentiment.warnings:
                md += f"- ⚠️ {w}\n"
        md += "\n"

    # Scenario Analysis
    md += "## 🎯 시나리오 분석\n\n"

    md += f"""### 🟢 강세 시나리오 (Bull Case)
**논거:** {analysis.bull_case.thesis}

"""
    if analysis.bull_case.catalysts:
        md += "**촉매:**\n"
        for c in analysis.bull_case.catalysts:
            md += f"- {c}\n"
    if analysis.bull_case.risks:
        md += "\n**위험:**\n"
        for r in analysis.bull_case.risks:
            md += f"- {r}\n"

    md += f"""
### ⚪ 기본 시나리오 (Base Case)
**논거:** {analysis.base_case.thesis}

"""
    if analysis.base_case.drivers:
        md += "**동인:**\n"
        for d in analysis.base_case.drivers:
            md += f"- {d}\n"

    md += f"""
### 🔴 약세 시나리오 (Bear Case)
**논거:** {analysis.bear_case.thesis}

"""
    if analysis.bear_case.risks:
        md += "**위험:**\n"
        for r in analysis.bear_case.risks:
            md += f"- {r}\n"
    if analysis.bear_case.warning:
        md += f"\n⚠️ **경고:** {analysis.bear_case.warning}\n"

    if analysis.key_drivers:
        md += "\n**핵심 동인:**\n"
        for kd in analysis.key_drivers:
            md += f"- {kd}\n"

    # Recommendation
    md += f"""
---

## ✅ 투자 추천

| 항목 | 내용 |
|---|---|
| 등급 | **{rating}** |
| 신뢰도 | {confidence}% |

"""
    if recommendation.rationale:
        md += "**근거:**\n"
        for r in recommendation.rationale:
            md += f"- {r}\n"

    if recommendation.invalidation_triggers:
        md += "\n**무효화 트리거:**\n"
        for t_item in recommendation.invalidation_triggers:
            md += f"- {t_item}\n"

    # Risks & Watch
    md += "\n---\n\n## ⚠️ 리스크 & 관찰 사항\n\n"
    if recommendation.risk_notes:
        md += f"{recommendation.risk_notes}\n\n"
    if data.anomalies:
        md += "**데이터 이상:**\n"
        for a in data.anomalies:
            md += f"- ⚠️ {a}\n"

    md += f"\n---\n\n> {DISCLAIMER_KR}\n"
    return md


# ──────────────────────────────────────────────
# PDF generator
# ──────────────────────────────────────────────

def generate_pdf(
    data: DataAgentOutput,
    research: ResearchAgentOutput | None,
    sentiment: SentimentAgentOutput | None,
    analysis: AnalysisAgentOutput,
    recommendation: RecommendationAgentOutput,
    depth: ReportDepth,
) -> bytes:
    """
    Generate a full PDF report with all 6 agent outputs.
    Returns PDF as bytes.
    """
    font_name = _register_font()
    styles = _get_styles(font_name)

    buffer = io.BytesIO()

    def _add_footer(canvas, doc):
        canvas.saveState()
        canvas.setFont(font_name, 7)
        canvas.setFillColor(C_TEXT_LIGHT)
        canvas.drawCentredString(A4[0] / 2, 10 * mm, DISCLAIMER_KR)
        canvas.drawRightString(A4[0] - 15 * mm, 10 * mm, f"p.{canvas.getPageNumber()}")
        canvas.restoreState()

    doc = SimpleDocTemplate(
        buffer, pagesize=A4,
        topMargin=15 * mm, bottomMargin=20 * mm,
        leftMargin=18 * mm, rightMargin=18 * mm,
    )

    story: list = []
    p = data.price
    f = data.fundamentals
    t = data.technicals
    rating = recommendation.rating
    confidence = recommendation.confidence
    rating_color = RATING_COLORS.get(rating, C_WARNING)

    # ── Cover ──
    story.append(Spacer(1, 25 * mm))
    story.append(Paragraph("SmartStock AI Analyzer", styles["title"]))
    story.append(Paragraph("AI 주식 분석 리포트", styles["subtitle"]))
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(f"{data.company_name} ({data.ticker})", styles["heading"]))
    story.append(Paragraph(
        f"분석 깊이: {depth.value} | 생성: {datetime.now():%Y-%m-%d %H:%M}",
        styles["small"],
    ))
    story.append(Spacer(1, 6 * mm))

    rating_style = ParagraphStyle(
        "RatingBadge", fontName=font_name, fontSize=22,
        alignment=TA_CENTER, textColor=rating_color, leading=28,
    )
    story.append(Paragraph(f"투자 등급: {rating} (신뢰도 {confidence}%)", rating_style))
    story.append(PageBreak())

    # ── Executive Summary ──
    story.append(Paragraph("Executive Summary", styles["heading"]))
    if recommendation.rationale:
        for r in recommendation.rationale:
            story.append(Paragraph(f"• {r}", styles["body"]))
    story.append(Spacer(1, 4 * mm))

    # ── Price Chart ──
    chart_path = _render_price_chart(data.history_df_path, data.ticker) if data.history_df_path else None
    if chart_path and Path(chart_path).exists():
        story.append(Paragraph("가격 차트 (종가 + 20일/60일 이동평균)", styles["subheading"]))
        story.append(Image(chart_path, width=170 * mm, height=76 * mm))
        story.append(Spacer(1, 4 * mm))

    # ── Technicals Table ──
    story.append(Paragraph("가격 동향 & 기술적 분석", styles["heading"]))
    tech_data = [
        ["항목", "값", "항목", "값"],
        ["현재가", f"${p.current:.2f}", "RSI(14)", f"{t.rsi_14 or 'N/A'}"],
        ["일간 변동", f"{p.change_pct:+.2f}%", "MACD", f"{t.macd or 'N/A'}"],
        ["52주 최고", f"${p.high_52w:.2f}", "20일 이평", f"{t.ma_20 or 'N/A'}"],
        ["52주 최저", f"${p.low_52w:.2f}", "60일 이평", f"{t.ma_60 or 'N/A'}"],
        ["MDD", f"{t.mdd or 'N/A'}%", "변동성", f"{t.volatility or 'N/A'}%"],
    ]
    story.append(_make_table(tech_data, font_name))
    story.append(Spacer(1, 4 * mm))

    # ── Fundamentals Table ──
    story.append(Paragraph("펀더멘탈 분석", styles["heading"]))
    fund_table_data = [
        ["항목", "값", "항목", "값"],
        ["시가총액", _fmt_num(f.market_cap), "PER", f"{f.per or 'N/A'}"],
        ["PBR", f"{f.pbr or 'N/A'}", "ROE", _fmt_pct(f.roe)],
        ["부채비율", f"{f.debt_ratio or 'N/A'}", "EPS", f"{f.eps or 'N/A'}"],
        ["매출", _fmt_num(f.revenue), "영업이익률", _fmt_pct(f.operating_profit)],
    ]
    story.append(_make_table(fund_table_data, font_name))
    story.append(Spacer(1, 4 * mm))

    # ── News & Research ──
    if research is not None and research.news_count > 0:
        story.append(Paragraph("뉴스 & 리서치", styles["heading"]))
        story.append(Paragraph(
            f"수집 기사: {research.news_count}건 | 데이터 품질: {research.data_quality}",
            styles["body"],
        ))
        if research.key_themes:
            story.append(Paragraph("핵심 테마:", styles["body_bold"]))
            for theme in research.key_themes:
                story.append(Paragraph(f"  • {theme}", styles["body"]))
        if research.timeline:
            story.append(Paragraph("타임라인:", styles["body_bold"]))
            for evt in research.timeline[:5]:
                story.append(Paragraph(f"  • {evt.date}: {evt.event}", styles["body"]))
        story.append(Spacer(1, 4 * mm))

    # ── Sentiment ──
    if sentiment is not None and sentiment.sentiment_label != "neutral" or (
        sentiment is not None and (sentiment.pros or sentiment.cons)
    ):
        sent_color = SENTIMENT_COLORS.get(sentiment.sentiment_label, C_WARNING)
        sent_style = ParagraphStyle(
            "SentBadge", fontName=font_name, fontSize=14,
            alignment=TA_CENTER, textColor=sent_color, leading=18,
        )
        story.append(Paragraph("센티멘트 분석", styles["heading"]))
        story.append(Paragraph(
            f"센티멘트: {sentiment.sentiment_label.upper()} (점수: {sentiment.sentiment_score:.2f})",
            sent_style,
        ))
        if sentiment.pros:
            story.append(Paragraph("긍정 요인:", styles["body_bold"]))
            for pro in sentiment.pros:
                story.append(Paragraph(f"  ✅ {pro}", styles["body"]))
        if sentiment.cons:
            story.append(Paragraph("부정 요인:", styles["body_bold"]))
            for con in sentiment.cons:
                story.append(Paragraph(f"  ❌ {con}", styles["body"]))
        if sentiment.citations:
            story.append(Paragraph("출처:", styles["body_bold"]))
            for c in sentiment.citations:
                cite_text = f'"{c.text}" — {c.source} ({c.timestamp})'
                story.append(Paragraph(cite_text, styles["citation"]))
        story.append(Spacer(1, 4 * mm))

    # ── Scenario Analysis ──
    story.append(Paragraph("시나리오 분석", styles["heading"]))

    story.append(Paragraph("강세 시나리오 (Bull Case)", styles["subheading"]))
    story.append(Paragraph(analysis.bull_case.thesis, styles["body"]))
    if analysis.bull_case.catalysts:
        story.append(Paragraph("촉매:", styles["body_bold"]))
        for c in analysis.bull_case.catalysts:
            story.append(Paragraph(f"  • {c}", styles["body"]))
    if analysis.bull_case.risks:
        story.append(Paragraph("위험:", styles["body_bold"]))
        for r in analysis.bull_case.risks:
            story.append(Paragraph(f"  • {r}", styles["body"]))

    story.append(Paragraph("기본 시나리오 (Base Case)", styles["subheading"]))
    story.append(Paragraph(analysis.base_case.thesis, styles["body"]))
    if analysis.base_case.drivers:
        story.append(Paragraph("동인:", styles["body_bold"]))
        for d in analysis.base_case.drivers:
            story.append(Paragraph(f"  • {d}", styles["body"]))

    story.append(Paragraph("약세 시나리오 (Bear Case)", styles["subheading"]))
    story.append(Paragraph(analysis.bear_case.thesis, styles["body"]))
    if analysis.bear_case.risks:
        story.append(Paragraph("위험:", styles["body_bold"]))
        for r in analysis.bear_case.risks:
            story.append(Paragraph(f"  • {r}", styles["body"]))
    if analysis.bear_case.warning:
        story.append(Paragraph(f"경고: {analysis.bear_case.warning}", styles["body"]))

    if analysis.key_drivers:
        story.append(Paragraph("핵심 동인:", styles["subheading"]))
        for kd in analysis.key_drivers:
            story.append(Paragraph(f"  • {kd}", styles["body"]))

    story.append(Spacer(1, 4 * mm))

    # ── Recommendation ──
    story.append(Paragraph("투자 추천", styles["heading"]))
    story.append(Paragraph(f"등급: {rating} | 신뢰도: {confidence}%", rating_style))
    if recommendation.rationale:
        story.append(Paragraph("근거:", styles["body_bold"]))
        for r in recommendation.rationale:
            story.append(Paragraph(f"  • {r}", styles["body"]))
    if recommendation.invalidation_triggers:
        story.append(Paragraph("무효화 트리거:", styles["body_bold"]))
        for t_item in recommendation.invalidation_triggers:
            story.append(Paragraph(f"  • {t_item}", styles["body"]))

    # ── Risks ──
    story.append(Paragraph("리스크 & 관찰 사항", styles["heading"]))
    if recommendation.risk_notes:
        story.append(Paragraph(recommendation.risk_notes, styles["body"]))
    if data.anomalies:
        story.append(Paragraph("데이터 이상:", styles["body_bold"]))
        for a in data.anomalies:
            story.append(Paragraph(f"  ⚠ {a}", styles["body"]))

    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(DISCLAIMER_KR, styles["disclaimer"]))

    # Build PDF
    doc.build(story, onFirstPage=_add_footer, onLaterPages=_add_footer)

    # Cleanup temp chart
    if chart_path and Path(chart_path).exists():
        try:
            os.unlink(chart_path)
        except OSError:
            pass

    pdf_bytes = buffer.getvalue()
    buffer.close()
    log_agent("PDF", f"PDF 생성 완료 ({len(pdf_bytes):,} bytes)")
    return pdf_bytes


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _make_table(data: list[list[str]], font_name: str) -> Table:
    table = Table(data, colWidths=[38 * mm, 45 * mm, 38 * mm, 45 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), C_PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, -1), font_name),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, C_BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, C_BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    return table


def _fmt_num(val: float | None) -> str:
    if val is None:
        return "N/A"
    if abs(val) >= 1e12:
        return f"${val / 1e12:,.2f}T"
    if abs(val) >= 1e9:
        return f"${val / 1e9:,.2f}B"
    if abs(val) >= 1e6:
        return f"${val / 1e6:,.2f}M"
    return f"${val:,.0f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"

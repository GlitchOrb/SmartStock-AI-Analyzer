"""
SmartStock AI Analyzer — PDF Report Generator
Produces professional PDF reports using ReportLab.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Image,
    Table,
    TableStyle,
    PageBreak,
)
from reportlab.lib import colors

from reporting.charts import render_price_chart, render_signal_gauge
from reporting.styles import Colors, SIGNAL_COLORS, get_styles
from schemas.agents import DataAgentOutput, RecommendationAgentOutput, ReportSection
from schemas.config import settings
from schemas.enums import ReportDepth
from utils.logger import log_agent


def generate_pdf(
    ticker: str,
    depth: ReportDepth,
    sections: list[ReportSection],
    executive_summary: str,
    data: DataAgentOutput,
    recommendation: RecommendationAgentOutput,
) -> str:
    """Generate a full PDF report. Returns the file path."""

    pdf_path = str(settings.reports_dir / f"{ticker}_{depth.value}_{datetime.now():%Y%m%d_%H%M%S}.pdf")
    styles = get_styles()

    doc = SimpleDocTemplate(
        pdf_path,
        pagesize=A4,
        topMargin=15 * mm,
        bottomMargin=15 * mm,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
    )

    story: list = []

    # ── Cover Page ──
    story.append(Spacer(1, 30 * mm))
    story.append(Paragraph("SmartStock AI Analyzer", styles["title"]))
    story.append(Paragraph(f"Stock Analysis Report", styles["subtitle"]))
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        f"<b>{data.company_name}</b> ({data.ticker})",
        styles["heading"],
    ))
    story.append(Paragraph(
        f"Report Depth: {depth.value} &nbsp;|&nbsp; "
        f"Generated: {datetime.now():%Y-%m-%d %H:%M}",
        styles["small"],
    ))

    # Signal badge
    signal_color = SIGNAL_COLORS.get(recommendation.signal.value, Colors.TEXT)
    signal_style = styles["signal"]
    signal_style.textColor = signal_color
    story.append(Spacer(1, 8 * mm))
    story.append(Paragraph(recommendation.signal.value, signal_style))

    # Signal gauge image
    gauge_path = render_signal_gauge(recommendation.signal.value, data.ticker)
    if gauge_path and Path(gauge_path).exists():
        story.append(Image(gauge_path, width=140 * mm, height=35 * mm))

    story.append(PageBreak())

    # ── Executive Summary ──
    story.append(Paragraph("Executive Summary", styles["heading"]))
    story.append(Paragraph(executive_summary, styles["body"]))
    story.append(Spacer(1, 4 * mm))

    # ── Price Chart ──
    chart_path = render_price_chart(
        data.history_df_path,
        data.ticker,
        support_levels=[],  # Will be populated from analysis if available
        resistance_levels=[],
    )
    if chart_path and Path(chart_path).exists():
        story.append(Image(chart_path, width=170 * mm, height=102 * mm))
        story.append(Spacer(1, 4 * mm))

    # ── Key Metrics Table ──
    f = data.fundamentals
    metrics_data = [
        ["Metric", "Value", "Metric", "Value"],
        ["Price", f"${data.price.current:.2f}", "Market Cap", _fmt_mc(f.market_cap)],
        ["52W High", f"${data.price.high_52w:.2f}", "P/E Ratio", f"{f.pe_ratio or 'N/A'}"],
        ["52W Low", f"${data.price.low_52w:.2f}", "Forward P/E", f"{f.forward_pe or 'N/A'}"],
        ["Volume", f"{data.volume.current:,}", "EPS", f"{f.eps or 'N/A'}"],
        ["Rel. Volume", f"{data.volume.relative_volume:.2f}x", "Div Yield", _fmt_pct(f.dividend_yield)],
    ]
    table = Table(metrics_data, colWidths=[40 * mm, 45 * mm, 40 * mm, 45 * mm])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), Colors.PRIMARY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTSIZE", (0, 0), (-1, -1), 9),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("ALIGN", (1, 0), (1, -1), "RIGHT"),
        ("ALIGN", (3, 0), (3, -1), "RIGHT"),
        ("GRID", (0, 0), (-1, -1), 0.5, Colors.BORDER),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [Colors.BG_CARD, Colors.BG_LIGHT]),
        ("TOPPADDING", (0, 0), (-1, -1), 3),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
    ]))
    story.append(Paragraph("Key Metrics", styles["heading"]))
    story.append(table)
    story.append(Spacer(1, 4 * mm))

    # ── Report Sections ──
    for section in sections:
        story.append(Paragraph(section.title, styles["heading"]))
        # Convert markdown bold to ReportLab bold
        content = section.content.replace("**", "<b>").replace("<b>", "</b>", )
        # Simple alternation fix: ensure proper open/close tags
        parts = section.content.split("**")
        formatted = ""
        for i, part in enumerate(parts):
            if i % 2 == 1:
                formatted += f"<b>{part}</b>"
            else:
                formatted += part
        # Replace newlines with <br/>
        formatted = formatted.replace("\n", "<br/>")
        story.append(Paragraph(formatted, styles["body"]))
        story.append(Spacer(1, 3 * mm))

    # ── Footer ──
    story.append(Spacer(1, 10 * mm))
    story.append(Paragraph(
        "<i>Disclaimer: This report is generated by AI for informational purposes only. "
        "It does not constitute financial advice. Always consult a qualified financial "
        "advisor before making investment decisions.</i>",
        styles["small"],
    ))
    story.append(Paragraph(
        f"Generated by SmartStock AI Analyzer | {datetime.now():%Y-%m-%d %H:%M:%S}",
        styles["small"],
    ))

    # Build PDF
    doc.build(story)
    log_agent("PDF", f"[green]Report saved → {pdf_path}[/green]")
    return pdf_path


def _fmt_mc(val: float | None) -> str:
    if val is None:
        return "N/A"
    if val >= 1e12:
        return f"${val / 1e12:.2f}T"
    if val >= 1e9:
        return f"${val / 1e9:.2f}B"
    if val >= 1e6:
        return f"${val / 1e6:.2f}M"
    return f"${val:,.0f}"


def _fmt_pct(val: float | None) -> str:
    if val is None:
        return "N/A"
    return f"{val * 100:.2f}%"

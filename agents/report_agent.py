"""Report agent orchestrator.

Runs the full pipeline and compiles report sections, markdown, and optional PDF bytes.
"""

from __future__ import annotations

import time
from datetime import datetime

from agents.analysis_agent import AnalysisAgent
from agents.base import BaseAgent
from agents.data_agent import DataAgent
from agents.recommendation_agent import RecommendationAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from schemas.agents import (
    AnalysisAgentOutput,
    DataAgentOutput,
    RecommendationAgentOutput,
    ReportAgentOutput,
    ReportSection,
    ResearchAgentOutput,
    SentimentAgentOutput,
)
from schemas.enums import ReportDepth
from utils.gemini import gemini_client
from utils.helpers import fmt_number, fmt_pct
from utils.logger import log_agent

_EXEC_SUMMARY_PROMPT = """\
You are a senior equity analyst.
Write a concise 5-7 sentence executive summary in Korean using the provided context.
Cover: current status, key drivers, risk factors, and recommendation.
Return plain text only. No markdown, no JSON.
"""


class ReportAgent(BaseAgent):
    """Orchestrates all agents and builds final report output."""

    name = "ReportAgent"

    def run(
        self,
        ticker: str,
        depth: ReportDepth = ReportDepth.STANDARD,
    ) -> ReportAgentOutput:
        start_time = time.time()
        gemini_client.reset_counters()
        log_agent(self.name, f"Starting {depth.value} analysis for [bold]{ticker}[/bold]")

        data = DataAgent().run(ticker)
        research = ResearchAgent().run(data, depth)
        sentiment = SentimentAgent().run(data, research, depth)
        analysis = AnalysisAgent().run(data, sentiment, depth)
        recommendation = RecommendationAgent().run(data, analysis, sentiment, depth)

        sections = self._build_sections(data, research, sentiment, analysis, recommendation, depth)

        if depth == ReportDepth.DEEP:
            exec_summary = self._generate_executive_summary(
                data, research, sentiment, analysis, recommendation, depth
            )
        else:
            exec_summary = self._template_summary(data, analysis, recommendation)

        markdown_report = self._build_markdown_report(exec_summary, sections)
        pdf_bytes = self._generate_pdf_bytes(data, recommendation, sections, exec_summary, depth)

        elapsed = time.time() - start_time
        log_agent(self.name, f"[green]Report complete in {elapsed:.1f}s[/green]")

        return ReportAgentOutput(
            ticker=data.ticker,
            report_depth=depth,
            executive_summary=exec_summary,
            sections=sections,
            pdf_bytes=pdf_bytes,
            markdown_report=markdown_report,
            gemini_calls_used=gemini_client.call_count,
            total_tokens_used=gemini_client.total_tokens,
            generated_at=datetime.now(),
            generation_time_s=round(elapsed, 2),
        )

    def _build_sections(
        self,
        data: DataAgentOutput,
        research: ResearchAgentOutput,
        sentiment: SentimentAgentOutput,
        analysis: AnalysisAgentOutput,
        recommendation: RecommendationAgentOutput,
        depth: ReportDepth,
    ) -> list[ReportSection]:
        f = data.fundamentals
        sections: list[ReportSection] = []

        sections.append(
            ReportSection(
                title="Market Data",
                content=(
                    f"**{data.company_name}** ({data.ticker}) - {data.sector.value}\n\n"
                    f"- Price: ${data.price.current:.2f} ({fmt_pct(data.price.change_pct)})\n"
                    f"- 52W Range: ${data.price.low_52w:.2f} ~ ${data.price.high_52w:.2f}\n"
                    f"- Market Cap: {fmt_number(f.market_cap)}\n"
                    f"- PER/PBR: {f.per if f.per is not None else 'N/A'} / "
                    f"{f.pbr if f.pbr is not None else 'N/A'}\n"
                    f"- ROE: {f.roe if f.roe is not None else 'N/A'}\n"
                    f"- Volume: {data.volume.current:,} ({data.volume.relative_volume:.2f}x avg)\n"
                    f"- Data anomalies: {', '.join(data.anomalies) if data.anomalies else 'None'}"
                ),
            )
        )

        if depth == ReportDepth.DEEP:
            themes = "\n".join(f"- {t}" for t in research.key_themes) if research.key_themes else "- N/A"
            timeline = (
                "\n".join(f"- {item.date}: {item.event}" for item in research.timeline[:8])
                if research.timeline
                else "- N/A"
            )
            sections.append(
                ReportSection(
                    title="Research",
                    content=(
                        f"- News articles: {research.news_count}\n"
                        f"- Data quality: {research.data_quality}\n\n"
                        f"**Key themes**\n{themes}\n\n"
                        f"**Timeline**\n{timeline}\n\n"
                        f"**Warnings**\n"
                        f"{chr(10).join(f'- {w}' for w in research.warnings) if research.warnings else '- None'}"
                    ),
                )
            )

        if depth in (ReportDepth.STANDARD, ReportDepth.DEEP):
            pros = "\n".join(f"- {p}" for p in sentiment.pros[:6]) if sentiment.pros else "- N/A"
            cons = "\n".join(f"- {c}" for c in sentiment.cons[:6]) if sentiment.cons else "- N/A"
            cites = (
                "\n".join(
                    f"- {c.source} ({c.timestamp}) {c.url}".strip()
                    for c in sentiment.citations[:5]
                )
                if sentiment.citations
                else "- N/A"
            )
            sections.append(
                ReportSection(
                    title="Sentiment",
                    content=(
                        f"- Label: {sentiment.sentiment_label}\n"
                        f"- Score: {sentiment.sentiment_score:+.2f}\n\n"
                        f"**Pros**\n{pros}\n\n"
                        f"**Cons**\n{cons}\n\n"
                        f"**Citations**\n{cites}"
                    ),
                )
            )

        key_drivers = "\n".join(f"- {d}" for d in analysis.key_drivers) if analysis.key_drivers else "- N/A"
        sections.append(
            ReportSection(
                title="Scenario Analysis",
                content=(
                    f"**Bull case**\n"
                    f"- Thesis: {analysis.bull_case.thesis or 'N/A'}\n"
                    f"- Catalysts: {', '.join(analysis.bull_case.catalysts) if analysis.bull_case.catalysts else 'N/A'}\n"
                    f"- Risks: {', '.join(analysis.bull_case.risks) if analysis.bull_case.risks else 'N/A'}\n\n"
                    f"**Base case**\n"
                    f"- Thesis: {analysis.base_case.thesis or 'N/A'}\n"
                    f"- Drivers: {', '.join(analysis.base_case.drivers) if analysis.base_case.drivers else 'N/A'}\n\n"
                    f"**Bear case**\n"
                    f"- Thesis: {analysis.bear_case.thesis or 'N/A'}\n"
                    f"- Risks: {', '.join(analysis.bear_case.risks) if analysis.bear_case.risks else 'N/A'}\n"
                    f"- Warning: {analysis.bear_case.warning or 'N/A'}\n\n"
                    f"**Key drivers**\n{key_drivers}"
                ),
            )
        )

        confidence_ratio = self._confidence_to_ratio(recommendation.confidence)
        sections.append(
            ReportSection(
                title="Recommendation",
                content=(
                    f"- Rating: {recommendation.rating}\n"
                    f"- Confidence: {confidence_ratio:.0%}\n"
                    f"- Risk notes: {recommendation.risk_notes or 'N/A'}\n\n"
                    f"**Rationale**\n"
                    f"{chr(10).join(f'- {r}' for r in recommendation.rationale) if recommendation.rationale else '- N/A'}\n\n"
                    f"**Invalidation triggers**\n"
                    f"{chr(10).join(f'- {t}' for t in recommendation.invalidation_triggers) if recommendation.invalidation_triggers else '- N/A'}"
                ),
            )
        )

        return sections

    def _generate_executive_summary(
        self,
        data: DataAgentOutput,
        research: ResearchAgentOutput,
        sentiment: SentimentAgentOutput,
        analysis: AnalysisAgentOutput,
        recommendation: RecommendationAgentOutput,
        depth: ReportDepth,
    ) -> str:
        context = (
            f"Ticker: {data.ticker}\n"
            f"Company: {data.company_name}\n"
            f"Price: ${data.price.current:.2f} ({data.price.change_pct:+.2f}%)\n"
            f"Research themes: {', '.join(research.key_themes) if research.key_themes else 'N/A'}\n"
            f"Sentiment: {sentiment.sentiment_label} ({sentiment.sentiment_score:+.2f})\n"
            f"Bull thesis: {analysis.bull_case.thesis}\n"
            f"Base thesis: {analysis.base_case.thesis}\n"
            f"Bear thesis: {analysis.bear_case.thesis}\n"
            f"Recommendation: {recommendation.rating}\n"
            f"Confidence: {self._confidence_to_ratio(recommendation.confidence):.0%}\n"
            f"Rationale: {', '.join(recommendation.rationale) if recommendation.rationale else 'N/A'}"
        )
        try:
            return self.call_gemini(_EXEC_SUMMARY_PROMPT, context, depth)
        except Exception as exc:
            log_agent(self.name, f"[yellow]Executive summary fallback: {exc}[/yellow]")
            return self._template_summary(data, analysis, recommendation)

    @staticmethod
    def _template_summary(
        data: DataAgentOutput,
        analysis: AnalysisAgentOutput,
        recommendation: RecommendationAgentOutput,
    ) -> str:
        confidence_ratio = ReportAgent._confidence_to_ratio(recommendation.confidence)
        return (
            f"{data.company_name} ({data.ticker}) is trading at ${data.price.current:.2f} "
            f"with a daily move of {fmt_pct(data.price.change_pct)}. "
            f"Scenario analysis indicates a base thesis of "
            f"'{analysis.base_case.thesis or 'insufficient evidence'}'. "
            f"The current recommendation is {recommendation.rating} "
            f"with {confidence_ratio:.0%} confidence. "
            f"Primary drivers include "
            f"{', '.join(analysis.key_drivers[:3]) if analysis.key_drivers else 'limited data signals'}."
        )

    @staticmethod
    def _build_markdown_report(executive_summary: str, sections: list[ReportSection]) -> str:
        parts = ["# Executive Summary", executive_summary.strip(), ""]
        for section in sections:
            parts.append(f"# {section.title}")
            parts.append(section.content.strip())
            parts.append("")
        return "\n".join(parts).strip()

    def _generate_pdf_bytes(
        self,
        data: DataAgentOutput,
        recommendation: RecommendationAgentOutput,
        sections: list[ReportSection],
        executive_summary: str,
        depth: ReportDepth,
    ) -> bytes | None:
        try:
            from reporting.generator import generate_pdf

            pdf_path = generate_pdf(
                ticker=data.ticker,
                depth=depth,
                sections=sections,
                executive_summary=executive_summary,
                data=data,
                recommendation=recommendation,
            )
            with open(pdf_path, "rb") as f:
                return f.read()
        except Exception as exc:
            log_agent(self.name, f"[yellow]PDF generation skipped: {exc}[/yellow]")
            return None

    @staticmethod
    def _confidence_to_ratio(value: float | int | None) -> float:
        if value is None:
            return 0.0
        try:
            conf = float(value)
        except (TypeError, ValueError):
            return 0.0
        if conf > 1.0:
            conf /= 100.0
        return max(0.0, min(1.0, conf))

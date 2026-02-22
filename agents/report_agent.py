"""
SmartStock AI Analyzer — ReportAgent (Orchestrator)
Runs the full pipeline, merges all outputs, and triggers PDF generation.
"""

from __future__ import annotations

import time
from datetime import datetime

from agents.base import BaseAgent
from agents.data_agent import DataAgent
from agents.research_agent import ResearchAgent
from agents.sentiment_agent import SentimentAgent
from agents.analysis_agent import AnalysisAgent
from agents.recommendation_agent import RecommendationAgent
from schemas.agents import (
    DataAgentOutput,
    RecommendationAgentOutput,
    ReportAgentOutput,
    ReportSection,
    ResearchAgentOutput,
    SentimentAgentOutput,
    AnalysisAgentOutput,
)
from schemas.enums import ReportDepth
from utils.gemini import gemini_client
from utils.logger import log_agent
from utils.helpers import fmt_number, fmt_pct

_EXEC_SUMMARY_PROMPT = """\
You are a senior analyst writing an executive summary for a stock report.
Given the analysis results below, write a 5-7 sentence executive summary
covering: current status, key findings, risks, and final recommendation.
Return ONLY the summary text, no JSON."""


class ReportAgent(BaseAgent):
    """Orchestrates the full analysis pipeline and compiles the report."""

    name = "ReportAgent"

    def run(
        self,
        ticker: str,
        depth: ReportDepth = ReportDepth.STANDARD,
    ) -> ReportAgentOutput:

        start_time = time.time()
        gemini_client.reset_counters()
        log_agent(self.name, f"Starting {depth.value} analysis for [bold]{ticker}[/bold]")

        # ── Stage 1: Data (no Gemini call) ──
        data = DataAgent().run(ticker)

        # ── Stage 2: Research (Deep only) ──
        research = ResearchAgent().run(data, depth)

        # ── Stage 3: Sentiment (Standard + Deep) ──
        sentiment = SentimentAgent().run(data, depth)

        # ── Stage 4: Analysis (all modes) ──
        analysis = AnalysisAgent().run(data, sentiment, depth)

        # ── Stage 5: Recommendation (all modes) ──
        recommendation = RecommendationAgent().run(data, analysis, sentiment, depth)

        # ── Stage 6: Executive Summary (Deep: Gemini call, others: template) ──
        sections = self._build_sections(data, research, sentiment, analysis, recommendation, depth)

        if depth == ReportDepth.DEEP:
            exec_summary = self._generate_executive_summary(
                data, research, sentiment, analysis, recommendation, depth
            )
        else:
            exec_summary = self._template_summary(data, analysis, recommendation)

        # ── PDF generation ──
        pdf_path = None
        try:
            from reporting.generator import generate_pdf
            pdf_path = generate_pdf(
                ticker=data.ticker,
                depth=depth,
                sections=sections,
                executive_summary=exec_summary,
                data=data,
                recommendation=recommendation,
            )
        except Exception as e:
            log_agent(self.name, f"[red]PDF generation failed: {e}[/red]")

        elapsed = time.time() - start_time
        log_agent(self.name, f"[green]✓ Report complete in {elapsed:.1f}s[/green]")

        return ReportAgentOutput(
            ticker=data.ticker,
            report_depth=depth,
            executive_summary=exec_summary,
            sections=sections,
            pdf_path=pdf_path,
            gemini_calls_used=gemini_client.call_count,
            total_tokens_used=gemini_client.total_tokens,
            generated_at=datetime.now(),
            generation_time_s=round(elapsed, 2),
        )

    # ── Private helpers ──

    def _build_sections(
        self,
        data: DataAgentOutput,
        research: ResearchAgentOutput,
        sentiment: SentimentAgentOutput,
        analysis: AnalysisAgentOutput,
        recommendation: RecommendationAgentOutput,
        depth: ReportDepth,
    ) -> list[ReportSection]:
        sections = []

        # Market Data section (always)
        sections.append(ReportSection(
            title="Market Data",
            content=(
                f"**{data.company_name}** ({data.ticker}) — {data.sector.value}\n\n"
                f"- Price: ${data.price.current:.2f} ({fmt_pct(data.price.change_pct)})\n"
                f"- 52W Range: ${data.price.low_52w:.2f} – ${data.price.high_52w:.2f}\n"
                f"- Market Cap: {fmt_number(data.fundamentals.market_cap)}\n"
                f"- P/E: {data.fundamentals.pe_ratio or 'N/A'}\n"
                f"- Volume: {data.volume.current:,} ({data.volume.relative_volume}x avg)"
            ),
        ))

        # Research section (Deep only)
        if depth == ReportDepth.DEEP:
            comp_list = ", ".join(c.name for c in research.competitors) if research.competitors else "N/A"
            sections.append(ReportSection(
                title="Company Research",
                content=(
                    f"{research.company_summary}\n\n"
                    f"**Business Model:** {research.business_model}\n\n"
                    f"**Competitive Moat:** {research.moat}\n\n"
                    f"**Key Competitors:** {comp_list}\n\n"
                    f"**Industry Outlook:** {research.industry_outlook}"
                ),
            ))

        # Sentiment section (Standard + Deep)
        if depth in (ReportDepth.STANDARD, ReportDepth.DEEP):
            news_lines = "\n".join(
                f"- {n.headline} ({n.sentiment.value})" for n in sentiment.news_items[:5]
            )
            sections.append(ReportSection(
                title="Sentiment Analysis",
                content=(
                    f"**Overall:** {sentiment.overall_sentiment.value} "
                    f"(confidence: {sentiment.confidence:.0%}, score: {sentiment.score:+.2f})\n\n"
                    f"{sentiment.summary}\n\n"
                    f"**Recent News:**\n{news_lines}"
                ),
            ))

        # Technical & Fundamental Analysis (always)
        tech_lines = "\n".join(
            f"- {ind.name}: {ind.value:.2f} → {ind.signal.value}" for ind in analysis.technical_indicators[:6]
        )
        fund_lines = "\n".join(
            f"- {v.metric}: {v.assessment} — {v.detail}" for v in analysis.fundamental_verdicts[:5]
        )
        sections.append(ReportSection(
            title="Analysis",
            content=(
                f"**Combined Signal:** {analysis.combined_signal.value}\n\n"
                f"**Technical ({analysis.technical_signal.value}):**\n{tech_lines}\n\n"
                f"**Fundamental ({analysis.fundamental_signal.value}):**\n{fund_lines}\n\n"
                f"**Support:** {', '.join(f'${s:.2f}' for s in analysis.support_levels)}\n"
                f"**Resistance:** {', '.join(f'${r:.2f}' for r in analysis.resistance_levels)}\n\n"
                f"{analysis.summary}"
            ),
        ))

        # Recommendation (always)
        sections.append(ReportSection(
            title="Recommendation",
            content=(
                f"**Signal:** {recommendation.signal.value} "
                f"(confidence: {recommendation.confidence:.0%})\n\n"
                f"**Target Price:** ${recommendation.target_price:.2f}\n"
                f"**Time Horizon:** {recommendation.time_horizon}\n\n"
                f"**Risk/Reward:** {recommendation.risk_reward.risk_reward_ratio:.2f}x "
                f"(upside ${recommendation.risk_reward.upside_target:.2f} / "
                f"downside ${recommendation.risk_reward.downside_risk:.2f})\n\n"
                f"**Rationale:** {recommendation.rationale}\n\n"
                f"**Key Risks:**\n" + "\n".join(f"- {r}" for r in recommendation.key_risks) + "\n\n"
                f"**Action Items:**\n" + "\n".join(f"- {a}" for a in recommendation.action_items)
            ),
        ))

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
        """Generate AI-written executive summary (Deep mode Gemini call)."""
        context = (
            f"Stock: {data.company_name} ({data.ticker})\n"
            f"Price: ${data.price.current:.2f} ({data.price.change_pct:+.2f}%)\n"
            f"Research: {research.company_summary}\n"
            f"Sentiment: {sentiment.overall_sentiment.value} ({sentiment.score:+.2f})\n"
            f"Analysis Signal: {analysis.combined_signal.value}\n"
            f"Recommendation: {recommendation.signal.value} "
            f"(target: ${recommendation.target_price or 0:.2f})\n"
            f"Rationale: {recommendation.rationale}"
        )
        return self.call_gemini(_EXEC_SUMMARY_PROMPT, context, depth)

    @staticmethod
    def _template_summary(
        data: DataAgentOutput,
        analysis: AnalysisAgentOutput,
        recommendation: RecommendationAgentOutput,
    ) -> str:
        """Simple template-based summary for Quick/Standard modes."""
        return (
            f"{data.company_name} ({data.ticker}) is currently trading at "
            f"${data.price.current:.2f} ({fmt_pct(data.price.change_pct)}). "
            f"Our analysis produces a {analysis.combined_signal.value} signal "
            f"based on technical and fundamental factors. "
            f"The recommendation is {recommendation.signal.value} with "
            f"{recommendation.confidence:.0%} confidence"
            f"{f', targeting ${recommendation.target_price:.2f}' if recommendation.target_price else ''}. "
            f"{recommendation.rationale}"
        )

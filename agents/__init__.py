"""Agents package exports.

Uses lazy imports to avoid package-level circular import failures.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

__all__ = [
    "BaseAgent",
    "DataAgent",
    "ResearchAgent",
    "SentimentAgent",
    "AnalysisAgent",
    "RecommendationAgent",
    "ReportAgent",
]

if TYPE_CHECKING:
    from agents.analysis_agent import AnalysisAgent
    from agents.base import BaseAgent
    from agents.data_agent import DataAgent
    from agents.recommendation_agent import RecommendationAgent
    from agents.report_agent import ReportAgent
    from agents.research_agent import ResearchAgent
    from agents.sentiment_agent import SentimentAgent


def __getattr__(name: str):
    if name == "BaseAgent":
        from agents.base import BaseAgent

        return BaseAgent
    if name == "DataAgent":
        from agents.data_agent import DataAgent

        return DataAgent
    if name == "ResearchAgent":
        from agents.research_agent import ResearchAgent

        return ResearchAgent
    if name == "SentimentAgent":
        from agents.sentiment_agent import SentimentAgent

        return SentimentAgent
    if name == "AnalysisAgent":
        from agents.analysis_agent import AnalysisAgent

        return AnalysisAgent
    if name == "RecommendationAgent":
        from agents.recommendation_agent import RecommendationAgent

        return RecommendationAgent
    if name == "ReportAgent":
        from agents.report_agent import ReportAgent

        return ReportAgent
    raise AttributeError(f"module 'agents' has no attribute '{name}'")

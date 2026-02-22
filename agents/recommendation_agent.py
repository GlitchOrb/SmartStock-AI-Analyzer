"""
SmartStock AI Analyzer — RecommendationAgent
Produces Buy/Hold/Sell recommendation via Gemini.
Only called in Standard/Deep mode. Skipped in Quick mode (handled by AnalysisAgent).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.agents import AnalysisAgentOutput, RecommendationAgentOutput
from schemas.enums import ReportDepth
from utils.gemini import gemini_client
from utils.logger import log_agent


def _strip_json_fences(text: str) -> str:
    """Strip ```json fences and leading/trailing whitespace before json.loads()."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _safe_json_loads(text: str) -> dict:
    """Safely parse JSON with fence stripping. Returns empty dict on failure."""
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


_SYSTEM_PROMPT = """\
당신은 시니어 포트폴리오 전략가입니다. 제공된 분석 결과를 바탕으로 투자 추천을 생성하세요.

Return STRICT JSON with this exact structure. No markdown. No explanation. JSON only:
{
    "rating": "Buy|Hold|Sell",
    "confidence": 0,
    "rationale": [],
    "invalidation_triggers": [],
    "risk_notes": ""
}

Respond in Korean."""


_RETRY_SYSTEM_PROMPT = """\
이전 응답이 유효한 JSON이 아니었습니다. 반드시 순수 JSON만 반환하세요.
마크다운 없음. 설명 없음. 코드 펜스 없음. JSON만 반환하세요.
Respond in Korean."""


def _safe_default_recommendation(ticker: str) -> dict:
    """Safe fallback dict when all Gemini parsing fails."""
    return {
        "rating": "Hold",
        "confidence": 0,
        "rationale": ["AI 분석 실패로 인한 기본 Hold 등급"],
        "invalidation_triggers": [],
        "risk_notes": "AI 분석 실패. 수동 확인 필요.",
    }


def run_recommendation_agent(
    analysis: AnalysisAgentOutput,
    depth: ReportDepth,
) -> RecommendationAgentOutput:
    """
    Run the RecommendationAgent.
    - Quick mode: SKIP (return empty output — handled by AnalysisAgent)
    - Standard/Deep mode: 1 Gemini call
    Returns RecommendationAgentOutput.
    """
    # Quick mode: skip, recommendation is already in AnalysisAgentOutput
    if depth == ReportDepth.QUICK:
        log_agent("RecommendationAgent", "Quick 모드 — 건너뜀 (AnalysisAgent에서 처리)")
        return RecommendationAgentOutput(
            ticker=analysis.ticker,
            rating=analysis.rating or "Hold",
            confidence=analysis.confidence,
            rationale=analysis.rationale,
            invalidation_triggers=analysis.invalidation_triggers,
            risk_notes="",
            generated_at=datetime.now(),
        )

    # Build context from analysis
    user_prompt = (
        f"종목: {analysis.ticker}\n\n"
        f"=== 시나리오 분석 결과 ===\n\n"
        f"강세 시나리오 (Bull Case):\n"
        f"  논거: {analysis.bull_case.thesis}\n"
        f"  촉매: {', '.join(analysis.bull_case.catalysts) if analysis.bull_case.catalysts else 'N/A'}\n"
        f"  위험: {', '.join(analysis.bull_case.risks) if analysis.bull_case.risks else 'N/A'}\n\n"
        f"기본 시나리오 (Base Case):\n"
        f"  논거: {analysis.base_case.thesis}\n"
        f"  동인: {', '.join(analysis.base_case.drivers) if analysis.base_case.drivers else 'N/A'}\n\n"
        f"약세 시나리오 (Bear Case):\n"
        f"  논거: {analysis.bear_case.thesis}\n"
        f"  위험: {', '.join(analysis.bear_case.risks) if analysis.bear_case.risks else 'N/A'}\n"
        f"  경고: {analysis.bear_case.warning}\n\n"
        f"핵심 동인: {', '.join(analysis.key_drivers) if analysis.key_drivers else 'N/A'}\n\n"
        f"위 분석을 기반으로 투자 추천을 생성하세요.\n"
        f"Respond in Korean."
    )

    parsed: dict = {}

    # Attempt #1
    try:
        log_agent("RecommendationAgent", f"Gemini 호출 중 ({depth.value} 모드)...")
        response = gemini_client.invoke("RecommendationAgent", _SYSTEM_PROMPT, user_prompt, depth)
        parsed = _safe_json_loads(response)
    except Exception as e:
        log_agent("RecommendationAgent", f"[red]Gemini 호출 실패: {e}[/red]")

    # If parsing failed, retry once with stricter prompt
    if not parsed or "rating" not in parsed:
        log_agent("RecommendationAgent", "[yellow]JSON 파싱 실패 → 재시도 중...[/yellow]")
        try:
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"이전 응답의 JSON이 유효하지 않았습니다. "
                f"반드시 위 구조의 순수 JSON만 반환하세요.\n"
                f"Respond in Korean."
            )
            response2 = gemini_client.invoke("RecommendationAgent", _RETRY_SYSTEM_PROMPT, retry_prompt, depth)
            parsed = _safe_json_loads(response2)
        except Exception as e2:
            log_agent("RecommendationAgent", f"[red]재시도 실패: {e2}[/red]")

    # If still failed, use safe default
    if not parsed or "rating" not in parsed:
        log_agent("RecommendationAgent", "[red]안전 기본값 사용[/red]")
        parsed = _safe_default_recommendation(analysis.ticker)

    # Validate rating
    rating = parsed.get("rating", "Hold")
    if rating not in ("Buy", "Hold", "Sell"):
        rating = "Hold"

    return RecommendationAgentOutput(
        ticker=analysis.ticker,
        rating=rating,
        confidence=parsed.get("confidence", 0),
        rationale=parsed.get("rationale", []),
        invalidation_triggers=parsed.get("invalidation_triggers", []),
        risk_notes=parsed.get("risk_notes", ""),
        generated_at=datetime.now(),
    )

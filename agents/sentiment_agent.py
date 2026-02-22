"""
SmartStock AI Analyzer — SentimentAgent
Uses retrieved news chunks + Gemini to produce sentiment analysis.
Standard/Deep mode only. Quick mode defaults to neutral.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.agents import Citation, ResearchAgentOutput, SentimentAgentOutput
from schemas.enums import ReportDepth
from utils.gemini import gemini_client
from utils.logger import log_agent


def _strip_json_fences(text: str) -> str:
    """Strip ```json fences and leading/trailing whitespace."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = [line for line in lines if not line.strip().startswith("```")]
        cleaned = "\n".join(lines).strip()
    return cleaned


def _safe_json_loads(text: str) -> dict:
    """Safely parse JSON with fence stripping."""
    cleaned = _strip_json_fences(text)
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {}


_SYSTEM_PROMPT = """\
당신은 시니어 마켓 센티멘트 애널리스트입니다.
제공된 뉴스 기사들의 감성을 분석하여 종합적인 시장 심리를 평가하세요.

Return STRICT JSON. No markdown. No explanation. JSON only:
{
    "sentiment_score": 0.0,
    "sentiment_label": "positive|neutral|negative",
    "pros": [],
    "cons": [],
    "citations": [{"text": "", "source": "", "url": "", "timestamp": ""}]
}

sentiment_score: -1.0 (매우 부정적) ~ 1.0 (매우 긍정적)
sentiment_label: 반드시 "positive", "neutral", "negative" 중 하나
pros: 해당 종목의 긍정적 요인들
cons: 해당 종목의 부정적 요인들
citations: 근거가 되는 뉴스 인용 (최소 2개, 최대 5개)

Respond in Korean."""


_RETRY_SYSTEM_PROMPT = """\
이전 응답이 유효한 JSON이 아니었습니다. 반드시 순수 JSON만 반환하세요.
마크다운 없음. 설명 없음. 코드 펜스 없음. JSON만 반환하세요.
Respond in Korean."""


def run_sentiment_agent(
    research: ResearchAgentOutput,
    depth: ReportDepth,
) -> SentimentAgentOutput:
    """
    Run the SentimentAgent.
    - Quick mode: SKIP → default neutral sentiment
    - Standard/Deep mode: 1 Gemini call with retrieved chunks

    If news count < 3: force neutral, add warning to cons[].
    Returns SentimentAgentOutput.
    """
    ticker = research.ticker
    all_warnings: list[str] = list(research.warnings)

    # Quick mode → skip
    if depth == ReportDepth.QUICK:
        log_agent("SentimentAgent", "Quick 모드 — 건너뜀 (기본 중립)")
        return SentimentAgentOutput(
            ticker=ticker,
            sentiment_score=0.0,
            sentiment_label="neutral",
            pros=[],
            cons=[],
            citations=[],
            warnings=["Quick 모드: 센티멘트 분석 건너뜀"],
            generated_at=datetime.now(),
        )

    # If news count < 3 → force neutral with warning
    if research.news_count < 3:
        log_agent("SentimentAgent", f"[yellow]뉴스 {research.news_count}건 < 3 → 중립 기본값[/yellow]")
        warn_msg = f"뉴스 기사 {research.news_count}건으로 센티멘트 분석이 제한됩니다."
        all_warnings.append(warn_msg)
        return SentimentAgentOutput(
            ticker=ticker,
            sentiment_score=0.0,
            sentiment_label="neutral",
            pros=[],
            cons=[warn_msg],
            citations=[],
            warnings=all_warnings,
            generated_at=datetime.now(),
        )

    # Build context from retrieved chunks
    chunks = research.retrieved_chunks
    if not chunks:
        log_agent("SentimentAgent", "[yellow]검색된 청크 없음 → 중립 기본값[/yellow]")
        all_warnings.append("검색된 뉴스 청크가 없어 센티멘트 분석이 제한됩니다.")
        return SentimentAgentOutput(
            ticker=ticker,
            sentiment_score=0.0,
            sentiment_label="neutral",
            pros=[],
            cons=["검색된 뉴스 청크 없음"],
            citations=[],
            warnings=all_warnings,
            generated_at=datetime.now(),
        )

    chunks_text = "\n\n---\n\n".join(
        f"[출처: {c.get('source', 'N/A')} | 시간: {c.get('timestamp', 'N/A')} | URL: {c.get('url', '')}]\n{c['content']}"
        for c in chunks
    )

    user_prompt = (
        f"종목: {ticker}\n"
        f"수집된 뉴스 기사: {research.news_count}건\n"
        f"리서치 핵심 테마: {', '.join(research.key_themes) if research.key_themes else 'N/A'}\n\n"
        f"=== 관련 뉴스 기사 ===\n\n"
        f"{chunks_text}\n\n"
        f"위 뉴스 기사들의 감성을 분석하세요.\n"
        f"Respond in Korean."
    )

    parsed: dict = {}

    # Attempt #1
    try:
        log_agent("SentimentAgent", f"Gemini 호출 중 ({depth.value} 모드)...")
        response = gemini_client.invoke("SentimentAgent", _SYSTEM_PROMPT, user_prompt, depth)
        parsed = _safe_json_loads(response)
    except Exception as e:
        log_agent("SentimentAgent", f"[red]Gemini 호출 실패: {e}[/red]")
        all_warnings.append(f"AI 센티멘트 분석 실패: {e}")

    # If parsing failed, retry once
    if not parsed or "sentiment_label" not in parsed:
        log_agent("SentimentAgent", "[yellow]JSON 파싱 실패 → 재시도 중...[/yellow]")
        try:
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"이전 응답의 JSON이 유효하지 않았습니다. "
                f"반드시 위 구조의 순수 JSON만 반환하세요.\n"
                f"Respond in Korean."
            )
            response2 = gemini_client.invoke("SentimentAgent", _RETRY_SYSTEM_PROMPT, retry_prompt, depth)
            parsed = _safe_json_loads(response2)
        except Exception as e2:
            log_agent("SentimentAgent", f"[red]재시도 실패: {e2}[/red]")
            all_warnings.append(f"AI 재시도 실패: {e2}")

    # If still failed, use safe defaults
    if not parsed or "sentiment_label" not in parsed:
        log_agent("SentimentAgent", "[red]안전 기본값 사용[/red]")
        all_warnings.append("AI 분석 불가 — 중립 기본값 사용")
        return SentimentAgentOutput(
            ticker=ticker,
            sentiment_score=0.0,
            sentiment_label="neutral",
            pros=[],
            cons=["AI 분석 불가"],
            citations=[],
            warnings=all_warnings,
            generated_at=datetime.now(),
        )

    # Validate sentiment_label
    label = parsed.get("sentiment_label", "neutral")
    if label not in ("positive", "neutral", "negative"):
        label = "neutral"

    # Clamp sentiment_score
    score = parsed.get("sentiment_score", 0.0)
    try:
        score = float(score)
        score = max(-1.0, min(1.0, score))
    except (TypeError, ValueError):
        score = 0.0

    # Build citations
    citations_raw = parsed.get("citations", [])
    citations: list[Citation] = []
    for c in citations_raw:
        if isinstance(c, dict):
            citations.append(Citation(
                text=c.get("text", ""),
                source=c.get("source", ""),
                url=c.get("url", ""),
                timestamp=c.get("timestamp", ""),
            ))

    return SentimentAgentOutput(
        ticker=ticker,
        sentiment_score=score,
        sentiment_label=label,
        pros=parsed.get("pros", []),
        cons=parsed.get("cons", []),
        citations=citations,
        warnings=all_warnings,
        generated_at=datetime.now(),
    )

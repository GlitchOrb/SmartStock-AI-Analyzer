"""
SmartStock AI Analyzer — AnalysisAgent
Scenario-based analysis via Gemini: bull/base/bear cases + key drivers.
Stage 3: Now accepts SentimentAgentOutput for qualitative context.
Quick mode: merges analysis + recommendation into one call.
Standard/Deep mode: analysis-only call (recommendation is separate).
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from schemas.agents import (
    AnalysisAgentOutput,
    BaseCase,
    BearCase,
    BullCase,
    DataAgentOutput,
    SentimentAgentOutput,
)
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


def _build_data_context(
    data: DataAgentOutput,
    sentiment: SentimentAgentOutput | None = None,
) -> str:
    """Build a human-readable data context string for the prompt."""
    f = data.fundamentals
    t = data.technicals
    p = data.price

    lines = [
        f"종목: {data.company_name} ({data.ticker})",
        f"섹터: {data.sector.value}",
        f"현재가: ${p.current:.2f} (일간 변동: {p.change_pct:+.2f}%)",
        f"52주 최고: ${p.high_52w:.2f}, 52주 최저: ${p.low_52w:.2f}",
        f"거래량: {data.volume.current:,} (상대거래량: {data.volume.relative_volume:.2f}x)",
        "",
        "--- 펀더멘탈 ---",
        f"PER: {f.per}", f"PBR: {f.pbr}", f"ROE: {f.roe}",
        f"부채비율: {f.debt_ratio}", f"매출: {f.revenue}", f"영업이익률: {f.operating_profit}",
        f"시가총액: {f.market_cap}", f"EPS: {f.eps}", f"배당수익률: {f.dividend_yield}",
        "",
        "--- 기술적 지표 ---",
        f"RSI(14): {t.rsi_14}", f"MACD: {t.macd}",
        f"MACD Signal: {t.macd_signal}", f"MACD Histogram: {t.macd_histogram}",
        f"볼린저 상단: {t.bb_upper}, 중간: {t.bb_middle}, 하단: {t.bb_lower}",
        f"20일 이평: {t.ma_20}", f"60일 이평: {t.ma_60}",
        f"MDD: {t.mdd}%", f"변동성(연환산): {t.volatility}%",
    ]

    if data.anomalies:
        lines.append("")
        lines.append(f"⚠️ 데이터 이상: {', '.join(data.anomalies)}")

    # Add sentiment context if available
    if sentiment is not None:
        lines.append("")
        lines.append("--- 센티멘트 분석 ---")
        lines.append(f"센티멘트 점수: {sentiment.sentiment_score:.2f} (-1.0 ~ 1.0)")
        lines.append(f"센티멘트 레이블: {sentiment.sentiment_label}")
        if sentiment.pros:
            lines.append(f"긍정 요인: {', '.join(sentiment.pros[:5])}")
        if sentiment.cons:
            lines.append(f"부정 요인: {', '.join(sentiment.cons[:5])}")

    return "\n".join(lines)


# ──────────────────────────────────────────────
# Quick mode prompt (merged analysis + recommendation)
# ──────────────────────────────────────────────

_QUICK_SYSTEM_PROMPT = """\
당신은 시니어 주식 애널리스트입니다. 제공된 정량 데이터와 센티멘트 분석을 바탕으로 종합 분석과 투자 추천을 수행하세요.

Return STRICT JSON with this exact structure. No markdown. No explanation. JSON only:
{
    "bull_case": {"thesis": "", "catalysts": [], "risks": []},
    "base_case": {"thesis": "", "drivers": []},
    "bear_case": {"thesis": "", "risks": [], "warning": ""},
    "key_drivers": [],
    "rating": "Buy|Hold|Sell",
    "confidence": 0,
    "rationale": [],
    "invalidation_triggers": []
}

Respond in Korean."""


# ──────────────────────────────────────────────
# Standard/Deep mode prompt (analysis only)
# ──────────────────────────────────────────────

_ANALYSIS_SYSTEM_PROMPT = """\
당신은 시니어 주식 애널리스트입니다. 제공된 정량 데이터와 센티멘트 분석을 바탕으로 시나리오 기반 분석을 수행하세요.

Return STRICT JSON with this exact structure. No markdown. No explanation. JSON only:
{
    "bull_case": {"thesis": "", "catalysts": [], "risks": []},
    "base_case": {"thesis": "", "drivers": []},
    "bear_case": {"thesis": "", "risks": [], "warning": ""},
    "key_drivers": []
}

Respond in Korean."""


# ──────────────────────────────────────────────
# Retry prompt (stricter)
# ──────────────────────────────────────────────

_RETRY_SYSTEM_PROMPT = """\
이전 응답이 유효한 JSON이 아니었습니다. 반드시 순수 JSON만 반환하세요.
마크다운 없음. 설명 없음. 코드 펜스 없음. JSON만 반환하세요.
Respond in Korean."""


# ──────────────────────────────────────────────
# Safe defaults
# ──────────────────────────────────────────────

def _safe_default_analysis(ticker: str, is_quick: bool = False) -> dict:
    """Safe fallback dict when all Gemini parsing fails."""
    base = {
        "bull_case": {"thesis": "AI 분석 실패로 인한 기본값입니다.", "catalysts": [], "risks": []},
        "base_case": {"thesis": "데이터 부족 또는 AI 응답 오류로 분석이 제한됩니다.", "drivers": []},
        "bear_case": {"thesis": "분석 데이터를 확인한 후 재시도하세요.", "risks": [], "warning": "AI 분석 실패"},
        "key_drivers": ["데이터 확인 필요"],
    }
    if is_quick:
        base.update({
            "rating": "Hold",
            "confidence": 0,
            "rationale": ["AI 분석 실패로 인한 기본 Hold 등급"],
            "invalidation_triggers": [],
        })
    return base


# ──────────────────────────────────────────────
# Main function
# ──────────────────────────────────────────────

def run_analysis_agent(
    data: DataAgentOutput,
    depth: ReportDepth,
    sentiment: SentimentAgentOutput | None = None,
) -> AnalysisAgentOutput:
    """
    Run the AnalysisAgent.
    - Quick mode: 1 Gemini call (merged analysis + recommendation)
    - Standard/Deep mode: 1 Gemini call (analysis only)
    - If SentimentAgentOutput is None (Quick mode), proceed with data only.
    Returns AnalysisAgentOutput.
    """
    is_quick = depth == ReportDepth.QUICK
    system_prompt = _QUICK_SYSTEM_PROMPT if is_quick else _ANALYSIS_SYSTEM_PROMPT
    user_prompt = _build_data_context(data, sentiment)

    parsed: dict = {}

    # Attempt #1
    try:
        log_agent("AnalysisAgent", f"Gemini 호출 중 ({depth.value} 모드)...")
        response = gemini_client.invoke("AnalysisAgent", system_prompt, user_prompt, depth)
        parsed = _safe_json_loads(response)
    except Exception as e:
        log_agent("AnalysisAgent", f"[red]Gemini 호출 실패: {e}[/red]")

    # If parsing failed, retry once with stricter prompt
    if not parsed or "bull_case" not in parsed:
        log_agent("AnalysisAgent", "[yellow]JSON 파싱 실패 → 재시도 중...[/yellow]")
        try:
            retry_prompt = (
                f"{user_prompt}\n\n"
                f"이전 응답의 JSON이 유효하지 않았습니다. "
                f"반드시 위 구조의 순수 JSON만 반환하세요.\n"
                f"Respond in Korean."
            )
            response2 = gemini_client.invoke("AnalysisAgent", _RETRY_SYSTEM_PROMPT, retry_prompt, depth)
            parsed = _safe_json_loads(response2)
        except Exception as e2:
            log_agent("AnalysisAgent", f"[red]재시도 실패: {e2}[/red]")

    # If still failed, use safe default
    if not parsed or "bull_case" not in parsed:
        log_agent("AnalysisAgent", "[red]안전 기본값 사용[/red]")
        parsed = _safe_default_analysis(data.ticker, is_quick)

    # Build output
    bull_raw = parsed.get("bull_case", {})
    base_raw = parsed.get("base_case", {})
    bear_raw = parsed.get("bear_case", {})

    return AnalysisAgentOutput(
        ticker=data.ticker,
        bull_case=BullCase(
            thesis=bull_raw.get("thesis", ""),
            catalysts=bull_raw.get("catalysts", []),
            risks=bull_raw.get("risks", []),
        ),
        base_case=BaseCase(
            thesis=base_raw.get("thesis", ""),
            drivers=base_raw.get("drivers", []),
        ),
        bear_case=BearCase(
            thesis=bear_raw.get("thesis", ""),
            risks=bear_raw.get("risks", []),
            warning=bear_raw.get("warning", ""),
        ),
        key_drivers=parsed.get("key_drivers", []),
        rating=parsed.get("rating", "") if is_quick else "",
        confidence=parsed.get("confidence", 0) if is_quick else 0,
        rationale=parsed.get("rationale", []) if is_quick else [],
        invalidation_triggers=parsed.get("invalidation_triggers", []) if is_quick else [],
        generated_at=datetime.now(),
    )

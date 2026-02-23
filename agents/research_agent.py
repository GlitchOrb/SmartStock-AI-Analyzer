"""
SmartStock AI Analyzer — ResearchAgent
Fetches news via RAG loader → builds vectorstore → retrieves top chunks →
calls Gemini for key_themes/timeline/data_quality.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base import BaseAgent
from schemas.agents import ResearchAgentOutput, TimelineEvent, DataAgentOutput
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
당신은 시니어 금융 리서치 애널리스트입니다.
제공된 뉴스 기사들을 분석하여 핵심 테마, 타임라인, 데이터 품질을 평가하세요.

Return STRICT JSON. No markdown. No explanation. JSON only:
{
    "key_themes": [],
    "timeline": [{"date": "", "event": ""}],
    "data_quality": "sufficient|insufficient"
}

Respond in Korean."""


_RETRY_SYSTEM_PROMPT = """\
이전 응답이 유효한 JSON이 아니었습니다. 반드시 순수 JSON만 반환하세요.
마크다운 없음. 설명 없음. 코드 펜스 없음. JSON만 반환하세요.
Respond in Korean."""


class ResearchAgent(BaseAgent):
    """
    Fetches news via RAG loader → builds vectorstore → retrieves top chunks →
    calls Gemini for key_themes/timeline/data_quality.
    """
    name = "ResearchAgent"

    def run(
        self,
        data: DataAgentOutput,
        depth: ReportDepth = ReportDepth.STANDARD,
    ) -> ResearchAgentOutput:
        """
        Run the ResearchAgent pipeline.
        """
        ticker = data.ticker
        from rag.loader import load_rss_news
        from rag.vectorstore import build_vectorstore, retrieve

        all_warnings: list[str] = []
        retrieved_chunks: list[dict] = []

        # 1. Fetch news
        log_agent(self.name, f"뉴스 수집 시작: {ticker}")
        try:
            documents, loader_warnings = load_rss_news(ticker)
            all_warnings.extend(loader_warnings)
        except Exception as e:
            log_agent(self.name, f"[red]뉴스 수집 실패: {e}[/red]")
            all_warnings.append(f"뉴스 수집 실패: {e}")
            documents = []

        news_count = len(documents)

        # If 0 articles → skip RAG, return with warnings
        if news_count == 0:
            log_agent(self.name, "[yellow]뉴스 0건 → RAG 건너뜀, 기본값 반환[/yellow]")
            all_warnings.append("뉴스 기사를 찾을 수 없어 리서치가 제한됩니다.")
            return ResearchAgentOutput(
                ticker=ticker.upper(),
                key_themes=["뉴스 데이터 없음"],
                timeline=[],
                data_quality="insufficient",
                news_count=0,
                retrieved_chunks=[],
                warnings=all_warnings,
                generated_at=datetime.now(),
            )

        # 2. Build vectorstore
        log_agent(self.name, f"벡터스토어 구축 중 ({news_count} 문서)...")
        try:
            vs = build_vectorstore(ticker, documents)
        except Exception as e:
            log_agent(self.name, f"[red]벡터스토어 구축 실패: {e}[/red]")
            all_warnings.append(f"벡터스토어 구축 실패: {e}")
            vs = None

        # 3. Retrieve top-5 chunks
        if vs is not None:
            query = f"{ticker} stock analysis outlook performance"
            retrieved_chunks = retrieve(vs, query, k=5)
        else:
            # Fallback: use raw documents directly
            log_agent(self.name, "[yellow]벡터스토어 없음 → 원본 문서 사용[/yellow]")
            for doc in documents[:5]:
                retrieved_chunks.append({
                    "content": doc.page_content,
                    "source": doc.metadata.get("source", ""),
                    "url": doc.metadata.get("url", ""),
                    "timestamp": doc.metadata.get("timestamp", ""),
                })

        # 4. Call Gemini with chunks as context
        chunks_text = "\n\n---\n\n".join(
            f"[출처: {c.get('source', 'N/A')} | {c.get('timestamp', 'N/A')}]\n{c['content']}"
            for c in retrieved_chunks
        )

        user_prompt = (
            f"종목: {ticker}\n"
            f"수집된 뉴스 기사: {news_count}건\n\n"
            f"=== 관련 뉴스 기사 (상위 {len(retrieved_chunks)}건) ===\n\n"
            f"{chunks_text}\n\n"
            f"위 뉴스 기사들을 분석하여 핵심 테마, 타임라인, 데이터 품질을 평가하세요.\n"
            f"Respond in Korean."
        )

        parsed: dict = {}

        # Attempt #1
        try:
            log_agent(self.name, f"Gemini 호출 중 ({depth.value} 모드)...")
            response = self.call_gemini(_SYSTEM_PROMPT, user_prompt, depth)
            parsed = self.parse_json_response(response)
        except Exception as e:
            log_agent(self.name, f"[red]Gemini 호출 실패: {e}[/red]")
            all_warnings.append(f"AI 분석 실패: {e}")

        # If parsing failed, retry once
        if not parsed or "key_themes" not in parsed:
            log_agent(self.name, "[yellow]JSON 파싱 실패 → 재시도 중...[/yellow]")
            try:
                retry_prompt = (
                    f"{user_prompt}\n\n"
                    f"이전 응답의 JSON이 유효하지 않았습니다. "
                    f"반드시 위 구조의 순수 JSON만 반환하세요.\n"
                    f"Respond in Korean."
                )
                response2 = self.call_gemini(_RETRY_SYSTEM_PROMPT, retry_prompt, depth)
                parsed = self.parse_json_response(response2)
            except Exception as e2:
                log_agent(self.name, f"[red]재시도 실패: {e2}[/red]")
                all_warnings.append(f"AI 재시도 실패: {e2}")

        # If still failed, use safe defaults
        if not parsed or "key_themes" not in parsed:
            log_agent(self.name, "[red]안전 기본값 사용[/red]")
            all_warnings.append("AI 분석 불가 — 기본값을 사용합니다.")
            parsed = {
                "key_themes": ["AI 분석 실패"],
                "timeline": [],
                "data_quality": "insufficient",
            }

        # 5. Build output
        timeline_raw = parsed.get("timeline", [])
        timeline = []
        for item in timeline_raw:
            if isinstance(item, dict):
                timeline.append(TimelineEvent(
                    date=item.get("date", ""),
                    event=item.get("event", ""),
                ))

        return ResearchAgentOutput(
            ticker=ticker.upper(),
            key_themes=parsed.get("key_themes", []),
            timeline=timeline,
            data_quality=parsed.get("data_quality", "insufficient"),
            news_count=news_count,
            retrieved_chunks=retrieved_chunks,
            warnings=all_warnings,
            generated_at=datetime.now(),
        )


def run_research_agent(
    ticker: str,
    period: str = "1y",
    depth: ReportDepth = ReportDepth.STANDARD,
) -> ResearchAgentOutput:
    """Helper for functional calls (maintains compatibility if needed)."""
    from agents.data_agent import DataAgent

    # Keep period in signature for backward compatibility; DataAgent uses 1y internally.
    _ = period
    data = DataAgent().run(ticker)
    return ResearchAgent().run(data, depth)

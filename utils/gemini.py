"""
SmartStock AI Analyzer — Gemini Client Wrapper
Provides a call-counted Gemini invocation layer with 7-second rate-limit sleep.
Free tier: 10 RPM → sleep 7s between calls for safety.
"""

from __future__ import annotations

import time
import threading

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

from schemas.config import settings
from schemas.enums import ReportDepth
from utils.logger import log_gemini_call, log_agent

GEMINI_SLEEP_SECONDS = 7

CALL_BUDGET: dict[ReportDepth, int] = {
    ReportDepth.QUICK: 2,
    ReportDepth.STANDARD: 4,
    ReportDepth.DEEP: 6,
}


class GeminiClient:
    """Thread-safe Gemini client with call counting and rate-limit sleep."""

    def __init__(self) -> None:
        self._llm: ChatGoogleGenerativeAI | None = None
        self._call_count = 0
        self._lock = threading.Lock()
        self._total_tokens = 0
        self._last_call_time: float = 0.0

    @property
    def llm(self) -> ChatGoogleGenerativeAI:
        if self._llm is None:
            self._llm = ChatGoogleGenerativeAI(
                model=settings.model_name,
                google_api_key=settings.gemini_api_key,
                temperature=0.3,
                convert_system_message_to_human=True,
            )
        return self._llm

    @property
    def call_count(self) -> int:
        return self._call_count

    @property
    def total_tokens(self) -> int:
        return self._total_tokens

    def get_budget(self, depth: ReportDepth) -> int:
        return CALL_BUDGET[depth]

    def _enforce_rate_limit(self) -> None:
        """Sleep if less than 7 seconds have passed since the last call."""
        elapsed = time.time() - self._last_call_time
        if self._last_call_time > 0 and elapsed < GEMINI_SLEEP_SECONDS:
            wait = GEMINI_SLEEP_SECONDS - elapsed
            log_agent("RateLimit", f"무료 티어 보호: {wait:.1f}초 대기 중...")
            time.sleep(wait)

    def invoke(
        self,
        agent_name: str,
        system_prompt: str,
        user_prompt: str,
        depth: ReportDepth | None = None,
    ) -> str:
        """
        Make a single Gemini call with rate-limit sleep, logging, and budget tracking.
        Raises on failure — callers must wrap in try/except.
        Returns the model's text response.
        """
        budget = CALL_BUDGET.get(depth, 99) if depth else 99

        self._enforce_rate_limit()

        with self._lock:
            self._call_count += 1
            current = self._call_count

        log_gemini_call(agent_name, current, budget)

        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        response = self.llm.invoke(messages)
        self._last_call_time = time.time()

        if hasattr(response, "usage_metadata") and response.usage_metadata:
            with self._lock:
                self._total_tokens += response.usage_metadata.get("total_tokens", 0)

        log_agent(agent_name, f"응답 수신 완료 ({len(response.content)} chars)")
        return response.content

    def reset_counters(self) -> None:
        """Reset call and token counters for a new analysis run."""
        with self._lock:
            self._call_count = 0
            self._total_tokens = 0


gemini_client = GeminiClient()

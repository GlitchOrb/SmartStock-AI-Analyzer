"""
SmartStock AI Analyzer — Base Agent
Abstract base class that all agents extend.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel

from schemas.enums import ReportDepth
from utils.gemini import gemini_client
from utils.logger import log_agent


class BaseAgent(ABC):
    """Abstract base for all agents."""

    name: str = "BaseAgent"

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> BaseModel:
        """Execute the agent and return a Pydantic output model."""
        ...

    def call_gemini(
        self,
        system_prompt: str,
        user_prompt: str,
        depth: ReportDepth | None = None,
    ) -> str:
        """Convenience wrapper to invoke Gemini with agent name tracking."""
        return gemini_client.invoke(
            agent_name=self.name,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            depth=depth,
        )

    @staticmethod
    def parse_json_response(text: str) -> dict:
        """Extract JSON from a Gemini response (handles ```json blocks)."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            lines = cleaned.split("\n")
            # Remove first and last ``` lines
            lines = [l for l in lines if not l.strip().startswith("```")]
            cleaned = "\n".join(lines)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            log_agent("BaseAgent", f"[red]JSON parse failed, returning raw text[/red]")
            return {"raw_response": text}

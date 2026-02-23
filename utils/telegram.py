"""Telegram alert sender for momentum scanner notifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from schemas.config import settings


@dataclass
class TelegramConfig:
    bot_token: str
    chat_id: str
    timeout_seconds: int = 10

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


class TelegramAlerter:
    """Thin Telegram Bot API client for Markdown alert delivery."""

    def __init__(self, config: TelegramConfig | None = None) -> None:
        self.config = config or TelegramConfig(
            bot_token=settings.telegram_bot_token,
            chat_id=settings.telegram_chat_id,
        )

    def send_markdown(self, message: str) -> tuple[bool, dict[str, Any]]:
        """Send Markdown message to configured chat."""
        if not self.config.enabled:
            return False, {"error": "telegram_not_configured"}

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": message,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout_seconds)
            resp.raise_for_status()
            body = resp.json() if resp.content else {}
            return bool(body.get("ok", True)), body
        except Exception as exc:
            return False, {"error": str(exc)}


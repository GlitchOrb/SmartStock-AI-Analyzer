from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any

import requests


@dataclass
class TelegramAlertConfig:
    bot_token: str = os.environ.get("TELEGRAM_BOT_TOKEN", "")
    chat_id: str = os.environ.get("TELEGRAM_CHAT_ID", "")
    timeout_s: int = 10

    @property
    def enabled(self) -> bool:
        return bool(self.bot_token and self.chat_id)


class TelegramAlertSender:
    def __init__(self, config: TelegramAlertConfig | None = None) -> None:
        self.config = config or TelegramAlertConfig()

    def send_markdown(self, text: str) -> tuple[bool, dict[str, Any]]:
        if not self.config.enabled:
            return False, {"error": "telegram_not_configured"}

        url = f"https://api.telegram.org/bot{self.config.bot_token}/sendMessage"
        payload = {
            "chat_id": self.config.chat_id,
            "text": text,
            "parse_mode": "Markdown",
            "disable_web_page_preview": True,
        }
        try:
            resp = requests.post(url, json=payload, timeout=self.config.timeout_s)
            resp.raise_for_status()
            body = resp.json() if resp.content else {}
            return bool(body.get("ok", True)), body
        except Exception as exc:
            return False, {"error": str(exc)}


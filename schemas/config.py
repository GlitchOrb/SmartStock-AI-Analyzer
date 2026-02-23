"""
SmartStock AI Analyzer — App Settings (loads .env)
"""

from pathlib import Path

from pydantic_settings import BaseSettings
from pydantic import Field

from schemas.enums import ReportDepth


class Settings(BaseSettings):
    """Application configuration — loads from .env file."""

    gemini_api_key: str = Field(default="", alias="GEMINI_API_KEY")
    model_name: str = Field(default="gemini-2.0-flash", alias="MODEL_NAME")
    cache_ttl: int = Field(default=3600, alias="CACHE_TTL")
    report_depth: ReportDepth = Field(default=ReportDepth.STANDARD, alias="REPORT_DEPTH")
    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    market_data_provider: str = Field(default="alpaca", alias="MARKET_DATA_PROVIDER")
    alpaca_api_key: str = Field(default="", alias="ALPACA_API_KEY")
    alpaca_secret_key: str = Field(default="", alias="ALPACA_SECRET_KEY")
    alpaca_data_feed: str = Field(default="iex", alias="ALPACA_DATA_FEED")

    project_root: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)

    @property
    def cache_dir(self) -> Path:
        p = self.project_root / "data" / "cache"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def vectordb_dir(self) -> Path:
        p = self.project_root / "data" / "vectordb"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def reports_dir(self) -> Path:
        p = self.project_root / "data" / "reports"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def fonts_dir(self) -> Path:
        return self.project_root / "assets" / "fonts"

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }

settings = Settings()

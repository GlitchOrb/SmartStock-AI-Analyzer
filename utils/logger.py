"""
SmartStock AI Analyzer — Structured Logger
"""

import io
import sys
from rich.console import Console
from rich.logging import RichHandler
import logging

# Force UTF-8 output to avoid cp949 encoding errors on Windows
_utf8_out = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")
console = Console(file=_utf8_out, force_terminal=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console, rich_tracebacks=True, markup=True)],
)

logger = logging.getLogger("smartstock")


def log_agent(agent_name: str, message: str, level: str = "info") -> None:
    """Log an agent action with a colored prefix."""
    prefix = f"[bold cyan]⟨{agent_name}⟩[/bold cyan]"
    getattr(logger, level)(f"{prefix} {message}", extra={"markup": True})


def log_gemini_call(agent_name: str, call_number: int, total_budget: int) -> None:
    """Log a Gemini API call usage."""
    log_agent(
        agent_name,
        f"[yellow]Gemini call {call_number}/{total_budget}[/yellow]",
    )

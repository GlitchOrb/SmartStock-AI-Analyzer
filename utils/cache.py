"""
SmartStock AI Analyzer — TTL File Cache
"""

import json
import hashlib
import time
from pathlib import Path
from typing import Any

from schemas.config import settings
from utils.logger import log_agent


def _cache_path(key: str) -> Path:
    """Generate a deterministic cache file path."""
    hashed = hashlib.sha256(key.encode()).hexdigest()[:16]
    return settings.cache_dir / f"{hashed}.json"


def cache_get(key: str) -> Any | None:
    """Read from cache if valid (within TTL). Returns None if miss."""
    path = _cache_path(key)
    if not path.exists():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if time.time() - data.get("_ts", 0) > settings.cache_ttl:
            path.unlink(missing_ok=True)
            return None
        return data.get("payload")
    except (json.JSONDecodeError, KeyError):
        path.unlink(missing_ok=True)
        return None


def cache_set(key: str, payload: Any) -> None:
    """Write payload to cache with current timestamp."""
    path = _cache_path(key)
    data = {"_ts": time.time(), "payload": payload}
    path.write_text(json.dumps(data, default=str, ensure_ascii=False), encoding="utf-8")
    log_agent("Cache", f"Saved → {path.name}")


def cache_invalidate(key: str) -> None:
    """Remove a cached entry."""
    path = _cache_path(key)
    path.unlink(missing_ok=True)


def cache_clear() -> int:
    """Clear all cache files. Returns number of files removed."""
    count = 0
    for f in settings.cache_dir.glob("*.json"):
        f.unlink()
        count += 1
    return count

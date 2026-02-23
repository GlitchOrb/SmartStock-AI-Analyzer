from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any


class CacheBackend:
    def get(self, key: str) -> Any | None:  # pragma: no cover - interface
        raise NotImplementedError

    def set(self, key: str, value: Any, ttl_s: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError


class InMemoryTTLCache(CacheBackend):
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._data: dict[str, tuple[float, Any]] = {}

    def get(self, key: str) -> Any | None:
        now = time.time()
        with self._lock:
            item = self._data.get(key)
            if not item:
                return None
            expires_at, value = item
            if expires_at < now:
                self._data.pop(key, None)
                return None
            return value

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        with self._lock:
            self._data[key] = (time.time() + max(1, ttl_s), value)


class RedisCacheBackend(CacheBackend):
    def __init__(self, redis_url: str) -> None:
        import redis  # optional dependency

        self.client = redis.Redis.from_url(redis_url, decode_responses=True)

    def get(self, key: str) -> Any | None:
        raw = self.client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except Exception:
            return None

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        self.client.setex(key, max(1, ttl_s), json.dumps(value, ensure_ascii=False))


@dataclass
class CacheLayer:
    backend: CacheBackend

    @classmethod
    def from_env(cls) -> "CacheLayer":
        redis_url = os.environ.get("REDIS_URL", "").strip()
        if redis_url:
            try:
                return cls(backend=RedisCacheBackend(redis_url))
            except Exception:
                pass
        return cls(backend=InMemoryTTLCache())

    def get(self, key: str) -> Any | None:
        return self.backend.get(key)

    def set(self, key: str, value: Any, ttl_s: int) -> None:
        self.backend.set(key, value, ttl_s)

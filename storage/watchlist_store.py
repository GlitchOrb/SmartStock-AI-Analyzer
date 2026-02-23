from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path


class WatchlistStore:
    def __init__(self, db_path: str | Path = "data/plans.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS watchlist (
                    ticker TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def list_all(self) -> list[str]:
        with self._connect() as conn:
            rows = conn.execute("SELECT ticker FROM watchlist ORDER BY ticker").fetchall()
            return [str(r["ticker"]).upper() for r in rows]

    def add(self, ticker: str) -> None:
        t = ticker.upper().strip()
        if not t:
            return
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO watchlist (ticker, created_at)
                VALUES (?, ?)
                ON CONFLICT(ticker) DO NOTHING
                """,
                (t, datetime.now(timezone.utc).isoformat()),
            )
            conn.commit()

    def remove(self, ticker: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM watchlist WHERE ticker=?", (ticker.upper().strip(),))
            conn.commit()

    def replace_all(self, tickers: list[str]) -> list[str]:
        normalized = sorted({t.upper().strip() for t in tickers if t and t.strip()})
        with self._connect() as conn:
            conn.execute("DELETE FROM watchlist")
            for t in normalized:
                conn.execute(
                    "INSERT INTO watchlist (ticker, created_at) VALUES (?, ?)",
                    (t, datetime.now(timezone.utc).isoformat()),
                )
            conn.commit()
        return normalized

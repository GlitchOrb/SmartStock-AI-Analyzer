from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path


@dataclass
class TradeLevels:
    ticker: str
    entry: float | None = None
    stop: float | None = None
    target1: float | None = None
    target2: float | None = None
    updated_at: str | None = None


class PlanStore:
    """SQLite-backed user plan storage."""

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
                CREATE TABLE IF NOT EXISTS trade_plans (
                    ticker TEXT PRIMARY KEY,
                    entry REAL NULL,
                    stop REAL NULL,
                    target1 REAL NULL,
                    target2 REAL NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            conn.commit()

    def upsert(self, levels: TradeLevels) -> TradeLevels:
        ticker = levels.ticker.upper().strip()
        ts = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO trade_plans (ticker, entry, stop, target1, target2, updated_at)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker) DO UPDATE SET
                    entry=excluded.entry,
                    stop=excluded.stop,
                    target1=excluded.target1,
                    target2=excluded.target2,
                    updated_at=excluded.updated_at
                """,
                (ticker, levels.entry, levels.stop, levels.target1, levels.target2, ts),
            )
            conn.commit()
        return TradeLevels(
            ticker=ticker,
            entry=levels.entry,
            stop=levels.stop,
            target1=levels.target1,
            target2=levels.target2,
            updated_at=ts,
        )

    def get(self, ticker: str) -> TradeLevels | None:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT ticker, entry, stop, target1, target2, updated_at FROM trade_plans WHERE ticker=?",
                (ticker.upper().strip(),),
            ).fetchone()
            if not row:
                return None
            return TradeLevels(
                ticker=row["ticker"],
                entry=row["entry"],
                stop=row["stop"],
                target1=row["target1"],
                target2=row["target2"],
                updated_at=row["updated_at"],
            )

    def list_all(self) -> list[TradeLevels]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT ticker, entry, stop, target1, target2, updated_at FROM trade_plans ORDER BY ticker"
            ).fetchall()
            return [
                TradeLevels(
                    ticker=row["ticker"],
                    entry=row["entry"],
                    stop=row["stop"],
                    target1=row["target1"],
                    target2=row["target2"],
                    updated_at=row["updated_at"],
                )
                for row in rows
            ]

    def delete(self, ticker: str) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM trade_plans WHERE ticker=?", (ticker.upper().strip(),))
            conn.commit()


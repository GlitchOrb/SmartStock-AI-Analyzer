from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from api.scanner_engine import RealtimeScannerEngine, ScannerConfig
from marketdata import get_provider
from realtime.aggregator import TickBarAggregator
from storage.plan_store import PlanStore, TradeLevels


class PlanInput(BaseModel):
    entry: float | None = Field(default=None)
    stop: float | None = Field(default=None)
    target1: float | None = Field(default=None)
    target2: float | None = Field(default=None)


app = FastAPI(title="SmartStock Realtime API", version="1.0.0")
provider = get_provider()
plan_store = PlanStore()
scanner = RealtimeScannerEngine(
    provider=provider,
    plan_store=plan_store,
    config=ScannerConfig(),
)
aggregator = TickBarAggregator()


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "ok": True,
        "provider": provider.name,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/scan")
def scan(send_alerts: bool = Query(default=False)) -> dict[str, Any]:
    try:
        return scanner.run_scan(send_alerts=send_alerts)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scan_failed: {exc}")


@app.get("/ticker/{ticker}/snapshot")
def ticker_snapshot(
    ticker: str,
    timeframe: str = Query(default="1m", pattern="^(1m|5m|15m|1h|1d)$"),
) -> dict[str, Any]:
    try:
        payload = scanner.get_ticker_snapshot(ticker, timeframe=timeframe)
        if "error" in payload:
            raise HTTPException(status_code=404, detail=payload["error"])
        payload["tape"] = aggregator.get_tape(ticker, n=120)
        payload["bars_1s"] = aggregator.get_bars(ticker, timeframe="1s", n=180)
        payload["bars_1m_live"] = aggregator.get_bars(ticker, timeframe="1m", n=180)
        return payload
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"snapshot_failed: {exc}")


@app.get("/plan/{ticker}")
def get_plan(ticker: str) -> dict[str, Any]:
    p = plan_store.get(ticker)
    return {"ticker": ticker.upper(), "plan": p.__dict__ if p else {}}


@app.get("/plans")
def list_plans() -> dict[str, Any]:
    return {"plans": [p.__dict__ for p in plan_store.list_all()]}


@app.post("/plan/{ticker}")
def upsert_plan(ticker: str, payload: PlanInput) -> dict[str, Any]:
    levels = TradeLevels(
        ticker=ticker.upper(),
        entry=payload.entry,
        stop=payload.stop,
        target1=payload.target1,
        target2=payload.target2,
    )
    saved = plan_store.upsert(levels)
    return {"ok": True, "plan": saved.__dict__}


@app.delete("/plan/{ticker}")
def delete_plan(ticker: str) -> dict[str, Any]:
    plan_store.delete(ticker)
    return {"ok": True, "ticker": ticker.upper()}


@app.websocket("/stream")
async def stream(ws: WebSocket, ticker: str = Query(...)) -> None:
    await ws.accept()
    ticker = ticker.upper().strip()
    try:
        try:
            async for event in provider.websocket_stream([ticker]):
                aggregator.update(event)
                breakout_alert = scanner.check_realtime_breakout_alert(
                    ticker=ticker,
                    session=event.get("session", "UNKNOWN"),
                    price=float(event.get("price", 0.0) or 0.0),
                    send=True,
                )
                await ws.send_json(
                    {
                        "ticker": ticker,
                        "event": event,
                        "breakout_alert": breakout_alert,
                        "tape": aggregator.get_tape(ticker, n=80),
                        "bars_1s": aggregator.get_bars(ticker, timeframe="1s", n=120),
                        "bars_1m": aggregator.get_bars(ticker, timeframe="1m", n=120),
                    }
                )
        except Exception:
            while True:
                snap = provider.get_snapshot([ticker]).get(ticker)
                if snap:
                    event = {
                        "event": "tick",
                        "ticker": ticker,
                        "price": snap.price,
                        "volume": snap.volume,
                        "timestamp": snap.timestamp.isoformat(),
                        "session": snap.session,
                        "provider": provider.name,
                    }
                    aggregator.update(event)
                    breakout_alert = scanner.check_realtime_breakout_alert(
                        ticker=ticker,
                        session=event.get("session", "UNKNOWN"),
                        price=float(event.get("price", 0.0) or 0.0),
                        send=True,
                    )
                    await ws.send_json(
                        {
                            "ticker": ticker,
                            "event": event,
                            "breakout_alert": breakout_alert,
                            "tape": aggregator.get_tape(ticker, n=80),
                            "bars_1s": aggregator.get_bars(ticker, timeframe="1s", n=120),
                            "bars_1m": aggregator.get_bars(ticker, timeframe="1m", n=120),
                        }
                    )
                await asyncio.sleep(1.0)
    except WebSocketDisconnect:
        return
    except Exception:
        await ws.close(code=1011)

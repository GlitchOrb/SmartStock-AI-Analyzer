from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
import time

from fastapi import Body, FastAPI, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel, Field

from api.scanner_engine import RealtimeScannerEngine, ScannerConfig
from marketdata import get_provider
from realtime.aggregator import TickBarAggregator
from storage.plan_store import PlanStore, TradeLevels
from storage.watchlist_store import WatchlistStore


class PlanInput(BaseModel):
    entry: float | None = Field(default=None)
    stop: float | None = Field(default=None)
    target1: float | None = Field(default=None)
    target2: float | None = Field(default=None)


app = FastAPI(title="SmartStock Realtime API", version="1.0.0")
provider = get_provider()
plan_store = PlanStore()
watchlist_store = WatchlistStore()
scanner = RealtimeScannerEngine(
    provider=provider,
    plan_store=plan_store,
    config=ScannerConfig(),
)
aggregator = TickBarAggregator()
request_metrics: dict[str, dict[str, float]] = {}


@app.middleware("http")
async def metrics_middleware(request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    key = f"{request.method} {request.url.path}"
    bucket = request_metrics.setdefault(key, {"count": 0.0, "total_ms": 0.0, "last_ms": 0.0})
    bucket["count"] += 1.0
    bucket["total_ms"] += elapsed_ms
    bucket["last_ms"] = elapsed_ms
    return response


@app.on_event("startup")
async def startup_info() -> None:
    configured = bool(getattr(provider, "configured", True))
    print(
        f"[startup] api.server file={Path(__file__).resolve()} "
        f"provider={provider.name} configured={configured}"
    )


@app.get("/health")
def health() -> dict[str, Any]:
    configured = bool(getattr(provider, "configured", True))
    return {
        "ok": True,
        "provider": provider.name,
        "provider_configured": configured,
        "watchlist_count": len(watchlist_store.list_all()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/universe")
def universe(
    q: str = Query(default="", max_length=20),
    limit: int = Query(default=5000, ge=1, le=20000),
    include_etf: bool = Query(default=False),
) -> dict[str, Any]:
    tickers = provider.get_universe(include_etf=include_etf)
    query = q.strip().upper()
    if query:
        tickers = [t for t in tickers if query in t]
    return {
        "count": len(tickers),
        "tickers": tickers[:limit],
    }


@app.get("/scan")
def scan(
    send_alerts: bool = Query(default=False),
    full_universe: bool = Query(default=False),
    fast_top_k: int = Query(default=120, ge=20, le=2000),
    deep_top_n: int = Query(default=40, ge=10, le=500),
) -> dict[str, Any]:
    try:
        return scanner.run_scan(
            send_alerts=send_alerts,
            full_universe=full_universe,
            fast_top_k=fast_top_k,
            deep_top_n=deep_top_n,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"scan_failed: {exc}")


@app.get("/metrics")
def metrics() -> dict[str, Any]:
    http_metrics = {}
    for key, value in request_metrics.items():
        count = value.get("count", 0.0) or 0.0
        avg_ms = (value.get("total_ms", 0.0) / count) if count else 0.0
        http_metrics[key] = {
            "count": int(count),
            "avg_ms": round(avg_ms, 2),
            "last_ms": round(value.get("last_ms", 0.0), 2),
        }
    return {
        **scanner.get_metrics(),
        "http": http_metrics,
    }


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


@app.get("/watchlist")
def get_watchlist() -> dict[str, Any]:
    tickers = watchlist_store.list_all()
    return {"count": len(tickers), "tickers": tickers}


@app.put("/watchlist")
def replace_watchlist(payload: list[str] = Body(default=[])) -> dict[str, Any]:
    tickers = watchlist_store.replace_all(payload)
    return {"ok": True, "count": len(tickers), "tickers": tickers}


@app.post("/watchlist/{ticker}")
def add_watchlist_ticker(ticker: str) -> dict[str, Any]:
    watchlist_store.add(ticker)
    tickers = watchlist_store.list_all()
    return {"ok": True, "count": len(tickers), "tickers": tickers}


@app.delete("/watchlist/{ticker}")
def remove_watchlist_ticker(ticker: str) -> dict[str, Any]:
    watchlist_store.remove(ticker)
    tickers = watchlist_store.list_all()
    return {"ok": True, "count": len(tickers), "tickers": tickers}


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
    watchlist_store.add(ticker)
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

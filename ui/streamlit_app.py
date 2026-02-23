from __future__ import annotations

from collections import deque
import json
import threading
import time
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import websocket


st.set_page_config(page_title="SmartStock Realtime Scanner", page_icon="📈", layout="wide")


_WS_STATE: dict[str, Any] = {
    "key": "",
    "alive": False,
    "latest": {},
    "tape": deque(maxlen=300),
    "lock": threading.Lock(),
    "thread": None,
}


def api_get(base_url: str, path: str, params: dict[str, Any] | None = None) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, params=params, timeout=60)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url: str, path: str, payload: dict[str, Any]) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _to_ws_url(base_url: str, ticker: str) -> str:
    parsed = urlparse(base_url)
    scheme = "wss" if parsed.scheme == "https" else "ws"
    return f"{scheme}://{parsed.netloc}/stream?ticker={ticker}"


def _start_ws_stream(base_url: str, ticker: str) -> None:
    ws_url = _to_ws_url(base_url, ticker)
    state_key = f"{base_url}|{ticker}"
    if _WS_STATE["alive"] and _WS_STATE["key"] == state_key:
        return

    _WS_STATE["alive"] = False
    old_thread = _WS_STATE.get("thread")
    if old_thread and isinstance(old_thread, threading.Thread) and old_thread.is_alive():
        time.sleep(0.2)

    _WS_STATE["key"] = state_key
    _WS_STATE["latest"] = {}
    _WS_STATE["tape"] = deque(maxlen=300)
    _WS_STATE["alive"] = True

    def worker() -> None:
        while _WS_STATE["alive"] and _WS_STATE["key"] == state_key:
            ws = None
            try:
                ws = websocket.create_connection(ws_url, timeout=10)
                while _WS_STATE["alive"] and _WS_STATE["key"] == state_key:
                    raw = ws.recv()
                    if not raw:
                        break
                    msg = json.loads(raw)
                    with _WS_STATE["lock"]:
                        _WS_STATE["latest"] = msg
                        for item in msg.get("tape", [])[-6:]:
                            _WS_STATE["tape"].append(item)
            except Exception:
                time.sleep(1.5)
            finally:
                try:
                    if ws is not None:
                        ws.close()
                except Exception:
                    pass

    th = threading.Thread(target=worker, daemon=True, name=f"ws_{ticker}")
    _WS_STATE["thread"] = th
    th.start()


def _get_ws_payload() -> dict[str, Any]:
    with _WS_STATE["lock"]:
        return {
            "latest": dict(_WS_STATE.get("latest", {})),
            "tape": list(_WS_STATE.get("tape", [])),
        }


def render_top_cards(results: list[dict[str, Any]]) -> None:
    top = [r for r in results if float(r.get("score", 0)) >= 70]
    st.subheader(f"Top Momentum Picks (score >= 70): {len(top)}")
    if not top:
        st.info("No top momentum picks in current scan.")
        return
    cols = st.columns(min(4, len(top)))
    for i, row in enumerate(top[:8]):
        with cols[i % len(cols)]:
            with st.container(border=True):
                st.markdown(f"### {row['ticker']}")
                st.metric("Last", f"${row['last_price']:.2f}", f"{row['change_5m_pct']:+.2f}% (5m)")
                st.caption(f"{row['session']} | Score {row['score']:.1f}")
                st.write(f"Surge: {'✅' if row.get('surge_flag') else '—'}")
                if row.get("surge_reason"):
                    st.caption(", ".join(row["surge_reason"][:2]))


def render_scan_table(results: list[dict[str, Any]]) -> None:
    table = pd.DataFrame(
        [
            {
                "ticker": r["ticker"],
                "last_price": r["last_price"],
                "session": r["session"],
                "change_1m_pct": r["change_1m_pct"],
                "change_5m_pct": r["change_5m_pct"],
                "rel_volume": r["rel_volume"],
                "vwap_distance_pct": r["vwap_distance_pct"],
                "rsi_14": r["rsi_14"],
                "macd_status": r["macd_status"],
                "adx": r["adx"],
                "breakout_flag": r["breakout_flag"],
                "score": r["score"],
                "surge_flag": r["surge_flag"],
            }
            for r in results
        ]
    )
    st.subheader("Scanner Output")
    st.dataframe(table, use_container_width=True, hide_index=True)


def build_candles(snapshot_payload: dict[str, Any], levels: dict[str, Any]) -> go.Figure:
    bars = snapshot_payload.get("bars", [])
    df = pd.DataFrame(bars)
    if df.empty:
        return go.Figure()

    fig = go.Figure()
    fig.add_trace(
        go.Candlestick(
            x=df["timestamp"],
            open=df["open"],
            high=df["high"],
            low=df["low"],
            close=df["close"],
            name="Price",
        )
    )
    for key, color in [("vwap", "#0ea5e9"), ("ema20", "#22c55e"), ("ema50", "#f59e0b"), ("ema200", "#ef4444")]:
        if key in df.columns:
            fig.add_trace(go.Scatter(x=df["timestamp"], y=df[key], mode="lines", name=key.upper(), line={"width": 1.4, "color": color}))

    last_x = df["timestamp"].iloc[-1]
    for name, color in [("entry", "#22c55e"), ("stop", "#ef4444"), ("target1", "#f59e0b"), ("target2", "#a855f7")]:
        v = levels.get(name)
        if v is None:
            continue
        fig.add_hline(y=v, line_color=color, line_width=1.4, line_dash="dash", annotation_text=name.upper(), annotation_position="top right")
        fig.add_annotation(x=last_x, y=v, text=f"{name}:{v:.2f}", showarrow=False, xanchor="left")

    fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 25, "b": 10}, xaxis_rangeslider_visible=False)
    return fig


def render_plan_editor(base_url: str, ticker: str, current_price: float, plan: dict[str, Any]) -> None:
    st.markdown("#### Live Plan Levels")
    c1, c2, c3, c4 = st.columns(4)
    entry = c1.number_input("Entry", value=float(plan.get("entry") or 0.0), key=f"{ticker}_entry")
    stop = c2.number_input("Stop", value=float(plan.get("stop") or 0.0), key=f"{ticker}_stop")
    t1 = c3.number_input("Target1", value=float(plan.get("target1") or 0.0), key=f"{ticker}_t1")
    t2 = c4.number_input("Target2", value=float(plan.get("target2") or 0.0), key=f"{ticker}_t2")

    normalized = {
        "entry": entry if entry > 0 else None,
        "stop": stop if stop > 0 else None,
        "target1": t1 if t1 > 0 else None,
        "target2": t2 if t2 > 0 else None,
    }
    if st.button("Save Plan", key=f"{ticker}_save_plan"):
        api_post(base_url, f"/plan/{ticker}", normalized)
        st.success("Plan saved.")

    if normalized["entry"] and normalized["stop"] and normalized["target1"]:
        risk = abs(normalized["entry"] - normalized["stop"])
        reward = abs(normalized["target1"] - normalized["entry"])
        rr = (reward / risk) if risk > 0 else 0.0
        dist = ((current_price - normalized["entry"]) / normalized["entry"] * 100) if normalized["entry"] else 0.0
        st.info(f"Trade Plan Summary | R:R={rr:.2f} | Distance to Entry={dist:+.2f}% (${current_price - normalized['entry']:+.2f})")


def main() -> None:
    st.title("SmartStock Realtime Scanner")
    st.caption("Engineering tool for market monitoring only. Not financial advice.")

    with st.sidebar:
        base_url = st.text_input("Backend API URL", value="http://localhost:8000")
        send_alerts = st.checkbox("Send Telegram alerts on scan", value=False)
        auto_refresh = st.checkbox("Auto-refresh Detail", value=True)
        timeframe = st.selectbox("Detail Timeframe", ["1m", "5m", "15m", "1h", "1d"], index=0)
        run_scan = st.button("Run Scanner", type="primary", use_container_width=True)

    if run_scan or "scan_payload" not in st.session_state:
        try:
            st.session_state["scan_payload"] = api_get(base_url, "/scan", params={"send_alerts": send_alerts})
        except Exception as exc:
            st.error(f"Scan failed: {exc}")
            return

    payload = st.session_state.get("scan_payload", {})
    results = payload.get("results", [])
    if not results:
        st.warning("No scan results.")
        return

    stats = payload.get("stats", {})
    st.success(
        f"Provider={stats.get('provider')} | "
        f"Prefiltered={stats.get('liquidity_pass_count', 0)} | "
        f"Results={stats.get('result_count', len(results))}"
    )
    for w in payload.get("warnings", [])[:8]:
        st.warning(w)

    render_top_cards(results)
    st.divider()
    render_scan_table(results)

    ticker = st.selectbox("Detail Ticker", [r["ticker"] for r in results], index=0)
    if auto_refresh:
        st_autorefresh(interval=2000, key=f"refresh_{ticker}_{timeframe}")

    _start_ws_stream(base_url, ticker)
    ws_payload = _get_ws_payload()

    try:
        detail = api_get(base_url, f"/ticker/{ticker}/snapshot", params={"timeframe": timeframe})
    except Exception as exc:
        st.error(f"Detail fetch failed: {exc}")
        return

    row = next((r for r in results if r["ticker"] == ticker), {})
    st.subheader(f"{ticker} Detail | Session: {detail.get('session', row.get('session', 'UNKNOWN'))}")
    c1, c2, c3, c4 = st.columns(4)
    live_price = detail.get("price", row.get("last_price", 0.0))
    if ws_payload.get("latest", {}).get("event", {}).get("price"):
        live_price = ws_payload["latest"]["event"]["price"]
    c1.metric("Price", f"${float(live_price):.2f}")
    c2.metric("Score", f"{row.get('score', 0.0):.1f}")
    c3.metric("1m%", f"{row.get('change_1m_pct', 0.0):+.2f}%")
    c4.metric("5m%", f"{row.get('change_5m_pct', 0.0):+.2f}%")

    indicators = detail.get("indicators", {})
    st.write(
        f"RSI: {indicators.get('rsi_14', 0)} | "
        f"MACD Hist: {indicators.get('macd_hist', 0)} | "
        f"VWAP: {indicators.get('vwap', 0)} | "
        f"EMA20/50/200: {indicators.get('ema20', 0)}/{indicators.get('ema50', 0)}/{indicators.get('ema200', 0)}"
    )

    levels = detail.get("plan", {})
    fig = build_candles(detail, levels)
    st.plotly_chart(fig, use_container_width=True)

    surge_reasons = row.get("surge_reason", [])
    st.markdown("#### Surge Indicators")
    st.write(
        f"Surge Flag: {'✅' if row.get('surge_flag') else '—'} | "
        f"Reasons: {', '.join(surge_reasons) if surge_reasons else 'None'}"
    )
    st.write(
        f"Breakout: {row.get('breakout_flag')} | "
        f"RelVol: {row.get('rel_volume')} | "
        f"VWAP Distance: {row.get('vwap_distance_pct')}%"
    )

    render_plan_editor(base_url, ticker, float(live_price), levels)

    st.markdown("#### Mini Tape (latest ticks)")
    tape_source = ws_payload.get("tape") or detail.get("tape", [])
    tape_df = pd.DataFrame(tape_source)
    if not tape_df.empty:
        st.dataframe(tape_df.tail(40), use_container_width=True, hide_index=True)
    else:
        st.caption("No live tape yet. Use backend /stream websocket for live events.")

    with st.expander("Raw Detail JSON"):
        st.json(detail)


if __name__ == "__main__":
    main()

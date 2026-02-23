from __future__ import annotations

from collections import deque
import json
import os
import threading
import time
from typing import Any
from urllib.parse import urlparse

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import websocket

try:
    from streamlit_autorefresh import st_autorefresh
    HAS_ST_AUTOREFRESH = True
except ImportError:
    HAS_ST_AUTOREFRESH = False

    def st_autorefresh(*_: Any, **__: Any) -> int:
        return 0


st.set_page_config(page_title="SmartStock Realtime Scanner", page_icon="S", layout="wide")

DEFAULT_API_URL = os.getenv("SMARTSTOCK_API_URL", "http://localhost:8000")
SCAN_INTERVAL_MAP = {
    "Off": 0,
    "1 min": 60,
    "3 min": 180,
    "5 min": 300,
}

_WS_STATE: dict[str, Any] = {
    "key": "",
    "alive": False,
    "latest": {},
    "tape": deque(maxlen=300),
    "lock": threading.Lock(),
    "thread": None,
}


def render_app_chrome() -> None:
    st.markdown(
        """
        <style>
        [data-testid="stDeployButton"] { display: none !important; }
        [data-testid="stToolbar"] [aria-label="Deploy"] { display: none !important; }
        div[data-testid="stDecoration"] { display: none !important; }
        #sylee2412-watermark {
            position: fixed;
            top: 8px;
            left: 14px;
            z-index: 99999;
            font-size: 11px;
            color: #8a8f98;
            letter-spacing: 0.04em;
            user-select: none;
            pointer-events: none;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        </style>
        <div id="sylee2412-watermark">sylee2412</div>
        """,
        unsafe_allow_html=True,
    )


def api_get(base_url: str, path: str, params: dict[str, Any] | None = None, timeout: int = 60) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.get(url, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json()


def api_post(base_url: str, path: str, payload: dict[str, Any]) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.post(url, json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def api_put(base_url: str, path: str, payload: Any) -> Any:
    url = f"{base_url.rstrip('/')}{path}"
    resp = requests.put(url, json=payload, timeout=30)
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


def render_backend_unavailable(base_url: str, exc: Exception) -> None:
    st.error(f"Backend connection failed: {exc}")
    st.info(f"Expected API endpoint: {base_url}")
    st.markdown("Run backend in a separate terminal:")
    st.code("uvicorn api.server:app --host 0.0.0.0 --port 8000", language="bash")


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
                st.caption(f"{row['session']} | Score {row['score']:.1f} | {row.get('strategy_label', '-')}")
                positives = [k for k, v in row.get("signals", {}).items() if v]
                st.caption("Signals: " + (", ".join(positives[:3]) if positives else "None"))
                if row.get("surge_reason"):
                    st.caption("Surge: " + ", ".join(row["surge_reason"][:2]))


def render_scan_table(results: list[dict[str, Any]]) -> tuple[pd.DataFrame, str | None]:
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
                "strategy_label": r.get("strategy_label", "-"),
            }
            for r in results
        ]
    )
    if table.empty:
        return table, None
    table = table.sort_values(by=["score", "change_5m_pct"], ascending=[False, False]).reset_index(drop=True)
    table.insert(0, "rank", table.index + 1)
    st.subheader("Scanner Output")
    selector = table.copy()
    selector.insert(0, "open", False)
    edited = st.data_editor(
        selector,
        width="stretch",
        hide_index=True,
        key="scan_selector_table",
        disabled=[c for c in selector.columns if c != "open"],
    )
    selected_rows = edited[edited["open"] == True]
    selected_ticker = str(selected_rows.iloc[0]["ticker"]) if not selected_rows.empty else None
    return table, selected_ticker


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
    for key, color in [
        ("vwap", "#0ea5e9"),
        ("ema20", "#22c55e"),
        ("ema50", "#f59e0b"),
        ("ema200", "#ef4444"),
        ("bb_upper", "#a78bfa"),
        ("bb_lower", "#a78bfa"),
    ]:
        if key in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=df["timestamp"],
                    y=df[key],
                    mode="lines",
                    name=key.upper(),
                    line={"width": 1.2, "color": color},
                )
            )

    last_x = df["timestamp"].iloc[-1]
    for name, color in [("entry", "#22c55e"), ("stop", "#ef4444"), ("target1", "#f59e0b"), ("target2", "#a855f7")]:
        v = levels.get(name)
        if v is None:
            continue
        fig.add_hline(y=v, line_color=color, line_width=1.4, line_dash="dash", annotation_text=name.upper(), annotation_position="top right")
        fig.add_annotation(x=last_x, y=v, text=f"{name}:{v:.2f}", showarrow=False, xanchor="left")

    fig.update_layout(height=560, margin={"l": 10, "r": 10, "t": 25, "b": 10}, xaxis_rangeslider_visible=False)
    return fig


def render_plan_editor(base_url: str, ticker: str, current_price: float, initial_plan: dict[str, Any], suggested: dict[str, Any]) -> dict[str, Any]:
    st.markdown("#### Live Plan Levels")

    seed = {
        "entry": initial_plan.get("entry"),
        "stop": initial_plan.get("stop"),
        "target1": initial_plan.get("target1"),
        "target2": initial_plan.get("target2"),
    }
    if not any(seed.values()) and suggested:
        seed = {
            "entry": suggested.get("entry_price"),
            "stop": suggested.get("stop_loss"),
            "target1": suggested.get("target_price_1"),
            "target2": suggested.get("target_price_2"),
        }

    c1, c2, c3, c4 = st.columns(4)
    entry = c1.number_input("Entry", value=float(seed.get("entry") or 0.0), key=f"{ticker}_entry")
    stop = c2.number_input("Stop", value=float(seed.get("stop") or 0.0), key=f"{ticker}_stop")
    t1 = c3.number_input("Target1", value=float(seed.get("target1") or 0.0), key=f"{ticker}_t1")
    t2 = c4.number_input("Target2", value=float(seed.get("target2") or 0.0), key=f"{ticker}_t2")

    normalized = {
        "entry": entry if entry > 0 else None,
        "stop": stop if stop > 0 else None,
        "target1": t1 if t1 > 0 else None,
        "target2": t2 if t2 > 0 else None,
    }

    save_col, use_suggest_col = st.columns(2)
    if save_col.button("Save Plan", key=f"{ticker}_save_plan"):
        api_post(base_url, f"/plan/{ticker}", normalized)
        st.success("Plan saved.")
    if suggested and use_suggest_col.button("Apply Suggested Plan", key=f"{ticker}_apply_suggest"):
        suggested_payload = {
            "entry": suggested.get("entry_price"),
            "stop": suggested.get("stop_loss"),
            "target1": suggested.get("target_price_1"),
            "target2": suggested.get("target_price_2"),
        }
        api_post(base_url, f"/plan/{ticker}", suggested_payload)
        st.success("Suggested plan saved.")

    if normalized["entry"] and normalized["stop"] and normalized["target1"]:
        risk = abs(normalized["entry"] - normalized["stop"])
        reward = abs(normalized["target1"] - normalized["entry"])
        rr = (reward / risk) if risk > 0 else 0.0
        dist = ((current_price - normalized["entry"]) / normalized["entry"] * 100) if normalized["entry"] else 0.0
        st.info(f"Trade Plan Summary | R:R={rr:.2f} | Distance to Entry={dist:+.2f}% (${current_price - normalized['entry']:+.2f})")

    return normalized


def _fetch_universe_candidates(base_url: str, query: str) -> list[str]:
    if len(query.strip()) < 1:
        return []
    try:
        payload = api_get(base_url, "/universe", params={"q": query.strip().upper(), "limit": 50})
        return payload.get("tickers", [])
    except Exception:
        return []


def _fetch_watchlist(base_url: str) -> list[str]:
    try:
        payload = api_get(base_url, "/watchlist")
        tickers = payload.get("tickers", [])
        return [str(t).upper() for t in tickers]
    except Exception:
        return []


def _save_watchlist(base_url: str, tickers: list[str]) -> list[str]:
    try:
        payload = api_put(base_url, "/watchlist", tickers)
        return [str(t).upper() for t in payload.get("tickers", [])]
    except Exception:
        return tickers


def _maybe_refresh_scan(
    base_url: str,
    send_alerts: bool,
    interval_sec: int,
    full_universe_scan: bool,
    fast_top_k: int,
    deep_top_n: int,
    force: bool = False,
) -> tuple[dict[str, Any] | None, Exception | None]:
    now = time.time()
    last_run = float(st.session_state.get("scan_last_run_ts", 0.0) or 0.0)
    has_payload = "scan_payload" in st.session_state
    due = interval_sec > 0 and (now - last_run) >= interval_sec

    if force or (not has_payload) or due:
        try:
            payload = api_get(
                base_url,
                "/scan",
                params={
                    "send_alerts": send_alerts,
                    "full_universe": full_universe_scan,
                    "fast_top_k": fast_top_k,
                    "deep_top_n": deep_top_n,
                },
                timeout=300,
            )
            st.session_state["scan_payload"] = payload
            st.session_state["scan_last_run_ts"] = now
            st.session_state["scan_last_run_label"] = time.strftime("%Y-%m-%d %H:%M:%S")
            return payload, None
        except Exception as exc:
            return None, exc

    return st.session_state.get("scan_payload"), None


def _scan_should_refresh(interval_sec: int, force: bool) -> bool:
    if force:
        return True
    has_payload = "scan_payload" in st.session_state
    if not has_payload:
        return True
    if interval_sec <= 0:
        return False
    last_run = float(st.session_state.get("scan_last_run_ts", 0.0) or 0.0)
    return (time.time() - last_run) >= interval_sec


def main() -> None:
    render_app_chrome()
    st.title("SmartStock Realtime Scanner")
    st.caption("Engineering monitor tool. Not financial advice.")
    base_url = DEFAULT_API_URL

    with st.sidebar:
        st.markdown(f"API: `{base_url}`")
        send_alerts = st.checkbox("Send Telegram alerts on scan", value=False)
        full_universe_scan = st.checkbox("Full NASDAQ scan (slow)", value=False)
        fast_top_k = st.number_input("FAST top K", min_value=50, max_value=2000, value=120, step=10)
        deep_top_n = st.number_input("DEEP top N", min_value=20, max_value=500, value=40, step=10)
        auto_scan_label = st.selectbox("Scanner auto refresh", list(SCAN_INTERVAL_MAP.keys()), index=1)
        detail_timeframe = st.selectbox("Detail timeframe", ["1m", "5m", "15m", "1h", "1d"], index=0)
        auto_refresh_detail = st.checkbox("Auto-refresh detail", value=True)
        run_scan = st.button("Run Scanner Now", type="primary", width="stretch")

    scan_interval_sec = SCAN_INTERVAL_MAP[auto_scan_label]
    if scan_interval_sec > 0 and HAS_ST_AUTOREFRESH:
        st_autorefresh(interval=scan_interval_sec * 1000, key="scan_auto_refresh")
    elif scan_interval_sec > 0 and not HAS_ST_AUTOREFRESH:
        st.caption("Install streamlit-autorefresh for timed scanner refresh.")

    should_refresh = _scan_should_refresh(scan_interval_sec, run_scan)
    if should_refresh:
        scan_mode = "FULL NASDAQ" if full_universe_scan else "PREFILTER"
        with st.spinner(f"Scanner running ({scan_mode})... collecting market data."):
            t0 = time.time()
            payload, scan_exc = _maybe_refresh_scan(
                base_url,
                send_alerts,
                scan_interval_sec,
                full_universe_scan=full_universe_scan,
                fast_top_k=int(fast_top_k),
                deep_top_n=int(deep_top_n),
                force=True,
            )
            st.session_state["scan_last_elapsed_sec"] = round(time.time() - t0, 1)
    else:
        payload, scan_exc = _maybe_refresh_scan(
            base_url,
            send_alerts,
            scan_interval_sec,
            full_universe_scan=full_universe_scan,
            fast_top_k=int(fast_top_k),
            deep_top_n=int(deep_top_n),
            force=False,
        )

    if scan_exc:
        render_backend_unavailable(base_url, scan_exc)
        return

    if not payload:
        st.warning("No scan payload.")
        return

    results = payload.get("results", [])
    stats = payload.get("stats", {})
    st.success(
        f"Provider={stats.get('provider')} | "
        f"DelayedMode={stats.get('delayed_mode')} | "
        f"Prefiltered={stats.get('liquidity_pass_count', 0)} | "
        f"Method={stats.get('prefilter_method', '-')} | "
        f"Mode={'FULL' if stats.get('full_universe_mode') else 'PREFILTER'} | "
        f"StageA={stats.get('stage_a_count', '-')} ({stats.get('stage_a_seconds', '-') }s) | "
        f"StageB={stats.get('stage_b_count', '-')} ({stats.get('stage_b_seconds', '-') }s) | "
        f"Results={stats.get('result_count', len(results))} | "
        f"LastScan={st.session_state.get('scan_last_run_label', '-') } | "
        f"Elapsed={st.session_state.get('scan_last_elapsed_sec', '-') }s"
    )
    for w in payload.get("warnings", [])[:8]:
        st.warning(w)
    if should_refresh and not scan_exc:
        st.info("Scan completed. Table is updated below.")

    if not results:
        st.warning("No scan results. You can still search any NASDAQ ticker and open detail below.")
    else:
        render_top_cards(results)
        st.divider()

    search_col, score_col, top_col = st.columns([2, 1, 1])
    search_text = search_col.text_input("Ticker search", value="", placeholder="AAPL / NVDA / TSLA ...")
    min_score = score_col.slider("Min score", min_value=0, max_value=100, value=0, step=5)
    top_only = top_col.checkbox("Top only (>=70)", value=False)

    filtered: list[dict[str, Any]] = []
    for row in results:
        ticker = str(row.get("ticker", ""))
        score = float(row.get("score", 0.0) or 0.0)
        if search_text and search_text.upper() not in ticker:
            continue
        if score < min_score:
            continue
        if top_only and score < 70:
            continue
        filtered.append(row)

    if results and not filtered:
        st.info("No rows matched filters. Showing full table.")
        filtered = results

    table, clicked_ticker = render_scan_table(filtered) if filtered else (pd.DataFrame(), None)

    universe_matches = _fetch_universe_candidates(base_url, search_text)
    server_watchlist = _fetch_watchlist(base_url)
    detail_candidates = []
    if not table.empty:
        detail_candidates.extend(table["ticker"].tolist())
    detail_candidates.extend(server_watchlist)
    detail_candidates.extend(universe_matches)
    if not detail_candidates:
        detail_candidates.extend([r["ticker"] for r in results])
    detail_candidates = list(dict.fromkeys([str(t).upper() for t in detail_candidates if t]))

    with st.sidebar:
        st.markdown("### Watchlist")
        add_symbol = st.text_input("Add symbol", value="", placeholder="AAPL")
        if st.button("Add to Watchlist", width="stretch") and add_symbol.strip():
            updated = list(dict.fromkeys(server_watchlist + [add_symbol.strip().upper()]))
            _save_watchlist(base_url, updated)
            st.rerun()
        current_watchlist = st.multiselect(
            "Tracked symbols",
            options=detail_candidates if detail_candidates else server_watchlist,
            default=[s for s in server_watchlist if s in (detail_candidates if detail_candidates else server_watchlist)],
            key="watchlist_symbols",
        )
        if st.button("Save Watchlist", width="stretch"):
            _save_watchlist(base_url, current_watchlist)
            st.success("Watchlist updated.")

    st.markdown("#### Detail")
    if not detail_candidates:
        st.info("Type a ticker in search to load details.")
        return
    default_index = 0
    if clicked_ticker and clicked_ticker in detail_candidates:
        default_index = detail_candidates.index(clicked_ticker)
    ticker = st.selectbox("Select ticker", detail_candidates, index=default_index)

    if auto_refresh_detail and HAS_ST_AUTOREFRESH:
        st_autorefresh(interval=2000, key=f"detail_refresh_{ticker}_{detail_timeframe}")

    _start_ws_stream(base_url, ticker)
    ws_payload = _get_ws_payload()

    try:
        detail = api_get(base_url, f"/ticker/{ticker}/snapshot", params={"timeframe": detail_timeframe})
    except Exception as exc:
        render_backend_unavailable(base_url, exc)
        return

    if "error" in detail:
        st.error(f"Detail error: {detail['error']}")
        return

    selected_row = next((r for r in results if r["ticker"] == ticker), {})
    live_price = float(detail.get("price", selected_row.get("last_price", 0.0)) or 0.0)
    ws_event = ws_payload.get("latest", {}).get("event", {})
    if ws_event.get("price") is not None:
        live_price = float(ws_event["price"])

    session_label = detail.get("session", selected_row.get("session", "UNKNOWN"))
    st.subheader(f"{ticker} Detail | Session: {session_label}")
    if session_label == "CLOSED":
        st.caption("Market is outside PRE/REG/AFTER. Showing latest available trade/snapshot price.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Price", f"${live_price:.2f}")
    c2.metric("Composite Score", f"{float(detail.get('composite_score', selected_row.get('score', 0.0))):.1f}")
    c3.metric("1m%", f"{float(selected_row.get('change_1m_pct', 0.0)):+.2f}%")
    c4.metric("5m%", f"{float(selected_row.get('change_5m_pct', 0.0)):+.2f}%")

    indicators = detail.get("indicators", {})
    st.write(
        f"RSI(14): {indicators.get('rsi_14', 0)} | "
        f"MACD Hist: {indicators.get('macd_hist', 0)} | "
        f"VWAP: {indicators.get('vwap', 0)} | "
        f"EMA20/50/200: {indicators.get('ema20', 0)}/{indicators.get('ema50', 0)}/{indicators.get('ema200', 0)}"
    )

    strategy_label = detail.get("strategy_label", selected_row.get("strategy_label", "-"))
    rationale = detail.get("rationale", selected_row.get("rationale", ""))
    st.write(f"Strategy: {strategy_label}")
    if rationale:
        st.caption(rationale)
    score_breakdown = detail.get("score_breakdown", selected_row.get("score_breakdown", {}))
    if score_breakdown:
        st.markdown("#### Score Breakdown")
        sb_df = pd.DataFrame(
            [{"component": k, "points": v} for k, v in score_breakdown.items()]
        ).sort_values("points", ascending=False)
        st.dataframe(sb_df, width="stretch", hide_index=True)
    surge_reasons = detail.get("surge_reason", selected_row.get("surge_reason", []))
    if surge_reasons:
        st.markdown("#### Surge Reasons")
        for reason in surge_reasons:
            st.caption(f"- {reason}")

    saved_plan = detail.get("plan", {})
    suggested = detail.get("recommended_trade_plan", {})
    if suggested:
        st.markdown("#### Suggested Plan")
        st.write(
            f"Entry {suggested.get('entry_price', '-')} | "
            f"Stop {suggested.get('stop_loss', '-')} | "
            f"T1 {suggested.get('target_price_1', '-')} | "
            f"T2 {suggested.get('target_price_2', '-')} | "
            f"R:R {suggested.get('risk_reward_ratio', '-')}"
        )

    active_levels = dict(saved_plan)
    if not any(active_levels.get(k) for k in ("entry", "stop", "target1", "target2")) and suggested:
        active_levels = {
            "entry": suggested.get("entry_price"),
            "stop": suggested.get("stop_loss"),
            "target1": suggested.get("target_price_1"),
            "target2": suggested.get("target_price_2"),
        }

    fig = build_candles(detail, active_levels)
    st.plotly_chart(fig, width="stretch")

    st.markdown("#### Signal Breakdown")
    signals = detail.get("signals", selected_row.get("signals", {}))
    if signals:
        signal_df = pd.DataFrame(
            [{"signal": k, "value": v} for k, v in signals.items()]
        )
        st.dataframe(signal_df, width="stretch", hide_index=True)

    render_plan_editor(base_url, ticker, live_price, saved_plan, suggested)

    st.markdown("#### Mini Tape (latest ticks)")
    tape_source = ws_payload.get("tape") or detail.get("tape", [])
    tape_df = pd.DataFrame(tape_source)
    if not tape_df.empty:
        st.dataframe(tape_df.tail(40), width="stretch", hide_index=True)
    else:
        st.caption("No live tape yet.")

    with st.expander("Raw Detail JSON"):
        st.json(detail)


if __name__ == "__main__":
    main()


from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from alerts.telegram import TelegramAlertSender
from agents.risk_agent import RiskAgent
from agents.catalyst_agent import run_catalyst_agent_fast
from data.universe import UniversePreFilterConfig, build_prefiltered_universe
from marketdata.provider_base import MarketDataProvider
from schemas.enums import Signal
from storage.plan_store import PlanStore


US_EASTERN = ZoneInfo("America/New_York")


def _rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    v = rsi.iloc[-1] if not rsi.empty else np.nan
    return float(v) if pd.notna(v) else 50.0


def _macd(series: pd.Series) -> tuple[float, float, float]:
    ema12 = series.ewm(span=12, adjust=False).mean()
    ema26 = series.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return float(macd_line.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])


def _adx(df: pd.DataFrame, period: int = 14) -> float:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / period, adjust=False).mean().iloc[-1]
    return float(adx) if pd.notna(adx) else 0.0


def _ema_stack(series: pd.Series) -> tuple[float, float, float, bool]:
    ema20 = float(series.ewm(span=20, adjust=False).mean().iloc[-1])
    ema50 = float(series.ewm(span=50, adjust=False).mean().iloc[-1])
    ema200 = float(series.ewm(span=200, adjust=False).mean().iloc[-1]) if len(series) >= 200 else ema50
    return ema20, ema50, ema200, bool(ema20 > ema50 > ema200)


def _bb_squeeze_break(series: pd.Series, period: int = 20) -> tuple[bool, float]:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = ma + (2 * std)
    lower = ma - (2 * std)
    bw = (upper - lower) / ma.replace(0, np.nan)
    if len(bw.dropna()) < period + 2:
        return False, float(bw.iloc[-1]) if pd.notna(bw.iloc[-1]) else 0.0
    recent = bw.dropna().tail(period + 1)
    squeeze = recent.iloc[:-1].min() <= recent.iloc[:-1].quantile(0.15)
    expansion = recent.iloc[-1] > (recent.iloc[-2] * 1.3)
    breakout = series.iloc[-1] > upper.iloc[-1] if pd.notna(upper.iloc[-1]) else False
    return bool(squeeze and expansion and breakout), float(recent.iloc[-1])


def _session_from_ts(ts: datetime) -> str:
    est = ts.astimezone(US_EASTERN)
    minutes = est.hour * 60 + est.minute
    if 4 * 60 <= minutes < 9 * 60 + 30:
        return "PRE"
    if 9 * 60 + 30 <= minutes < 16 * 60:
        return "REG"
    if 16 * 60 <= minutes < 20 * 60:
        return "AFTER"
    return "CLOSED"


def _time_progress_ratio(session: str, ts: datetime) -> float:
    est = ts.astimezone(US_EASTERN)
    m = est.hour * 60 + est.minute
    if session == "REG":
        elapsed = max(1, min(390, m - (9 * 60 + 30)))
        return elapsed / 390.0
    if session == "PRE":
        elapsed = max(1, min(330, m - (4 * 60)))
        return elapsed / 330.0
    if session == "AFTER":
        elapsed = max(1, min(240, m - (16 * 60)))
        return elapsed / 240.0
    return 0.1


@dataclass
class ScannerConfig:
    include_etf: bool = False
    min_price: float = 1.0
    min_avg_daily_dollar_volume: float = 5_000_000.0
    min_avg_daily_volume: int = 100_000
    target_min_size: int = 500
    target_max_size: int = 1200
    breakout_threshold: float = 70.0
    score_delta_alert: float = 20.0
    alert_history_path: str = "data/cache/realtime_score_history.json"
    catalyst_enabled: bool = False


class RealtimeScannerEngine:
    def __init__(
        self,
        provider: MarketDataProvider,
        plan_store: PlanStore,
        telegram_sender: TelegramAlertSender | None = None,
        config: ScannerConfig | None = None,
    ) -> None:
        self.provider = provider
        self.plan_store = plan_store
        self.telegram = telegram_sender or TelegramAlertSender()
        self.config = config or ScannerConfig()
        self.risk_agent = RiskAgent()
        self.warnings: list[str] = []
        self.latest_scan_by_ticker: dict[str, dict[str, Any]] = {}
        self.breakout_alert_sent: set[str] = set()

    def _load_history(self) -> dict[str, float]:
        p = Path(self.config.alert_history_path)
        if not p.exists():
            return {}
        try:
            payload = json.loads(p.read_text(encoding="utf-8"))
            return {str(k): float(v) for k, v in payload.items()}
        except Exception:
            return {}

    def _save_history(self, scores: dict[str, float]) -> None:
        p = Path(self.config.alert_history_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(scores, indent=2), encoding="utf-8")

    def _prefilter_universe(self) -> tuple[list[str], dict[str, Any]]:
        universe = self.provider.get_universe(include_etf=self.config.include_etf)
        cfg = UniversePreFilterConfig(
            liquidity_threshold=self.config.min_avg_daily_volume,
            min_dollar_volume_20d=self.config.min_avg_daily_dollar_volume,
            min_price=self.config.min_price,
            min_history_days=30,
            target_min_size=self.config.target_min_size,
            target_max_size=self.config.target_max_size,
        )
        filtered, stats = build_prefiltered_universe(universe, cfg)
        return filtered, stats

    def _build_alert_message(
        self,
        ticker: str,
        session: str,
        price: float,
        old_score: float,
        new_score: float,
        surge_reason: list[str],
        strategy_label: str,
        key_signals: list[str],
        rationale: str,
        plan: dict[str, Any] | None = None,
    ) -> str:
        plan_text = ""
        if plan:
            entry = plan.get("entry")
            stop = plan.get("stop")
            t1 = plan.get("target1")
            t2 = plan.get("target2")
            if any(v is not None for v in [entry, stop, t1, t2]):
                plan_text = (
                    f"\n*Plan:* Entry {entry if entry is not None else '-'} / "
                    f"Stop {stop if stop is not None else '-'} / "
                    f"T1 {t1 if t1 is not None else '-'} / T2 {t2 if t2 is not None else '-'}"
                )

        reasons = ", ".join(surge_reason) if surge_reason else "None"
        return (
            f"📈 *Momentum Alert — {ticker}* ({session})\n"
            f"💲 *Price:* `{price:.2f}`\n"
            f"🔺 *Score:* `{old_score:.1f} -> {new_score:.1f}`\n"
            f"🚀 *SurgeReason:* {reasons}\n"
            f"🎯 *Strategy:* {strategy_label}\n"
            f"📊 *Signals:* {', '.join(key_signals[:6])}\n"
            f"💡 *Rationale:* {rationale[:180]}{plan_text}"
        )

    def _score_one(
        self,
        ticker: str,
        snapshot: Any,
        bars_1m: pd.DataFrame | None,
        bars_1d: pd.DataFrame | None,
    ) -> dict[str, Any] | None:
        if snapshot is None or bars_1m is None or bars_1m.empty or bars_1d is None or bars_1d.empty:
            return None

        price = float(snapshot.price)
        ts = snapshot.timestamp if isinstance(snapshot.timestamp, datetime) else datetime.now(timezone.utc)
        session = snapshot.session or _session_from_ts(ts)
        df1m = bars_1m.copy().dropna(how="any")
        if len(df1m) < 30:
            return None

        if df1m.index.tz is None:
            df1m.index = df1m.index.tz_localize("UTC")
        est_idx = df1m.index.tz_convert(US_EASTERN)
        today = datetime.now(US_EASTERN).date()
        today_mask = est_idx.date == today
        df_today = df1m.loc[today_mask] if today_mask.any() else df1m.tail(390)
        today_idx = est_idx[today_mask] if today_mask.any() else est_idx[-len(df_today):]
        if df_today.empty:
            return None

        close_1m = df_today["Close"]
        vol_1m = df_today["Volume"]
        chg_1m = float(snapshot.change_1m_pct or 0.0)
        chg_5m = float(snapshot.change_5m_pct or 0.0)
        if chg_5m == 0.0 and len(close_1m) >= 6 and close_1m.iloc[-6] != 0:
            chg_5m = ((price - float(close_1m.iloc[-6])) / float(close_1m.iloc[-6])) * 100

        cum_vol = float(vol_1m.sum())
        avg_vol_20d = float(bars_1d["Volume"].tail(20).mean())
        progress_ratio = _time_progress_ratio(session, ts)
        expected_intraday = max(1.0, avg_vol_20d * progress_ratio)
        rel_volume = cum_vol / expected_intraday

        vwap_series = ((df_today["Close"] * df_today["Volume"]).cumsum() / df_today["Volume"].replace(0, np.nan).cumsum())
        vwap = float(vwap_series.iloc[-1]) if pd.notna(vwap_series.iloc[-1]) else price
        vwap_distance = ((price - vwap) / vwap * 100) if vwap else 0.0
        vwap_reclaim = bool(price > vwap and len(close_1m) >= 4 and (close_1m.tail(4) < vwap).any())
        above_vwap = bool(price > vwap)

        rsi_1m = _rsi(close_1m, 14)
        bars_5m = df_today.resample("5min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(how="any")
        rsi_5m = _rsi(bars_5m["Close"], 14) if len(bars_5m) >= 20 else 50.0
        macd_line, macd_signal, macd_hist = _macd(bars_5m["Close"]) if len(bars_5m) >= 30 else (0.0, 0.0, 0.0)
        macd_bull = bool(macd_line > macd_signal and macd_hist > 0)
        macd_status = "Bullish" if macd_bull else "Neutral/Bearish"

        bars_15m = df_today.resample("15min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}).dropna(how="any")
        adx = _adx(bars_15m, 14) if len(bars_15m) >= 20 else 0.0

        est_series = df_today.copy()
        est_series.index = today_idx
        pre_mask = est_series.index.time < datetime(2000, 1, 1, 9, 30).time()
        premkt_high = float(est_series.loc[pre_mask, "High"].max()) if pre_mask.any() else float(df_today["High"].max())
        day_high_prev = float(df_today["High"].iloc[:-1].max()) if len(df_today) > 1 else float(df_today["High"].iloc[-1])
        high_20d = float(bars_1d["High"].tail(20).max())

        break_premarket = bool(session == "REG" and price > premkt_high)
        break_day_high = bool(price > day_high_prev)
        break_20d = bool(price > high_20d)
        breakout_any = bool(break_premarket or break_day_high or break_20d)

        prev_close = float(bars_1d["Close"].iloc[-2]) if len(bars_1d) >= 2 else float(bars_1d["Close"].iloc[-1])
        reg_open_rows = est_series[
            (est_series.index.time >= datetime(2000, 1, 1, 9, 30).time())
            & (est_series.index.time < datetime(2000, 1, 1, 9, 35).time())
        ]
        today_open = float(reg_open_rows["Open"].iloc[0]) if not reg_open_rows.empty else float(df_today["Open"].iloc[0])
        gap_pct = ((today_open - prev_close) / prev_close * 100) if prev_close else 0.0
        pre_rows = est_series.loc[pre_mask]
        pre_range_pct = (
            ((float(pre_rows["High"].max()) - float(pre_rows["Low"].min())) / prev_close * 100)
            if (not pre_rows.empty and prev_close)
            else 0.0
        )

        ema20, ema50, ema200, ema_stack = _ema_stack(df_today["Close"])
        bb_break, bb_width = _bb_squeeze_break(bars_5m["Close"] if len(bars_5m) >= 30 else df_today["Close"])
        volatility_expansion = bool(bb_break)

        score = 0.0
        key_signals: list[str] = []
        if breakout_any:
            score += 25
            key_signals.append("Breakout")
        if above_vwap or vwap_reclaim:
            score += 15
            key_signals.append("VWAP Reclaim/Above")
        if rel_volume >= 2.0:
            score += 20
            key_signals.append("RelVol Spike")
        if (rsi_1m >= 60 or rsi_5m >= 60) and macd_bull:
            score += 15
            key_signals.append("Momentum Confirmed")
        if ema_stack:
            score += 10
            key_signals.append("EMA Stack")
        if volatility_expansion:
            score += 10
            key_signals.append("BB Expansion")

        catalyst_score = 0.0
        catalyst_events = []
        if self.config.catalyst_enabled:
            catalyst_events = run_catalyst_agent_fast(ticker)
            if catalyst_events:
                catalyst_score = min(10.0, 5.0 + 2.0 * len(catalyst_events))
                score += catalyst_score
                key_signals.append("Catalyst")

        score = min(100.0, score)
        momentum_category = (
            "Strong Breakout Candidate" if score >= 70 else
            "Moderate Momentum" if score >= 50 else
            "Weak / Neutral" if score >= 30 else
            "No Momentum / Bearish"
        )
        strategy_label = "Swing-breakout" if break_20d else "Day-momentum"

        surge_reasons: list[str] = []
        if rel_volume >= 3.0 and chg_5m >= 2.0:
            surge_reasons.append("RelVol>=3 & 5m>=2%")
        if break_premarket and vwap_reclaim and rel_volume >= 2.0:
            surge_reasons.append("PremarketHigh Break + VWAP reclaim + Volume spike")
        if bb_break and breakout_any:
            surge_reasons.append("BB squeeze break + breakout")
        surge_flag = bool(surge_reasons)

        halt_risk_proxy = bool(rel_volume >= 5.0 and abs(chg_5m) >= 5.0 and pre_range_pct >= 4.0)

        rationale = (
            f"Breakout={breakout_any}, RelVol={rel_volume:.2f}, VWAPdist={vwap_distance:.2f}%, "
            f"RSI1m/5m={rsi_1m:.1f}/{rsi_5m:.1f}, MACD={macd_status}, ADX={adx:.1f}."
        )

        signals = {
            "break_premarket_high": break_premarket,
            "break_day_high": break_day_high,
            "break_20d_high": break_20d,
            "vwap_reclaim": vwap_reclaim,
            "above_vwap": above_vwap,
            "macd_bullish": macd_bull,
            "ema_stack": ema_stack,
            "bb_squeeze_break": bb_break,
            "halt_risk_proxy": halt_risk_proxy,
        }
        indicators = {
            "rsi_14_1m": round(rsi_1m, 2),
            "rsi_14_5m": round(rsi_5m, 2),
            "macd_5m": round(macd_line, 5),
            "macd_signal_5m": round(macd_signal, 5),
            "macd_hist_5m": round(macd_hist, 5),
            "adx": round(adx, 2),
            "vwap": round(vwap, 4),
            "vwap_distance_pct": round(vwap_distance, 2),
            "ema20": round(ema20, 4),
            "ema50": round(ema50, 4),
            "ema200": round(ema200, 4),
            "bb_bandwidth": round(bb_width, 5),
            "gap_pct": round(gap_pct, 2),
            "premarket_range_pct": round(pre_range_pct, 2),
            "premarket_high": round(premkt_high, 4),
            "day_high_prev": round(day_high_prev, 4),
            "high_20d": round(high_20d, 4),
        }

        return {
            "ticker": ticker,
            "last_price": round(price, 4),
            "session": session,
            "change_1m_pct": round(chg_1m, 2),
            "change_5m_pct": round(chg_5m, 2),
            "rel_volume": round(rel_volume, 2),
            "vwap_distance_pct": round(vwap_distance, 2),
            "rsi_14": round(rsi_5m, 2),
            "macd_status": macd_status,
            "adx": round(adx, 2),
            "breakout_flag": breakout_any,
            "score": round(score, 1),
            "surge_flag": surge_flag,
            "surge_reason": surge_reasons,
            "strategy_label": strategy_label,
            "momentum_category": momentum_category,
            "indicators": indicators,
            "signals": signals,
            "catalysts": [c.model_dump() for c in catalyst_events] if catalyst_events else [],
            "rationale": rationale,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def run_scan(self, send_alerts: bool = False) -> dict[str, Any]:
        self.warnings = []
        universe, prefilter_stats = self._prefilter_universe()
        if not universe:
            return {"results": [], "stats": prefilter_stats, "warnings": ["No tickers after prefilter."]}

        snapshots = self.provider.get_snapshot(universe)
        valid_tickers = [t for t in universe if t in snapshots]
        bars_1m = self.provider.get_bars(valid_tickers, "1m", limit=800, extended_hours=True)
        bars_1d = self.provider.get_bars(valid_tickers, "1d", limit=60, extended_hours=False)

        results: list[dict[str, Any]] = []
        for ticker in valid_tickers:
            row = self._score_one(ticker, snapshots.get(ticker), bars_1m.get(ticker), bars_1d.get(ticker))
            if row:
                results.append(row)

        results.sort(key=lambda x: x["score"], reverse=True)
        history = self._load_history()
        new_scores: dict[str, float] = {}

        for row in results:
            ticker = row["ticker"]
            old_score = float(history.get(ticker, 0.0))
            new_score = float(row["score"])
            new_scores[ticker] = new_score
            delta = new_score - old_score
            triggered = False
            if old_score > 0 and delta >= self.config.score_delta_alert:
                triggered = True
            if old_score < self.config.breakout_threshold <= new_score:
                triggered = True

            plan = self.plan_store.get(ticker)
            plan_dict = plan.__dict__ if plan else None

            alert_message = ""
            if triggered:
                alert_message = self._build_alert_message(
                    ticker=ticker,
                    session=row["session"],
                    price=row["last_price"],
                    old_score=old_score,
                    new_score=new_score,
                    surge_reason=row.get("surge_reason", []),
                    strategy_label=row.get("strategy_label", "Day-momentum"),
                    key_signals=[k for k, v in row.get("signals", {}).items() if v],
                    rationale=row.get("rationale", ""),
                    plan=plan_dict,
                )
                if send_alerts:
                    self.telegram.send_markdown(alert_message)

            row["alert"] = {
                "triggered": triggered,
                "old_score": round(old_score, 1),
                "new_score": round(new_score, 1),
                "message": alert_message,
            }
            row["telegram_message"] = alert_message
            row["composite_score"] = row["score"]
            row["trade_plan"] = plan_dict or {}

        self.latest_scan_by_ticker = {row["ticker"]: row for row in results}
        self._save_history(new_scores)
        return {
            "results": results,
            "stats": {
                **prefilter_stats,
                "provider": self.provider.name,
                "result_count": len(results),
            },
            "warnings": self.warnings,
        }

    def check_realtime_breakout_alert(self, ticker: str, session: str, price: float, send: bool = False) -> dict[str, Any]:
        """
        Realtime breakout alert:
        break premarket/20d high with volume surge + VWAP reclaim.
        """
        ticker = ticker.upper()
        row = self.latest_scan_by_ticker.get(ticker)
        if not row:
            return {"triggered": False, "message": ""}
        if ticker in self.breakout_alert_sent:
            return {"triggered": False, "message": ""}

        signals = row.get("signals", {})
        breakout_ok = bool(signals.get("break_premarket_high") or signals.get("break_20d_high"))
        vwap_ok = bool(signals.get("vwap_reclaim") or signals.get("above_vwap"))
        vol_ok = float(row.get("rel_volume", 0.0)) >= 2.0
        if session != "REG":
            return {"triggered": False, "message": ""}
        if not (breakout_ok and vwap_ok and vol_ok):
            return {"triggered": False, "message": ""}

        plan = self.plan_store.get(ticker)
        plan_dict = plan.__dict__ if plan else None
        message = self._build_alert_message(
            ticker=ticker,
            session=session,
            price=price,
            old_score=float(row.get("score", 0.0)),
            new_score=float(row.get("score", 0.0)),
            surge_reason=row.get("surge_reason", ["Realtime breakout event"]),
            strategy_label=row.get("strategy_label", "Day-momentum"),
            key_signals=[k for k, v in signals.items() if v],
            rationale=row.get("rationale", "Realtime breakout conditions met."),
            plan=plan_dict,
        )
        self.breakout_alert_sent.add(ticker)
        if send:
            self.telegram.send_markdown(message)
        return {"triggered": True, "message": message}

    def get_ticker_snapshot(self, ticker: str, timeframe: str = "1m") -> dict[str, Any]:
        ticker = ticker.upper().strip()
        snap = self.provider.get_snapshot([ticker]).get(ticker)
        if not snap:
            return {"ticker": ticker, "error": "snapshot_not_available"}

        bars = self.provider.get_bars([ticker], timeframe=timeframe, limit=600, extended_hours=True).get(ticker)
        bars_daily = self.provider.get_bars([ticker], timeframe="1d", limit=80, extended_hours=False).get(ticker)
        if bars is None or bars.empty or bars_daily is None or bars_daily.empty:
            return {"ticker": ticker, "error": "bars_not_available"}

        if bars.index.tz is None:
            bars.index = bars.index.tz_localize("UTC")
        close = bars["Close"]
        vwap = ((bars["Close"] * bars["Volume"]).cumsum() / bars["Volume"].replace(0, np.nan).cumsum()).ffill()
        ema20 = close.ewm(span=20, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean()
        ma = close.rolling(20).mean()
        std = close.rolling(20).std()
        bb_up = ma + 2 * std
        bb_low = ma - 2 * std
        rsi_val = _rsi(close, 14)
        macd_line, macd_signal, macd_hist = _macd(close)

        latest_row = self.latest_scan_by_ticker.get(ticker, {})
        latest_score = float(latest_row.get("score", 0.0) or 0.0)
        if latest_score >= 70:
            signal = Signal.STRONG_BUY
        elif latest_score >= 50:
            signal = Signal.BUY
        else:
            signal = Signal.HOLD

        high_20d = float(bars_daily["High"].tail(20).max())
        daily_ema20 = float(bars_daily["Close"].ewm(span=20, adjust=False).mean().iloc[-1])
        context = {
            "breakout": bool(latest_row.get("breakout_flag", bool(snap.price > high_20d))),
            "adx_strong": bool(float(latest_row.get("adx", 0.0) or 0.0) >= 25.0),
            "ema20": float(latest_row.get("indicators", {}).get("ema20", daily_ema20)),
            "resistance_20": float(latest_row.get("indicators", {}).get("high_20d", high_20d)),
        }
        suggested_plan = self.risk_agent.plan_from_df(bars_daily.tail(120), signal=signal, context=context)
        suggested_plan_payload = suggested_plan.model_dump(mode="json") if suggested_plan else {}

        saved_plan = self.plan_store.get(ticker)
        saved_plan_payload = saved_plan.__dict__ if saved_plan else {}

        return {
            "ticker": ticker,
            "session": snap.session,
            "price": snap.price,
            "timestamp": snap.timestamp.isoformat(),
            "timeframe": timeframe,
            "bars": [
                {
                    "timestamp": idx.isoformat(),
                    "open": float(row["Open"]),
                    "high": float(row["High"]),
                    "low": float(row["Low"]),
                    "close": float(row["Close"]),
                    "volume": float(row["Volume"]),
                    "vwap": float(vwap.loc[idx]) if pd.notna(vwap.loc[idx]) else None,
                    "ema20": float(ema20.loc[idx]) if pd.notna(ema20.loc[idx]) else None,
                    "ema50": float(ema50.loc[idx]) if pd.notna(ema50.loc[idx]) else None,
                    "ema200": float(ema200.loc[idx]) if pd.notna(ema200.loc[idx]) else None,
                    "bb_upper": float(bb_up.loc[idx]) if pd.notna(bb_up.loc[idx]) else None,
                    "bb_lower": float(bb_low.loc[idx]) if pd.notna(bb_low.loc[idx]) else None,
                }
                for idx, row in bars.tail(300).iterrows()
            ],
            "indicators": {
                "rsi_14": round(rsi_val, 2),
                "macd": round(macd_line, 5),
                "macd_signal": round(macd_signal, 5),
                "macd_hist": round(macd_hist, 5),
                "vwap": round(float(vwap.iloc[-1]), 4) if not vwap.empty and pd.notna(vwap.iloc[-1]) else None,
                "ema20": round(float(ema20.iloc[-1]), 4),
                "ema50": round(float(ema50.iloc[-1]), 4),
                "ema200": round(float(ema200.iloc[-1]), 4),
            },
            "signals": latest_row.get("signals", {}),
            "composite_score": latest_score,
            "strategy_label": latest_row.get("strategy_label", "Day-momentum"),
            "rationale": latest_row.get("rationale", ""),
            "plan": saved_plan_payload,
            "recommended_trade_plan": suggested_plan_payload,
        }

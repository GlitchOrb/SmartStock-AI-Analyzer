from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from datetime import datetime
import time
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

from agents.catalyst_agent import run_catalyst_agent_deep, run_catalyst_agent_fast
from agents.risk_agent import RiskAgent
from data.universe import (
    UniverseConfig,
    UniversePreFilterConfig,
    build_prefiltered_universe,
    build_universe,
)
from schemas.agents import ScreenerAlert, ScreenerResult, UICard
from schemas.config import settings
from schemas.enums import Signal
from utils.telegram import TelegramAlerter


SCORING_WEIGHTS = {
    "Volume Surge": 20,
    "Breakout": 30,
    "RSI Strong": 15,
    "RSI Moderate": 10,
    "MACD Bullish": 15,
    "OBV Uptrend": 10,
    "ADX Strong Trend": 15,
    "Stochastic Bullish": 10,
    "Bullish MA Alignment": 15,
}
MAX_COMPOSITE_POINTS = 130.0
ALERT_STRONG_THRESHOLD = 70.0


def _download_with_retry(
    tickers: list[str] | str,
    *,
    period: str,
    interval: str = "1d",
    group_by: str = "ticker",
    attempts: int = 3,
    sleep_s: float = 1.2,
) -> pd.DataFrame:
    last_exc: Exception | None = None
    for attempt in range(1, attempts + 1):
        try:
            data = yf.download(
                tickers,
                period=period,
                interval=interval,
                group_by=group_by,
                threads=True,
                progress=False,
                auto_adjust=False,
            )
            if isinstance(data, pd.DataFrame) and not data.empty:
                return data
        except Exception as exc:
            last_exc = exc
        if attempt < attempts:
            time.sleep(sleep_s * attempt)
    if last_exc:
        raise last_exc
    return pd.DataFrame()


def _chunked(items: list[str], size: int) -> list[list[str]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def _normalize_columns(data: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(data.columns, pd.MultiIndex):
        return data
    if "Close" in list(map(str, data.columns.levels[0])):
        return data.swaplevel(0, 1, axis=1)
    return data


def _extract_df(batch: pd.DataFrame, ticker: str, is_single: bool) -> pd.DataFrame | None:
    if is_single:
        df = batch.copy()
    elif isinstance(batch.columns, pd.MultiIndex) and ticker in batch.columns.get_level_values(0):
        df = batch[ticker].copy()
    else:
        return None

    required = ["Open", "High", "Low", "Close", "Volume"]
    if any(col not in df.columns for col in required):
        return None
    df = df[required].dropna(how="any")
    return df if not df.empty else None


def _rsi14(close: pd.Series) -> float:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    val = rsi.iloc[-1]
    return float(val) if pd.notna(val) else 50.0


def _macd(close: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def _adx14(high: pd.Series, low: pd.Series, close: pd.Series) -> float:
    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0.0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0.0)

    tr = pd.concat(
        [
            (high - low),
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr = tr.ewm(alpha=1 / 14, adjust=False).mean()

    plus_di = 100 * (plus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan))
    minus_di = 100 * (minus_dm.ewm(alpha=1 / 14, adjust=False).mean() / atr.replace(0, np.nan))
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean().iloc[-1]
    return float(adx) if pd.notna(adx) else 0.0


def _stochastic(high: pd.Series, low: pd.Series, close: pd.Series) -> tuple[float, float, bool]:
    lowest = low.rolling(14).min()
    highest = high.rolling(14).max()
    denom = (highest - lowest).replace(0, np.nan)
    k = 100 * (close - lowest) / denom
    d = k.rolling(3).mean()

    k_last = float(k.iloc[-1]) if pd.notna(k.iloc[-1]) else 50.0
    d_last = float(d.iloc[-1]) if pd.notna(d.iloc[-1]) else 50.0

    bullish = False
    if len(k) >= 2 and pd.notna(k.iloc[-2]) and pd.notna(d.iloc[-2]) and pd.notna(k.iloc[-1]) and pd.notna(d.iloc[-1]):
        bullish = bool((k.iloc[-2] <= d.iloc[-2]) and (k.iloc[-1] > d.iloc[-1]))
    return k_last, d_last, bullish


def _obv_stats(close: pd.Series, volume: pd.Series) -> tuple[pd.Series, float, bool]:
    direction = np.sign(close.diff()).fillna(0.0)
    obv = (direction * volume).cumsum()
    if len(obv) < 10:
        return obv, 0.0, False

    y = obv.tail(10).to_numpy(dtype=float)
    x = np.arange(len(y), dtype=float)
    slope = float(np.polyfit(x, y, 1)[0]) if len(y) >= 2 else 0.0
    uptrend = bool((obv.iloc[-1] > obv.iloc[-6]) and slope > 0)
    return obv, slope, uptrend


class MarketScreener:
    def __init__(
        self,
        tickers: list[str] | None = None,
        universe_kind: str = "nasdaq_all",
        universe_size: int = 0,
        liquidity_threshold: int = 100_000,
        min_dollar_volume_20d: float = 5_000_000.0,
        min_price: float = 1.0,
        min_history_days: int = 30,
        target_universe_min: int = 500,
        target_universe_max: int = 1200,
        catalyst_lookahead_days: int = 14,
        deep_catalyst_top_n: int = 30,
        fast_catalyst_top_n: int = 250,
    ) -> None:
        self.risk = RiskAgent()
        self.catalyst_lookahead_days = catalyst_lookahead_days
        self.deep_catalyst_top_n = deep_catalyst_top_n
        self.fast_catalyst_top_n = fast_catalyst_top_n
        self.warnings: list[str] = []
        self.prefilter_cfg = UniversePreFilterConfig(
            liquidity_threshold=liquidity_threshold,
            min_dollar_volume_20d=min_dollar_volume_20d,
            min_price=min_price,
            min_history_days=min_history_days,
            target_min_size=target_universe_min,
            target_max_size=target_universe_max,
        )
        self.prefilter_stats: dict[str, Any] = {}

        if tickers:
            self.universe_tickers = sorted(list(dict.fromkeys([t.upper().strip() for t in tickers if t.strip()])))
        else:
            universe_cfg = UniverseConfig(
                kind=universe_kind,
                max_size=universe_size,
                include_etf=False,
            )
            self.universe_tickers = build_universe(universe_cfg)

    def _history_path(self) -> Any:
        return settings.cache_dir / "screener_history.json"

    def _load_history(self) -> dict[str, dict[str, Any]]:
        path = self._history_path()
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            out: dict[str, dict[str, Any]] = {}
            for ticker, value in payload.items():
                if isinstance(value, dict):
                    out[ticker] = {
                        "score": float(value.get("score", 0.0)),
                        "signals": list(value.get("signals", [])),
                    }
                else:
                    out[ticker] = {"score": float(value), "signals": []}
            return out
        except Exception:
            return {}

    def _save_history(self, results: list[ScreenerResult]) -> None:
        payload = {
            r.ticker: {
                "score": round(float(r.score), 2),
                "signals": list(r.key_positive_signals),
            }
            for r in results
        }
        self._history_path().write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def _build_prefiltered_universe(self) -> list[str]:
        selected, stats = build_prefiltered_universe(self.universe_tickers, self.prefilter_cfg)
        self.prefilter_stats = stats
        if not selected:
            self.warnings.append("No tickers passed prefilter thresholds (history/liquidity).")
        return selected

    def _fetch_history_for_scan(self, tickers: list[str], period: str = "9mo", chunk_size: int = 200) -> dict[str, pd.DataFrame]:
        out: dict[str, pd.DataFrame] = {}
        for chunk in _chunked(tickers, chunk_size):
            try:
                raw = _download_with_retry(
                    chunk,
                    period=period,
                    interval="1d",
                    group_by="ticker",
                )
            except Exception as exc:
                self.warnings.append(f"yfinance download failed for chunk size={len(chunk)}: {exc}")
                continue

            batch = _normalize_columns(raw)
            is_single = not isinstance(batch.columns, pd.MultiIndex)
            for ticker in chunk:
                df = _extract_df(batch, ticker, is_single=(is_single and len(chunk) == 1))
                if df is not None:
                    out[ticker] = df
        return out

    def _compute_signals(self, df: pd.DataFrame) -> dict[str, Any] | None:
        if df is None or df.empty or len(df) < self.prefilter_cfg.min_history_days:
            return None

        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        price = float(close.iloc[-1])
        prev = float(close.iloc[-2]) if len(close) >= 2 else price
        change_pct = ((price - prev) / prev) * 100 if prev else 0.0

        avg_vol_20 = float(volume.tail(20).mean())
        rel_vol = float(volume.iloc[-1] / avg_vol_20) if avg_vol_20 > 0 else 1.0
        volume_surge = rel_vol >= 2.0

        resistance_20 = float(close.shift(1).rolling(20).max().iloc[-1])
        breakout = bool(price > resistance_20) if pd.notna(resistance_20) else False

        rsi = _rsi14(close)
        rsi_strong = rsi >= 60
        rsi_moderate = 50 <= rsi < 60

        macd_line, macd_signal, macd_hist = _macd(close)
        macd_last = float(macd_line.iloc[-1])
        macd_signal_last = float(macd_signal.iloc[-1])
        macd_hist_last = float(macd_hist.iloc[-1])
        macd_bullish = bool((macd_last > macd_signal_last) and all(macd_hist.tail(3) > 0))

        adx_val = _adx14(high, low, close)
        adx_strong = adx_val >= 25

        _, obv_slope, obv_uptrend = _obv_stats(close, volume)
        k_val, d_val, stochastic_bullish = _stochastic(high, low, close)

        ema20 = float(close.ewm(span=20, adjust=False).mean().iloc[-1])
        ema50 = float(close.ewm(span=50, adjust=False).mean().iloc[-1])
        ema120 = float(close.ewm(span=120, adjust=False).mean().iloc[-1])
        ma_alignment = bool((ema20 > ema50) and (ema50 > ema120))

        atr14 = self.risk.atr14(df)

        raw_score = 0
        flags: list[dict[str, Any]] = []

        def add_flag(name: str, passed: bool, points: int, value: Any = None) -> None:
            nonlocal raw_score
            flags.append({"name": name, "flag": bool(passed), "points": points, "value": value})
            if passed:
                raw_score += points

        add_flag("Volume Surge", volume_surge, SCORING_WEIGHTS["Volume Surge"], round(rel_vol, 2))
        add_flag("Breakout", breakout, SCORING_WEIGHTS["Breakout"], round(resistance_20, 2))

        if rsi_strong:
            add_flag("RSI >= 60", True, SCORING_WEIGHTS["RSI Strong"], round(rsi, 2))
        elif rsi_moderate:
            add_flag("RSI 50-60", True, SCORING_WEIGHTS["RSI Moderate"], round(rsi, 2))
        else:
            add_flag("RSI 50-60", False, SCORING_WEIGHTS["RSI Moderate"], round(rsi, 2))

        add_flag("MACD Bullish", macd_bullish, SCORING_WEIGHTS["MACD Bullish"], round(macd_hist_last, 4))
        add_flag("OBV Uptrend", obv_uptrend, SCORING_WEIGHTS["OBV Uptrend"], round(obv_slope, 2))
        add_flag("ADX Strong Trend", adx_strong, SCORING_WEIGHTS["ADX Strong Trend"], round(adx_val, 2))
        add_flag("Stochastic Bullish", stochastic_bullish, SCORING_WEIGHTS["Stochastic Bullish"], round(k_val - d_val, 2))
        add_flag("Bullish MA Alignment", ma_alignment, SCORING_WEIGHTS["Bullish MA Alignment"], round(ema20 - ema50, 2))

        composite = min(100.0, (raw_score / MAX_COMPOSITE_POINTS) * 100.0)
        if composite >= 70:
            category = "Strong Breakout Candidate"
            signal = Signal.STRONG_BUY
        elif composite >= 50:
            category = "Moderate Momentum"
            signal = Signal.BUY
        elif composite >= 30:
            category = "Weak / Neutral"
            signal = Signal.HOLD
        else:
            category = "No Momentum / Bearish"
            signal = Signal.SELL

        short_swing = breakout and volume_surge and macd_bullish
        mid_swing = breakout and adx_strong and ma_alignment
        if short_swing:
            strategy_label = "Short Swing"
            strategy_rationale = "Recent breakout with volume surge and bullish MACD confirms short-term momentum."
        elif mid_swing:
            strategy_label = "Mid Swing"
            strategy_rationale = "Breakout with ADX>=25 and EMA20>EMA50>EMA120 supports a sustained trend setup."
        else:
            strategy_label = "Hold / Watch"
            strategy_rationale = "Momentum setup is incomplete; wait for stronger confirmation."

        positive_signals = [f["name"] for f in flags if f["flag"]]
        indicator_values = {
            "price": round(price, 2),
            "change_pct_1d": round(change_pct, 2),
            "avg_volume_20d": round(avg_vol_20, 2),
            "relative_volume": round(rel_vol, 2),
            "volume_surge": volume_surge,
            "resistance_close_20d": round(resistance_20, 2),
            "breakout": breakout,
            "rsi_14": round(rsi, 2),
            "macd_line": round(macd_last, 5),
            "macd_signal": round(macd_signal_last, 5),
            "macd_histogram": round(macd_hist_last, 5),
            "macd_bullish": macd_bullish,
            "adx_14": round(adx_val, 2),
            "adx_strong": adx_strong,
            "obv_slope_10d": round(obv_slope, 2),
            "obv_uptrend": obv_uptrend,
            "stochastic_k": round(k_val, 2),
            "stochastic_d": round(d_val, 2),
            "stochastic_bullish": stochastic_bullish,
            "ema_20": round(ema20, 2),
            "ema_50": round(ema50, 2),
            "ema_120": round(ema120, 2),
            "ma_alignment": ma_alignment,
            "atr_14": round(atr14, 2),
        }
        signals = {
            "volume_surge": volume_surge,
            "breakout": breakout,
            "rsi_strong": rsi_strong,
            "rsi_moderate": rsi_moderate,
            "macd_bullish": macd_bullish,
            "obv_uptrend": obv_uptrend,
            "adx_strong": adx_strong,
            "stochastic_bullish": stochastic_bullish,
            "bullish_ma_alignment": ma_alignment,
        }

        rationale = (
            f"Composite {composite:.1f}/100 driven by {', '.join(positive_signals[:5]) or 'no strong positive signals'}. "
            f"Trend summary: ADX {adx_val:.1f}, RSI {rsi:.1f}, MACD histogram {macd_hist_last:.3f}."
        )

        return {
            "price": price,
            "change_pct": change_pct,
            "avg_vol_20": avg_vol_20,
            "rel_vol": rel_vol,
            "resistance_20": resistance_20,
            "rsi": rsi,
            "macd_line": macd_last,
            "macd_signal": macd_signal_last,
            "macd_hist": macd_hist_last,
            "adx": adx_val,
            "obv_slope": obv_slope,
            "stoch_k": k_val,
            "stoch_d": d_val,
            "ema20": ema20,
            "ema50": ema50,
            "ema120": ema120,
            "atr14": atr14,
            "composite": composite,
            "category": category,
            "signal": signal,
            "flags": flags,
            "positive_signals": positive_signals,
            "strategy_label": strategy_label,
            "strategy_rationale": strategy_rationale,
            "indicator_values": indicator_values,
            "signals": signals,
            "rationale": rationale,
            "context": {
                "breakout": breakout,
                "adx_strong": adx_strong,
                "ema20": ema20,
                "resistance_20": resistance_20,
            },
        }

    @staticmethod
    def _changed_signals(old: list[str], new: list[str]) -> list[str]:
        old_set = set(old)
        new_set = set(new)
        added = [f"+ {name}" for name in sorted(new_set - old_set)]
        removed = [f"- {name}" for name in sorted(old_set - new_set)]
        return added + removed

    @staticmethod
    def _format_telegram_message(
        *,
        ticker: str,
        old_score: float,
        new_score: float,
        delta: float,
        strategy_label: str,
        key_signals: list[str],
        rationale: str,
    ) -> str:
        signals_text = ", ".join(key_signals[:6]) if key_signals else "None"
        short_rationale = rationale[:180] + ("..." if len(rationale) > 180 else "")
        return (
            f"📈 *Momentum Alert — {ticker}*\n"
            f"🔺 *Score* ↑ {delta:+.1f} ({old_score:.1f} → {new_score:.1f})\n"
            f"🎯 *Strategy:* {strategy_label}\n"
            f"📊 *Signals:* {signals_text}\n"
            f"💡 *Rationale:* {short_rationale}"
        )

    def _build_result(
        self,
        ticker: str,
        signal_data: dict[str, Any],
        prev_state: dict[str, Any],
    ) -> ScreenerResult:
        prev_score = float(prev_state.get("score", 0.0))
        prev_signals = list(prev_state.get("signals", []))
        new_score = float(signal_data["composite"])
        score_change = new_score - prev_score
        changed_signals = self._changed_signals(prev_signals, signal_data["positive_signals"])

        alert_triggered = False
        has_prev_score = bool(prev_state)
        if has_prev_score:
            if score_change >= 20:
                alert_triggered = True
            if prev_score < ALERT_STRONG_THRESHOLD <= new_score:
                alert_triggered = True

        alert_text = ""
        telegram_message = ""
        if alert_triggered:
            changed_text = ", ".join(changed_signals[:6]) if changed_signals else "signal mix changed"
            alert_text = (
                f"{ticker} score moved {prev_score:.1f} -> {new_score:.1f} ({score_change:+.1f}); "
                f"changes: {changed_text}."
            )
            telegram_message = self._format_telegram_message(
                ticker=ticker,
                old_score=prev_score,
                new_score=new_score,
                delta=score_change,
                strategy_label=(signal_data["strategy_label"] or "Hold / Watch"),
                key_signals=signal_data["positive_signals"],
                rationale=signal_data["rationale"],
            )

        signal_enum = signal_data["signal"]
        trade_plan = None
        if signal_enum in {Signal.BUY, Signal.STRONG_BUY}:
            trade_plan = self.risk.plan_from_df(df=signal_data["df"], signal=signal_enum, context=signal_data["context"])

        alert = ScreenerAlert(
            ticker=ticker,
            old_score=round(prev_score, 1),
            new_score=round(new_score, 1),
            score_change=round(score_change, 1),
            key_signals_changed=changed_signals,
            alert_triggered=alert_triggered,
            alert_text=alert_text,
            triggered=alert_triggered,
            message=alert_text,
            timestamp=datetime.now(),
        )

        ui_card = UICard(
            title=f"{ticker} | {signal_data['category']}",
            subtitle=f"Score {new_score:.1f} | {signal_data['strategy_label'] or 'Watch'}",
            key_metrics={
                "Price": f"${signal_data['price']:.2f}",
                "Change": f"{signal_data['change_pct']:+.2f}%",
                "RelVol": f"{signal_data['rel_vol']:.2f}x",
                "RSI": f"{signal_data['rsi']:.1f}",
            },
            detail_link=f"/ticker/{ticker}",
        )

        return ScreenerResult(
            ticker=ticker,
            company_name=ticker,
            current_price=round(signal_data["price"], 2),
            change_pct_1d=round(signal_data["change_pct"], 2),
            volume_relative=round(signal_data["rel_vol"], 2),
            average_volume_20d=round(signal_data["avg_vol_20"], 2),
            breakout_resistance_20d=round(signal_data["resistance_20"], 2),
            rsi_14=round(signal_data["rsi"], 2),
            macd_line=round(signal_data["macd_line"], 5),
            macd_signal=round(signal_data["macd_signal"], 5),
            macd_histogram=round(signal_data["macd_hist"], 5),
            adx_14=round(signal_data["adx"], 2),
            obv_slope_10d=round(signal_data["obv_slope"], 2),
            stochastic_k=round(signal_data["stoch_k"], 2),
            stochastic_d=round(signal_data["stoch_d"], 2),
            ema_20=round(signal_data["ema20"], 2),
            ema_50=round(signal_data["ema50"], 2),
            ema_120=round(signal_data["ema120"], 2),
            atr_14=round(signal_data["atr14"], 2),
            signal=signal_data["signal"],
            momentum_category=signal_data["category"],
            indicators=signal_data["positive_signals"],
            indicator_values=signal_data["indicator_values"],
            signals=signal_data["signals"],
            trade_plan=trade_plan,
            score=round(signal_data["composite"], 1),
            composite_score=round(signal_data["composite"], 1),
            key_positive_signals=signal_data["positive_signals"],
            rationale=signal_data["rationale"],
            signal_flags=signal_data["flags"],
            strategy_label=signal_data["strategy_label"],
            strategy_rationale=signal_data["strategy_rationale"],
            alert=alert,
            telegram_message=telegram_message,
            ui_card=ui_card,
        )

    def _attach_catalysts(self, results: list[ScreenerResult]) -> None:
        if not results:
            return

        ranked = sorted(results, key=lambda r: r.score, reverse=True)
        fast_candidates = ranked[: self.fast_catalyst_top_n]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {
                executor.submit(run_catalyst_agent_fast, r.ticker, self.catalyst_lookahead_days): r
                for r in fast_candidates
            }
            for future in as_completed(futures):
                result = futures[future]
                try:
                    result.catalysts = future.result()
                except Exception:
                    result.catalysts = []
                    self.warnings.append(f"Fast catalyst fetch failed for {result.ticker}.")

        top_candidates = ranked[: self.deep_catalyst_top_n]
        for result in top_candidates:
            try:
                deep_events = run_catalyst_agent_deep(result.ticker, self.catalyst_lookahead_days)
                if deep_events:
                    result.catalysts = deep_events
            except Exception:
                self.warnings.append(f"Deep catalyst fetch failed for {result.ticker}.")
                continue

    def run(self, send_telegram: bool = False) -> list[ScreenerResult]:
        prefiltered_tickers = self._build_prefiltered_universe()
        history = self._load_history()
        history_frames = self._fetch_history_for_scan(prefiltered_tickers)

        results: list[ScreenerResult] = []
        for ticker in prefiltered_tickers:
            df = history_frames.get(ticker)
            if df is None:
                self.warnings.append(f"Missing OHLCV for {ticker}; skipped.")
                continue
            signal_data = self._compute_signals(df)
            if signal_data is None:
                self.warnings.append(f"Insufficient indicator history for {ticker}; skipped.")
                continue
            signal_data["df"] = df
            prev = history.get(ticker, {})
            results.append(self._build_result(ticker, signal_data, prev))

        self._attach_catalysts(results)
        results.sort(key=lambda r: r.score, reverse=True)
        self._save_history(results)
        if send_telegram:
            self.dispatch_telegram_alerts(results)
        return results

    def dispatch_telegram_alerts(self, results: list[ScreenerResult]) -> list[dict[str, Any]]:
        """Send triggered alerts to Telegram if credentials are configured."""
        alerter = TelegramAlerter()
        delivery_logs: list[dict[str, Any]] = []
        for result in results:
            if not result.alert or not result.alert.alert_triggered:
                continue
            message = result.telegram_message or self._format_telegram_message(
                ticker=result.ticker,
                old_score=result.alert.old_score,
                new_score=result.alert.new_score,
                delta=result.alert.score_change,
                strategy_label=(result.strategy_label or "Hold / Watch"),
                key_signals=result.key_positive_signals,
                rationale=result.rationale,
            )
            ok, response = alerter.send_markdown(message)
            delivery_logs.append(
                {
                    "ticker": result.ticker,
                    "sent": ok,
                    "response": response,
                }
            )
        return delivery_logs

    @staticmethod
    def to_output_object(result: ScreenerResult) -> dict[str, Any]:
        """JSON-serializable output structure for downstream APIs/UI."""
        return {
            "ticker": result.ticker,
            "composite_score": float(result.composite_score or result.score),
            "momentum_category": result.momentum_category,
            "strategy_label": result.strategy_label or "Hold / Watch",
            "indicators": dict(result.indicator_values),
            "signals": dict(result.signals),
            "trade_plan": result.trade_plan.model_dump() if result.trade_plan else {},
            "catalysts": [c.model_dump() for c in result.catalysts],
            "rationale": result.rationale,
            "alert": {
                "triggered": bool(result.alert.alert_triggered) if result.alert else False,
                "old_score": float(result.alert.old_score) if result.alert else 0.0,
                "new_score": float(result.alert.new_score) if result.alert else float(result.score),
                "message": result.alert.alert_text if result.alert else "",
            },
            "telegram_message": result.telegram_message,
        }

    @staticmethod
    def to_output_objects(results: list[ScreenerResult]) -> list[dict[str, Any]]:
        return [MarketScreener.to_output_object(r) for r in results]

    @staticmethod
    def format_telegram_alert(result: ScreenerResult) -> dict[str, str]:
        if not result.alert or not result.alert.alert_triggered:
            return {}

        msg = result.telegram_message or MarketScreener._format_telegram_message(
            ticker=result.ticker,
            old_score=result.alert.old_score,
            new_score=result.alert.new_score,
            delta=result.alert.score_change,
            strategy_label=result.strategy_label or "Hold / Watch",
            key_signals=result.key_positive_signals,
            rationale=result.rationale,
        )
        return {"text": msg, "parse_mode": "Markdown"}

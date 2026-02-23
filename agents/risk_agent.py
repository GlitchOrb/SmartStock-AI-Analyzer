from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd

from schemas.agents import TradePlan
from schemas.enums import Signal


class RiskAgent:
    """Deterministic ATR-based trade planning for momentum setups."""

    @staticmethod
    def atr14(df: pd.DataFrame) -> float:
        high = df["High"]
        low = df["Low"]
        close = df["Close"]
        prev_close = close.shift(1)

        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(14).mean().iloc[-1]
        return float(atr) if pd.notna(atr) else 0.0

    def plan_from_df(
        self,
        df: pd.DataFrame,
        signal: Signal,
        context: dict[str, Any] | None = None,
    ) -> TradePlan | None:
        if df is None or df.empty or len(df) < 30:
            return None
        if signal not in {Signal.BUY, Signal.STRONG_BUY}:
            return None

        context = context or {}
        price = float(df["Close"].iloc[-1])
        atr = self.atr14(df)
        if atr <= 0:
            return None

        breakout = bool(context.get("breakout", False))
        adx_strong = bool(context.get("adx_strong", False))
        ema20 = float(context.get("ema20", price))
        resistance = float(context.get("resistance_20", price))

        if breakout:
            entry = max(price, resistance + (0.10 * atr))
        else:
            entry = price

        trend_stop = ema20 - (0.75 * atr)
        vol_stop = entry - (2.0 * atr)
        stop = min(trend_stop, vol_stop)

        if adx_strong:
            tp1 = entry + (2.2 * atr)
            tp2 = entry + (3.8 * atr)
        else:
            tp1 = entry + (1.7 * atr)
            tp2 = entry + (3.0 * atr)

        risk = abs(entry - stop)
        reward = abs(tp1 - entry)
        rr = (reward / risk) if risk > 0 else 0.0

        rationale = (
            f"ATR(14)={atr:.2f}. Entry uses {'breakout confirmation' if breakout else 'current trend'}; "
            f"stop anchored to EMA20/volatility; targets expand with {'strong' if adx_strong else 'moderate'} trend."
        )

        return TradePlan(
            strategy_name="ATR Momentum",
            entry_price=round(entry, 2),
            stop_loss=round(stop, 2),
            target_price_1=round(tp1, 2),
            target_price_2=round(tp2, 2),
            risk_reward_ratio=round(rr, 2),
            rationale=rationale,
            invalidation_triggers=[
                "Close breaks and holds below stop-loss level",
                "MACD turns bearish with declining relative volume",
            ],
            generated_at=datetime.now(),
        )

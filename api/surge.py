from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class SurgeInput:
    session: str
    trigger_time: str
    rel_volume: float
    change_5m_pct: float
    above_vwap: bool
    vwap_reclaim: bool
    break_premarket_high: bool
    breakout_any: bool
    bb_squeeze_break: bool
    bb_bandwidth_now: float
    bb_bandwidth_prev: float
    orb_breakout: bool


def evaluate_surge(inp: SurgeInput) -> dict[str, Any]:
    reasons: list[str] = []

    cond_a = inp.rel_volume >= 3.0 and inp.change_5m_pct >= 2.0 and inp.above_vwap
    if cond_a:
        reasons.append("RelVol>=3 + 5m>=2% + Above VWAP")

    cond_b = (
        inp.session == "REG"
        and inp.break_premarket_high
        and inp.vwap_reclaim
        and inp.rel_volume >= 2.0
    )
    if cond_b:
        reasons.append("Premarket high break + VWAP reclaim + volume spike")

    bandwidth_expansion = (
        (inp.bb_bandwidth_prev > 0)
        and (inp.bb_bandwidth_now >= inp.bb_bandwidth_prev * 1.2)
    )
    cond_c = inp.bb_squeeze_break and inp.breakout_any and bandwidth_expansion
    if cond_c:
        reasons.append("BB squeeze break + bandwidth expansion + breakout")

    cond_d = inp.session == "REG" and inp.orb_breakout and inp.rel_volume >= 2.0
    if cond_d:
        reasons.append("ORB high break + volume confirmation")

    return {
        "surge_flag": bool(reasons),
        "surge_reasons": reasons,
        "trigger_bar_time": inp.trigger_time,
        "supporting_evidence": {
            "rel_volume": round(inp.rel_volume, 3),
            "change_5m_pct": round(inp.change_5m_pct, 3),
            "above_vwap": inp.above_vwap,
            "vwap_reclaim": inp.vwap_reclaim,
            "bb_bandwidth_now": round(inp.bb_bandwidth_now, 6),
            "bb_bandwidth_prev": round(inp.bb_bandwidth_prev, 6),
            "break_premarket_high": inp.break_premarket_high,
            "orb_breakout": inp.orb_breakout,
        },
    }

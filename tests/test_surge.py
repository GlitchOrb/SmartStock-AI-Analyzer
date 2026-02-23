from api.surge import SurgeInput, evaluate_surge


def test_surge_condition_a():
    result = evaluate_surge(
        SurgeInput(
            session="REG",
            trigger_time="2026-02-23T14:31:00+00:00",
            rel_volume=3.4,
            change_5m_pct=2.5,
            above_vwap=True,
            vwap_reclaim=False,
            break_premarket_high=False,
            breakout_any=True,
            bb_squeeze_break=False,
            bb_bandwidth_now=0.04,
            bb_bandwidth_prev=0.03,
            orb_breakout=False,
        )
    )
    assert result["surge_flag"] is True
    assert any("RelVol" in reason for reason in result["surge_reasons"])


def test_surge_condition_c_bandwidth_expansion():
    result = evaluate_surge(
        SurgeInput(
            session="REG",
            trigger_time="2026-02-23T15:01:00+00:00",
            rel_volume=1.8,
            change_5m_pct=1.2,
            above_vwap=True,
            vwap_reclaim=True,
            break_premarket_high=False,
            breakout_any=True,
            bb_squeeze_break=True,
            bb_bandwidth_now=0.06,
            bb_bandwidth_prev=0.04,
            orb_breakout=False,
        )
    )
    assert result["surge_flag"] is True
    assert any("BB squeeze break" in reason for reason in result["surge_reasons"])

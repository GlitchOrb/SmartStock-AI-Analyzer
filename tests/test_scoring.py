from api.scanner_engine import _compose_score


def test_compose_score_weights():
    score, breakdown, positives = _compose_score(
        breakout_any=True,
        vwap_ok=True,
        rel_volume_spike=True,
        momentum_confirmed=True,
        ema_stack=True,
        volatility_expansion=True,
        catalyst_score=8.0,
    )
    assert score == 100.0
    assert breakdown["breakout"] == 25.0
    assert breakdown["vwap"] == 15.0
    assert breakdown["rel_volume"] == 20.0
    assert breakdown["momentum"] == 15.0
    assert breakdown["ema_stack"] == 10.0
    assert breakdown["volatility_expansion"] == 10.0
    assert breakdown["catalyst"] == 8.0
    assert "breakout" in positives


def test_compose_score_zero():
    score, breakdown, positives = _compose_score(
        breakout_any=False,
        vwap_ok=False,
        rel_volume_spike=False,
        momentum_confirmed=False,
        ema_stack=False,
        volatility_expansion=False,
        catalyst_score=0.0,
    )
    assert score == 0.0
    assert all(v == 0.0 for v in breakdown.values())
    assert positives == []

import pandas as pd

from me_screener_alerts import detect_screener_changes


def _row(
    ticker,
    state,
    decision,
    rank,
    score=70,
):
    return {
        "ticker": ticker,
        "trade_date": "2026-10-06",
        "company_name": f"Name {ticker}",
        "second_wave_state": state,
        "sw_decision": decision,
        "screen_rank": rank,
        "sw_score": score,
        "hist_edge_score": 60,
        "fcf_engine_score": 70,
        "mlp_c_core": 1.10,
        "me_velocity_pct": -1.0,
        "me_acceleration_pct": 0.3,
        "screen_reason": "test",
    }


def test_first_run_is_baseline():
    current = pd.DataFrame(
        [_row("1111", "RE-WATCH READY", "READY", 3)]
    )
    out = detect_screener_changes(
        pd.DataFrame(),
        current,
        baseline_if_no_previous=True,
    )
    assert out.empty


def test_detects_ready_progression():
    previous = pd.DataFrame(
        [_row("1111", "RE-WATCH EARLY", "WATCH", 12)]
    )
    current = pd.DataFrame(
        [_row("1111", "RE-WATCH READY", "READY", 7)]
    )
    out = detect_screener_changes(previous, current)

    assert "R-READY" in set(out["event_type"])
    assert "TOP10 NEW" in set(out["event_type"])


def test_detects_priority_watch_without_state_change():
    previous = pd.DataFrame(
        [_row("2222", "EXP. DECELERATING", "WATCH", 8, 65)]
    )
    current = pd.DataFrame(
        [_row("2222", "EXP. DECELERATING", "PRIORITY WATCH", 4, 76)]
    )
    out = detect_screener_changes(previous, current)

    assert "PRIORITY WATCH" in set(out["event_type"])


def test_no_repeated_alert_when_nothing_changed():
    previous = pd.DataFrame(
        [_row("3333", "RE-WATCH READY", "READY", 5)]
    )
    current = previous.copy()

    out = detect_screener_changes(previous, current)
    assert out.empty


def test_detects_reexp_confirmation():
    previous = pd.DataFrame(
        [_row("4444", "RE-WATCH READY", "READY", 4)]
    )
    current = pd.DataFrame(
        [_row("4444", "RE-EXP", "ACTIVE", 2, 84)]
    )

    out = detect_screener_changes(previous, current)
    assert "RE-EXP CONFIRMED" in set(out["event_type"])

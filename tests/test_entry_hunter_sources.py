import pandas as pd

from entry_hunter_sources import (
    combine_entry_candidates,
    select_me_entry_candidates,
)


def _me_row(
    ticker="6857",
    state="RE-WATCH READY",
    decision="READY",
    score=75,
    rank=3,
):
    return {
        "ticker": ticker,
        "company_name": f"Name {ticker}",
        "trade_date": "2026-10-05",
        "second_wave_state": state,
        "sw_decision": decision,
        "sw_score": score,
        "hist_edge_score": 62,
        "fcf_engine_score": 78,
        "screen_rank": rank,
    }


def test_select_me_entry_candidates_accepts_ready():
    frame = pd.DataFrame([_me_row()])
    out = select_me_entry_candidates(
        frame,
        as_of="2026-10-06",
        max_calendar_days=4,
        limit=5,
    )

    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "6857"
    assert out.iloc[0]["source"] == "ME HUNTER"
    assert "R-READY" in out.iloc[0]["alert_tier"]


def test_select_me_entry_candidates_rejects_plain_watch():
    frame = pd.DataFrame(
        [
            _me_row(
                ticker="4063",
                state="EXP. DECELERATING",
                decision="WATCH",
                score=40,
            )
        ]
    )
    out = select_me_entry_candidates(
        frame,
        as_of="2026-10-06",
    )
    assert out.empty


def test_combine_sources_promotes_confluence():
    short = pd.DataFrame(
        [
            {
                "ticker": "6857",
                "name": "Advantest",
                "alert_date": "2026-10-05",
                "alert_tier": "A+",
                "condition_version": "SC-v1",
                "alert_score": 82,
                "match_strength": 90,
                "alert_reason": "Short Cover active",
            }
        ]
    )
    me = select_me_entry_candidates(
        pd.DataFrame([_me_row()]),
        as_of="2026-10-06",
    )

    out = combine_entry_candidates(short, me, limit=8)

    assert len(out) == 1
    assert out.iloc[0]["source"] == "SHORT+ME"
    assert out.iloc[0]["alert_tier"] == "🔥 CONFLUENCE"
    assert float(out.iloc[0]["source_score"]) >= 82

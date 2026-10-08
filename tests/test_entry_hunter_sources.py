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


def test_source_adjustment_is_zero_with_too_few_samples():
    from entry_hunter_sources import build_source_adjustments

    summary = pd.DataFrame(
        [
            {
                "source": "SHORT+ME",
                "sample_5d": 3,
                "win_5d": 80,
                "avg_5d": 5.0,
                "avg_mfe_10d": 10.0,
                "avg_mae_10d": -2.0,
            }
        ]
    )
    out = build_source_adjustments(summary)
    assert out["SHORT+ME"]["bonus"] == 0.0
    assert out["SHORT+ME"]["confidence"] == "DATA BUILDING"


def test_source_adjustment_rewards_proven_confluence_conservatively():
    from entry_hunter_sources import build_source_adjustments

    summary = pd.DataFrame(
        [
            {
                "source": "SHORT+ME",
                "sample_5d": 20,
                "win_5d": 70,
                "avg_5d": 4.0,
                "avg_mfe_10d": 9.0,
                "avg_mae_10d": -3.0,
            }
        ]
    )
    out = build_source_adjustments(summary)
    assert 0.0 < float(out["SHORT+ME"]["bonus"]) <= 8.0
    assert out["SHORT+ME"]["confidence"] == "ADAPTIVE"


def test_combine_sources_uses_adaptive_bonus():
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
    summary = pd.DataFrame(
        [
            {
                "source": "SHORT+ME",
                "sample_5d": 20,
                "win_5d": 70,
                "avg_5d": 4.0,
                "avg_mfe_10d": 9.0,
                "avg_mae_10d": -3.0,
            }
        ]
    )

    out = combine_entry_candidates(
        short,
        me,
        limit=8,
        source_summary=summary,
    )

    assert len(out) == 1
    assert out.iloc[0]["source"] == "SHORT+ME"
    assert float(out.iloc[0]["adaptive_bonus"]) > 0
    assert out.iloc[0]["adaptive_confidence"] == "ADAPTIVE"


def test_signal_adjustment_requires_samples():
    from entry_hunter_sources import build_signal_adjustments

    summary = pd.DataFrame(
        [
            {
                "signal_key": "ME|R-READY",
                "sample_5d": 4,
                "win_5d": 80,
                "avg_5d": 5.0,
                "avg_mfe_10d": 10.0,
                "avg_mae_10d": -2.0,
            }
        ]
    )
    out = build_signal_adjustments(summary)
    assert out["ME|R-READY"]["bonus"] == 0.0


def test_signal_adjustment_is_capped_at_four_points():
    from entry_hunter_sources import build_signal_adjustments

    summary = pd.DataFrame(
        [
            {
                "signal_key": "ME|RE-EXP",
                "sample_5d": 30,
                "win_5d": 90,
                "avg_5d": 12.0,
                "avg_mfe_10d": 20.0,
                "avg_mae_10d": -2.0,
            }
        ]
    )
    out = build_signal_adjustments(summary)
    assert 0 < float(out["ME|RE-EXP"]["bonus"]) <= 4.0


def test_trait_classifier_builds_market_size_vol_buckets():
    from entry_hunter_sources import classify_candidate_traits

    frame = pd.DataFrame(
        [{
            "ticker": "6857",
            "market_name": "Prime Market",
            "market_cap": 3_000_000_000_000,
            "realized_vol20_pct": 50.0,
        }]
    )
    out = classify_candidate_traits(frame)
    assert out.iloc[0]["trait_market"] == "MARKET|PRIME"
    assert out.iloc[0]["trait_size"] == "SIZE|MEGA"
    assert out.iloc[0]["trait_vol"] == "VOL|HIGH"


def test_trait_adjustment_waits_for_eight_samples():
    from entry_hunter_sources import build_trait_adjustments

    summary = pd.DataFrame(
        [{
            "trait_key": "MARKET|GROWTH",
            "sample_5d": 7,
            "win_5d": 80,
            "avg_5d": 5.0,
        }]
    )
    out = build_trait_adjustments(summary)
    assert out["MARKET|GROWTH"]["bonus"] == 0.0


def test_trait_adjustment_is_tiny_and_capped():
    from entry_hunter_sources import build_trait_adjustments

    summary = pd.DataFrame(
        [{
            "trait_key": "VOL|HIGH",
            "sample_5d": 40,
            "win_5d": 80,
            "avg_5d": 8.0,
        }]
    )
    out = build_trait_adjustments(summary)
    assert 0 < float(out["VOL|HIGH"]["bonus"]) <= 1.0


def test_fast_feedback_adjustment_maps_dimension_and_key():
    from entry_hunter_sources import build_fast_feedback_adjustments

    summary = pd.DataFrame(
        [
            {
                "dimension": "SOURCE",
                "key": "SHORT+ME",
                "sample_0d": 20,
                "fast_bonus": 1.2,
                "confidence": "WARMING",
            },
            {
                "dimension": "SETUP",
                "key": "CONFLUENCE|R-READY|EARLY",
                "sample_0d": 20,
                "fast_bonus": 0.6,
                "confidence": "WARMING",
            },
        ]
    )
    out = build_fast_feedback_adjustments(summary)
    assert float(out["SOURCE:SHORT+ME"]["bonus"]) == 1.2
    assert float(out["SETUP:CONFLUENCE|R-READY|EARLY"]["bonus"]) == 0.6


def test_combine_sources_applies_small_fast_feedback_bonus():
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
    fast = pd.DataFrame(
        [
            {
                "dimension": "SOURCE",
                "key": "SHORT+ME",
                "sample_0d": 20,
                "fast_bonus": 1.2,
                "confidence": "WARMING",
            },
            {
                "dimension": "SETUP",
                "key": "CONFLUENCE|R-READY|OTHER",
                "sample_0d": 20,
                "fast_bonus": 0.6,
                "confidence": "WARMING",
            },
        ]
    )

    out = combine_entry_candidates(
        short,
        me,
        limit=8,
        fast_feedback_summary=fast,
    )

    assert len(out) == 1
    assert 0 < float(out.iloc[0]["fast_bonus"]) <= 2.5
    assert out.iloc[0]["fast_confidence"] == "WARMING"


def test_me_watch_candidate_does_not_promote_to_entry_hunter():
    from entry_hunter_sources import (
        select_me_entry_candidates,
        select_me_watch_candidates,
    )

    row = {
        "ticker": "2670",
        "company_name": "ABC-Mart",
        "trade_date": "2026-10-08",
        "second_wave_state": "EXP. DECELERATING",
        "sw_decision": "WATCH",
        "candidate_type": "SECOND WAVE",
        "sw_score": 45.1,
        "hist_edge_score": 45.0,
        "fcf_engine_score": None,
        "screen_rank": 1,
        "re_route": "LOCKED",
    }
    frame = pd.DataFrame([row])

    entry = select_me_entry_candidates(
        frame,
        as_of="2026-10-09",
    )
    watch = select_me_watch_candidates(
        frame,
        as_of="2026-10-09",
    )

    assert entry.empty
    assert len(watch) == 1
    assert watch.iloc[0]["ticker"] == "2670"
    assert watch.iloc[0]["signal_key"] == "ME|WATCH"
    assert watch.iloc[0]["alert_tier"] == "🟡 ME WATCH"

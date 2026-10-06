import pandas as pd

from daily_command_center import (
    build_command_center_handoff,
    build_daily_command_center,
    tradingview_url,
)

# Dashboard command-center regression tests.


def test_command_center_prefers_live_entry_status():
    status = {
        "run_at": "2026-10-06T09:30:00+09:00",
        "rows": [
            {
                "ticker": "6857",
                "name": "Advantest",
                "opportunity_score": 91,
                "opportunity_rating": "A+",
                "opportunity_action": "ENTRY PRIORITY",
                "source": "SHORT+ME",
                "signal_key": "CONFLUENCE|R-READY|EARLY",
                "status": "🟢 ENTRY READY",
                "opportunity_coverage": 100,
                "opportunity_reason": "SHORT+ME / 寄り後READY",
            },
            {
                "ticker": "6146",
                "name": "Disco",
                "opportunity_score": 82,
                "opportunity_rating": "A",
                "opportunity_action": "ENTRY READY",
                "source": "ME HUNTER",
                "signal_key": "ME|R-READY",
                "status": "🟢 ENTRY READY",
                "opportunity_coverage": 100,
                "opportunity_reason": "ME HUNTER / 寄り後READY",
            },
        ]
    }
    out = build_daily_command_center(status, pd.DataFrame(), as_of="2026-10-06 09:35:00", limit=3)
    assert len(out) == 2
    assert out.iloc[0]["ticker"] == "6857"
    assert out.iloc[0]["mode"] == "LIVE"


def test_command_center_falls_back_to_me_premarket():
    me = pd.DataFrame(
        [
            {
                "ticker": "6857",
                "company_name": "Advantest",
                "trade_date": "2026-10-05",
                "second_wave_state": "RE-WATCH READY",
                "sw_decision": "READY",
                "sw_score": 80,
                "hist_edge_score": 65,
                "fcf_engine_score": 75,
                "screen_rank": 1,
            }
        ]
    )
    out = build_daily_command_center({}, me, as_of="2026-10-06", limit=3)
    assert len(out) == 1
    assert out.iloc[0]["ticker"] == "6857"
    assert out.iloc[0]["mode"] == "PRE-MARKET"
    assert int(out.iloc[0]["coverage"]) == 55


def test_stale_live_status_falls_back_to_premarket():
    status = {
        "run_at": "2026-10-05T10:00:00+09:00",
        "rows": [{
            "ticker": "6146",
            "name": "Disco",
            "opportunity_score": 95,
            "opportunity_rating": "S",
            "opportunity_action": "ENTRY PRIORITY",
            "source": "SHORT+ME",
            "signal_key": "CONFLUENCE|RE-EXP|CONFIRMED",
            "status": "🟢 ENTRY READY",
            "opportunity_coverage": 100,
            "opportunity_reason": "old",
        }],
    }
    me = pd.DataFrame([{
        "ticker": "6857",
        "company_name": "Advantest",
        "trade_date": "2026-10-05",
        "second_wave_state": "RE-WATCH READY",
        "sw_decision": "READY",
        "sw_score": 80,
        "hist_edge_score": 65,
        "fcf_engine_score": 75,
        "screen_rank": 1,
    }])
    out = build_daily_command_center(status, me, as_of="2026-10-06", limit=3)
    assert out.iloc[0]["ticker"] == "6857"
    assert out.iloc[0]["mode"] == "PRE-MARKET"


def test_rank_change_uses_previous_day_snapshot():
    from daily_command_center import update_command_center_history

    previous = pd.DataFrame([
        {
            "rank": 2,
            "previous_rank": None,
            "rank_change": None,
            "rank_trend": "NEW",
            "ticker": "6857",
            "name": "Advantest",
            "opportunity_score": 82,
            "opportunity_rating": "A",
            "opportunity_action": "ENTRY READY",
            "source": "ME HUNTER",
            "signal_key": "ME|R-READY",
            "entry_status": "🟢 ENTRY READY",
            "coverage": 100,
            "decision_card": "ME候補｜R-READY｜寄り後READY",
            "reason": "",
            "mode": "LIVE",
        }
    ])
    history = update_command_center_history(
        pd.DataFrame(),
        previous,
        snapshot_at="2026-10-05 10:00:00",
    )

    current_status = {
        "run_at": "2026-10-06T09:30:00+09:00",
        "rows": [
            {
                "ticker": "6857",
                "name": "Advantest",
                "opportunity_score": 91,
                "opportunity_rating": "A+",
                "opportunity_action": "ENTRY PRIORITY",
                "source": "SHORT+ME",
                "signal_key": "CONFLUENCE|R-READY|EARLY",
                "status": "🟢 ENTRY READY",
                "opportunity_coverage": 100,
                "opportunity_reason": "SHORT+ME / 寄り後READY",
            }
        ],
    }
    out = build_daily_command_center(
        current_status,
        pd.DataFrame(),
        as_of="2026-10-06 09:35:00",
        limit=3,
        history=history,
    )
    assert out.iloc[0]["previous_rank"] == 2
    assert out.iloc[0]["rank_trend"] == "↑1"
    assert "SHORT+ME合流" in out.iloc[0]["decision_card"]


def test_tradingview_url_uses_tse_symbol():
    assert tradingview_url("6857") == "https://www.tradingview.com/chart/?symbol=TSE%3A6857"


def test_command_center_handoff_keeps_opportunity_context():
    row = {
        "ticker": "6857",
        "name": "Advantest",
        "source": "SHORT+ME",
        "signal_key": "CONFLUENCE|R-READY|EARLY",
        "opportunity_score": 91,
        "opportunity_rating": "A+",
        "opportunity_action": "ENTRY PRIORITY",
        "reason": "SHORT+ME / 寄り後READY",
        "coverage": 100,
        "decision_card": "SHORT+ME合流｜R-READY×EARLY｜寄り後READY",
        "mode": "LIVE",
    }
    out = build_command_center_handoff(row)
    assert out["ticker"] == "6857"
    assert out["opportunity_score"] == 91
    assert out["opportunity_rating"] == "A+"
    assert out["decision_card"].startswith("SHORT+ME合流")

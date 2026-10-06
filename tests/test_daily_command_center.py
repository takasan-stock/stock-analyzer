import pandas as pd

from daily_command_center import build_daily_command_center


def test_command_center_prefers_live_entry_status():
    status = {
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
    out = build_daily_command_center(status, pd.DataFrame(), limit=3)
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

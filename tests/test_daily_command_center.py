import pandas as pd

from daily_command_center import (
    build_command_center_handoff,
    build_command_center_session,
    build_daily_command_center,
    build_score_breakdown_rows,
    execution_handoff_allowed,
    resolve_command_center_mode,
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
                "source_bonus": 3.2,
                "setup_bonus": 1.4,
                "trait_bonus": 0.6,
                "fast_bonus": 0.8,
                "adaptive_total": 6.0,
                "adaptive_breakdown": "Source +3.2｜Setup +1.4｜Trait +0.6｜Fast0D +0.8",
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


def test_command_center_mode_uses_jst_clock():
    assert resolve_command_center_mode("2026-10-06 08:30:00+09:00") == "PRE-MARKET"
    assert resolve_command_center_mode("2026-10-06 10:00:00+09:00") == "LIVE"
    assert resolve_command_center_mode("2026-10-06 16:00:00+09:00") == "AFTER CLOSE"
    assert resolve_command_center_mode("2026-10-10 10:00:00+09:00") == "AFTER CLOSE"


def test_session_premarket_ignores_same_day_live_status():
    status = {
        "run_at": "2026-10-06T08:30:00+09:00",
        "rows": [{
            "ticker": "6146",
            "name": "Disco",
            "opportunity_score": 99,
            "opportunity_rating": "S",
            "opportunity_action": "ENTRY PRIORITY",
            "source": "SHORT+ME",
            "signal_key": "CONFLUENCE|RE-EXP|CONFIRMED",
            "status": "🟢 ENTRY READY",
            "opportunity_coverage": 100,
            "opportunity_reason": "same-day test row",
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
    session = build_command_center_session(
        status,
        me,
        as_of="2026-10-06 08:30:00+09:00",
    )
    assert session["mode"] == "PRE-MARKET"
    assert session["primary"].iloc[0]["ticker"] == "6857"


def test_session_live_prefers_entry_hunter():
    status = {
        "run_at": "2026-10-06T10:00:00+09:00",
        "rows": [{
            "ticker": "6146",
            "name": "Disco",
            "opportunity_score": 91,
            "opportunity_rating": "A+",
            "opportunity_action": "ENTRY PRIORITY",
            "source": "SHORT+ME",
            "signal_key": "CONFLUENCE|R-READY|EARLY",
            "status": "🟢 ENTRY READY",
            "opportunity_coverage": 100,
            "opportunity_reason": "live",
        }],
    }
    session = build_command_center_session(
        status,
        pd.DataFrame(),
        as_of="2026-10-06 10:05:00+09:00",
    )
    assert session["mode"] == "LIVE"
    assert session["primary"].iloc[0]["ticker"] == "6146"


def test_session_after_close_separates_today_and_next_session():
    status = {
        "run_at": "2026-10-06T14:55:00+09:00",
        "rows": [{
            "ticker": "6146",
            "name": "Disco",
            "opportunity_score": 88,
            "opportunity_rating": "A+",
            "opportunity_action": "ENTRY READY",
            "source": "SHORT+ME",
            "signal_key": "CONFLUENCE|R-READY|EARLY",
            "status": "🟢 ENTRY READY",
            "opportunity_coverage": 100,
            "opportunity_reason": "today result",
        }],
    }
    me = pd.DataFrame([{
        "ticker": "6857",
        "company_name": "Advantest",
        "trade_date": "2026-10-06",
        "second_wave_state": "RE-WATCH READY",
        "sw_decision": "READY",
        "sw_score": 82,
        "hist_edge_score": 66,
        "fcf_engine_score": 76,
        "screen_rank": 1,
    }])

    session = build_command_center_session(
        status,
        me,
        as_of="2026-10-06 18:00:00+09:00",
    )
    assert session["mode"] == "AFTER CLOSE"
    assert session["primary"].iloc[0]["ticker"] == "6146"
    assert session["secondary"].iloc[0]["ticker"] == "6857"
    assert session["secondary"].iloc[0]["mode"] == "NEXT SESSION"
    assert session["me_status"] == "UPDATED"


def test_command_center_keeps_adaptive_breakdown():
    status = {
        "run_at": "2026-10-06T09:30:00+09:00",
        "rows": [{
            "ticker": "6857",
            "name": "Advantest",
            "opportunity_score": 91,
            "opportunity_rating": "A+",
            "opportunity_action": "ENTRY PRIORITY",
            "source": "SHORT+ME",
            "signal_key": "CONFLUENCE|R-READY|EARLY",
            "status": "🟢 ENTRY READY",
            "opportunity_coverage": 100,
            "opportunity_reason": "SHORT+ME / 寄り後READY / Fast0D+",
            "source_bonus": 3.2,
            "setup_bonus": 1.4,
            "trait_bonus": 0.6,
            "fast_bonus": 0.8,
            "adaptive_total": 6.0,
            "adaptive_breakdown": "Source +3.2｜Setup +1.4｜Trait +0.6｜Fast0D +0.8",
        }],
    }
    out = build_daily_command_center(
        status,
        pd.DataFrame(),
        as_of="2026-10-06 09:35:00",
        limit=3,
    )
    row = out.iloc[0]
    assert float(row["fast_bonus"]) == 0.8
    assert float(row["adaptive_total"]) == 6.0
    assert "Fast0D +0.8" in row["adaptive_breakdown"]

    handoff = build_command_center_handoff(row)
    assert float(handoff["fast_bonus"]) == 0.8
    assert "Source +3.2" in handoff["adaptive_breakdown"]


def test_score_breakdown_rows_normalize_each_component():
    row = {
        "source_bonus": 4.0,
        "setup_bonus": -2.0,
        "trait_bonus": 1.25,
        "fast_bonus": 0.0,
    }
    out = build_score_breakdown_rows(row)
    by_label = {item["label"]: item for item in out}

    assert by_label["Source"]["percent"] == 50.0
    assert by_label["Source"]["direction"] == "positive"

    assert by_label["Setup"]["percent"] == 50.0
    assert by_label["Setup"]["direction"] == "negative"

    assert by_label["Trait"]["percent"] == 50.0
    assert by_label["Trait"]["direction"] == "positive"

    assert by_label["Fast0D"]["percent"] == 0.0
    assert by_label["Fast0D"]["direction"] == "neutral"


def test_score_breakdown_rows_clip_to_component_caps():
    row = {
        "source_bonus": 99,
        "setup_bonus": -99,
        "trait_bonus": 99,
        "fast_bonus": -99,
    }
    out = build_score_breakdown_rows(row)
    by_label = {item["label"]: item for item in out}

    assert by_label["Source"]["value"] == 8.0
    assert by_label["Setup"]["value"] == -4.0
    assert by_label["Trait"]["value"] == 2.5
    assert by_label["Fast0D"]["value"] == -2.5
    assert all(item["percent"] == 100.0 for item in out)


def test_command_center_shows_me_watch_without_promoting_entry():
    me = pd.DataFrame(
        [{
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
        }]
    )
    out = build_daily_command_center(
        {},
        me,
        as_of="2026-10-09 08:30:00+09:00",
        limit=3,
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["ticker"] == "2670"
    assert row["opportunity_action"] == "WATCH ONLY"
    assert row["entry_status"] == "🟡 ME WATCH"
    assert "再加速待ち" in row["decision_card"]


def test_watch_only_blocks_execution_handoff():
    row = {
        "ticker": "2670",
        "opportunity_action": "WATCH ONLY",
        "entry_status": "🟡 ME WATCH",
        "signal_key": "ME|WATCH",
    }
    assert execution_handoff_allowed(row) is False


def test_ready_me_allows_execution_handoff():
    row = {
        "ticker": "2670",
        "opportunity_action": "PRE-MARKET WATCH",
        "entry_status": "⏳ PRE-MARKET",
        "signal_key": "ME|R-READY",
    }
    assert execution_handoff_allowed(row) is True


def test_me_watch_promotes_when_state_becomes_ready():
    me = pd.DataFrame(
        [{
            "ticker": "2670",
            "company_name": "ABC-Mart",
            "trade_date": "2026-10-08",
            "second_wave_state": "RE-WATCH READY",
            "sw_decision": "READY",
            "candidate_type": "SECOND WAVE",
            "sw_score": 72.0,
            "hist_edge_score": 55.0,
            "fcf_engine_score": 60.0,
            "screen_rank": 1,
            "re_route": "OPEN",
        }]
    )
    out = build_daily_command_center(
        {},
        me,
        as_of="2026-10-09 08:30:00+09:00",
        limit=3,
    )

    assert len(out) == 1
    row = out.iloc[0]
    assert row["ticker"] == "2670"
    assert row["signal_key"] == "ME|R-READY"
    assert row["opportunity_action"] == "PRE-MARKET WATCH"
    assert row["entry_status"] == "⏳ PRE-MARKET"
    assert execution_handoff_allowed(row) is True

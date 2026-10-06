import pandas as pd

# Prospective source-comparison tests: ME vs Short Cover vs confluence.

from entry_source_performance import (
    build_ready_performance,
    summarize_source_performance,
    summarize_signal_performance,
    update_candidate_history,
)


def test_candidate_history_tracks_ready_by_source():
    candidates = pd.DataFrame(
        [
            {
                "ticker": "6857",
                "name": "Advantest",
                "source": "ME HUNTER",
                "source_detail": "R-READY",
                "source_score": 90,
                "source_rank": 1,
            }
        ]
    )
    status_rows = [
        {
            "ticker": "6857",
            "status": "🟢 ENTRY READY",
            "score": 82,
        }
    ]

    out = update_candidate_history(
        pd.DataFrame(),
        candidates,
        status_rows,
        observed_at="2026-10-06 09:25:00",
    )

    assert len(out) == 1
    assert out.iloc[0]["source"] == "ME HUNTER"
    assert bool(out.iloc[0]["ready_detected"])
    assert float(out.iloc[0]["best_entry_score"]) == 82.0


def test_ready_performance_uses_actual_entry_price():
    notifications = pd.DataFrame(
        [
            {
                "market_date": "2026-10-01",
                "ticker": "6857",
                "name": "Advantest",
                "source": "ME HUNTER",
                "source_detail": "R-READY",
                "entry_status": "🟢 ENTRY READY",
                "entry_score": 80,
                "current_price": 100.0,
                "first_detected_at": "2026-10-01 09:25:00",
            }
        ]
    )

    dates = pd.bdate_range("2026-10-01", periods=10)
    price = pd.DataFrame(
        {
            "Open": [100] * 10,
            "High": [102, 104, 105, 106, 108, 110, 111, 112, 113, 114],
            "Low": [98, 99, 100, 101, 101, 102, 103, 104, 105, 106],
            "Close": [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        },
        index=dates,
    )

    out = build_ready_performance(
        notifications,
        {"6857": price},
        updated_at="2026-10-15",
    )

    assert len(out) == 1
    assert round(float(out.iloc[0]["ret_5d"]), 6) == 5.0
    assert round(float(out.iloc[0]["ret_10d"]), 6) == 10.0
    assert round(float(out.iloc[0]["mfe_10d"]), 6) == 14.0
    assert round(float(out.iloc[0]["mae_10d"]), 6) == -2.0


def test_summary_compares_three_sources():
    candidates = pd.DataFrame(
        [
            {
                "monitor_date": "2026-10-01",
                "ticker": "1111",
                "name": "A",
                "source": "ME HUNTER",
                "source_detail": "",
                "source_score": 80,
                "source_rank": 1,
                "first_seen_at": "2026-10-01 09:20:00",
                "last_seen_at": "2026-10-01 09:30:00",
                "last_status": "🟢 ENTRY READY",
                "best_entry_score": 80,
                "ready_detected": True,
                "ready_detected_at": "2026-10-01 09:25:00",
            },
            {
                "monitor_date": "2026-10-01",
                "ticker": "2222",
                "name": "B",
                "source": "SHORT COVER",
                "source_detail": "",
                "source_score": 75,
                "source_rank": 2,
                "first_seen_at": "2026-10-01 09:20:00",
                "last_seen_at": "2026-10-01 09:30:00",
                "last_status": "🟡 WAIT",
                "best_entry_score": 50,
                "ready_detected": False,
                "ready_detected_at": None,
            },
            {
                "monitor_date": "2026-10-01",
                "ticker": "3333",
                "name": "C",
                "source": "SHORT+ME",
                "source_detail": "",
                "source_score": 95,
                "source_rank": 1,
                "first_seen_at": "2026-10-01 09:20:00",
                "last_seen_at": "2026-10-01 09:30:00",
                "last_status": "🟢 ENTRY READY",
                "best_entry_score": 90,
                "ready_detected": True,
                "ready_detected_at": "2026-10-01 09:22:00",
            },
        ]
    )

    performance = pd.DataFrame(
        [
            {
                "market_date": "2026-10-01",
                "ticker": "1111",
                "name": "A",
                "source": "ME HUNTER",
                "source_detail": "",
                "entry_price": 100,
                "entry_score": 80,
                "first_detected_at": "2026-10-01 09:25:00",
                "ret_1d": 1,
                "ret_3d": 2,
                "ret_5d": 3,
                "ret_10d": 4,
                "mfe_10d": 8,
                "mae_10d": -2,
                "outcome_status": "✅ 10D COMPLETE",
                "last_updated": "2026-10-15",
            },
            {
                "market_date": "2026-10-01",
                "ticker": "3333",
                "name": "C",
                "source": "SHORT+ME",
                "source_detail": "",
                "entry_price": 100,
                "entry_score": 90,
                "first_detected_at": "2026-10-01 09:22:00",
                "ret_1d": 2,
                "ret_3d": 4,
                "ret_5d": 6,
                "ret_10d": 9,
                "mfe_10d": 12,
                "mae_10d": -1,
                "outcome_status": "✅ 10D COMPLETE",
                "last_updated": "2026-10-15",
            },
        ]
    )

    out = summarize_source_performance(candidates, performance)

    by_source = out.set_index("source")
    assert float(by_source.loc["ME HUNTER", "entry_ready_rate"]) == 100.0
    assert float(by_source.loc["SHORT COVER", "entry_ready_rate"]) == 0.0
    assert float(by_source.loc["SHORT+ME", "avg_5d"]) == 6.0


def test_signal_summary_separates_concrete_setups():
    candidates = pd.DataFrame(
        [
            {
                "monitor_date": "2026-10-01",
                "ticker": "1111",
                "name": "A",
                "source": "ME HUNTER",
                "source_detail": "",
                "source_score": 80,
                "source_rank": 1,
                "signal_key": "ME|R-READY",
                "first_seen_at": "2026-10-01 09:20:00",
                "last_seen_at": "2026-10-01 09:30:00",
                "last_status": "🟢 ENTRY READY",
                "best_entry_score": 80,
                "ready_detected": True,
                "ready_detected_at": "2026-10-01 09:25:00",
            }
        ]
    )
    performance = pd.DataFrame(
        [
            {
                "market_date": "2026-10-01",
                "ticker": "1111",
                "name": "A",
                "source": "ME HUNTER",
                "source_detail": "",
                "signal_key": "ME|R-READY",
                "entry_price": 100,
                "entry_score": 80,
                "first_detected_at": "2026-10-01 09:25:00",
                "ret_1d": 1,
                "ret_3d": 2,
                "ret_5d": 4,
                "ret_10d": 6,
                "mfe_10d": 9,
                "mae_10d": -2,
                "outcome_status": "✅ 10D COMPLETE",
                "last_updated": "2026-10-15",
            }
        ]
    )
    out = summarize_signal_performance(candidates, performance)
    assert len(out) == 1
    assert out.iloc[0]["signal_key"] == "ME|R-READY"
    assert float(out.iloc[0]["avg_5d"]) == 4.0

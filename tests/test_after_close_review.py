import pandas as pd

from after_close_review import (
    build_after_close_review,
    summarize_after_close_feedback,
    update_after_close_history,
)


def _notifications():
    return pd.DataFrame(
        [
            {
                "market_date": "2026-10-06",
                "ticker": "6857",
                "name": "Advantest",
                "entry_status": "🟢 ENTRY READY",
                "current_price": 100.0,
                "first_detected_at": "2026-10-06 09:25:00",
                "source": "SHORT+ME",
                "signal_key": "CONFLUENCE|R-READY|EARLY",
            },
            {
                "market_date": "2026-10-06",
                "ticker": "6857",
                "name": "Advantest",
                "entry_status": "🟢 ENTRY CONFIRMED",
                "current_price": 103.0,
                "first_detected_at": "2026-10-06 10:00:00",
                "source": "SHORT+ME",
                "signal_key": "CONFLUENCE|R-READY|EARLY",
            },
            {
                "market_date": "2026-10-06",
                "ticker": "6146",
                "name": "Disco",
                "entry_status": "🟢 ENTRY READY",
                "current_price": 200.0,
                "first_detected_at": "2026-10-06 09:30:00",
                "source": "ME HUNTER",
                "signal_key": "ME|R-READY",
            },
            {
                "market_date": "2026-10-06",
                "ticker": "6146",
                "name": "Disco",
                "entry_status": "🟡 WEAKENING",
                "current_price": 196.0,
                "first_detected_at": "2026-10-06 10:20:00",
                "source": "ME HUNTER",
                "signal_key": "ME|R-READY",
            },
        ]
    )


def test_after_close_review_uses_ready_price_and_daily_bar():
    dates = pd.to_datetime(["2026-10-06"])
    frames = {
        "6857": pd.DataFrame(
            {
                "Open": [101.0],
                "High": [108.0],
                "Low": [98.0],
                "Close": [105.0],
            },
            index=dates,
        ),
        "6146": pd.DataFrame(
            {
                "Open": [199.0],
                "High": [204.0],
                "Low": [190.0],
                "Close": [194.0],
            },
            index=dates,
        ),
    }

    out = build_after_close_review(
        _notifications(),
        frames,
        review_date="2026-10-06",
        top_tickers=["6857", "6146"],
    )

    assert out["ready_count"] == 2
    assert out["confirmed_count"] == 1
    assert out["weakening_count"] == 1
    assert out["tracked_entries"] == 2
    assert out["price_coverage"] == 2
    assert round(float(out["avg_close_return_pct"]), 4) == 1.0
    assert round(float(out["top3_avg_return_pct"]), 4) == 1.0

    rows = {row["ticker"]: row for row in out["rows"]}
    assert round(float(rows["6857"]["close_return_pct"]), 4) == 5.0
    assert round(float(rows["6857"]["mfe_pct"]), 4) == 8.0
    assert round(float(rows["6857"]["mae_pct"]), 4) == -2.0
    assert rows["6857"]["carryover"] is True

    assert round(float(rows["6146"]["close_return_pct"]), 4) == -3.0
    assert rows["6146"]["carryover"] is False


def test_after_close_review_handles_no_entries():
    out = build_after_close_review(
        pd.DataFrame(),
        {},
        review_date="2026-10-06",
    )
    assert out["ready_count"] == 0
    assert out["tracked_entries"] == 0
    assert out["avg_close_return_pct"] is None


def test_after_close_feedback_waits_for_minimum_samples():
    rows = pd.DataFrame(
        [
            {
                "review_date": f"2026-10-{day:02d}",
                "ticker": f"68{day:02d}",
                "name": "A",
                "source": "ME HUNTER",
                "signal_key": "ME|R-READY",
                "trait_market": "MARKET|PRIME",
                "trait_size": "SIZE|LARGE",
                "trait_vol": "VOL|MID",
                "entry_price": 100,
                "close_price": 103,
                "close_return_pct": 3.0,
                "mfe_pct": 5.0,
                "mae_pct": -1.5,
                "latest_status": "🟢 ENTRY READY",
                "carryover": True,
            }
            for day in range(1, 8)
        ]
    )
    history = update_after_close_history(pd.DataFrame(), rows)
    summary = summarize_after_close_feedback(history)
    me_row = summary[
        (summary["dimension"] == "SOURCE")
        & (summary["key"] == "ME HUNTER")
    ].iloc[0]
    assert int(me_row["sample_0d"]) == 7
    assert float(me_row["fast_bonus"]) == 0.0
    assert me_row["confidence"] == "DATA BUILDING"


def test_after_close_feedback_is_small_and_positive_after_enough_samples():
    rows = pd.DataFrame(
        [
            {
                "review_date": f"2026-09-{day:02d}",
                "ticker": f"68{day:02d}",
                "name": "A",
                "source": "SHORT+ME",
                "signal_key": "CONFLUENCE|R-READY|EARLY",
                "trait_market": "MARKET|PRIME",
                "trait_size": "SIZE|LARGE",
                "trait_vol": "VOL|HIGH",
                "entry_price": 100,
                "close_price": 104,
                "close_return_pct": 4.0,
                "mfe_pct": 7.0,
                "mae_pct": -2.0,
                "latest_status": "🟢 ENTRY CONFIRMED",
                "carryover": True,
            }
            for day in range(1, 21)
        ]
    )
    history = update_after_close_history(pd.DataFrame(), rows)
    summary = summarize_after_close_feedback(history)

    source = summary[
        (summary["dimension"] == "SOURCE")
        & (summary["key"] == "SHORT+ME")
    ].iloc[0]
    setup = summary[
        (summary["dimension"] == "SETUP")
        & (summary["key"] == "CONFLUENCE|R-READY|EARLY")
    ].iloc[0]

    assert 0 < float(source["fast_bonus"]) <= 1.5
    assert 0 < float(setup["fast_bonus"]) <= 0.75
    assert source["confidence"] in {"WARMING", "ADAPTIVE"}

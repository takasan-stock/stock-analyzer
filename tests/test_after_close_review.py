import pandas as pd

from after_close_review import build_after_close_review


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

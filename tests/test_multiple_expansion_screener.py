import pandas as pd

from multiple_expansion_screener import (
    add_reacceleration_engine,
    build_daily_screener,
)


def _history(ticker: str = "6857") -> pd.DataFrame:
    dates = pd.bdate_range("2025-01-01", periods=180)
    mlp = []
    for i in range(len(dates)):
        if i < 70:
            mlp.append(1.00 + i * 0.004)
        elif i < 120:
            mlp.append(1.28 - (i - 70) * 0.0022)
        else:
            mlp.append(1.17 + (i - 120) * 0.0012)

    close = pd.Series(range(len(dates)), dtype=float) * 2 + 1000
    topix = pd.Series(range(len(dates)), dtype=float) * 0.4 + 2000

    return pd.DataFrame(
        {
            "ticker": ticker,
            "trade_date": dates,
            "mlp_c_core": mlp,
            "adj_close": close,
            "topix_close": topix,
            "fcf_engine_score": 78.0,
            "market_confirm_count": 3,
            "mex_score": 72.0,
            "state_confirmed": "WATCH",
        }
    )


def test_reacceleration_engine_adds_second_wave_columns():
    out = add_reacceleration_engine(_history())

    assert "second_wave_state" in out.columns
    assert "re_engine" in out.columns
    assert "re_route" in out.columns
    assert "me_velocity_pct" in out.columns
    assert "me_acceleration_pct" in out.columns
    assert "me_curvature_pct" in out.columns


def test_daily_screener_ranks_latest_row():
    out = build_daily_screener(_history())

    assert len(out) == 1
    assert int(out.iloc[0]["screen_rank"]) == 1
    assert 0 <= float(out.iloc[0]["sw_score"]) <= 100
    assert out.iloc[0]["candidate_type"] in {
        "SECOND WAVE",
        "FIRST WAVE",
        "OTHER",
    }

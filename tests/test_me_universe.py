import pandas as pd

from scripts.build_me_universe import (
    _build_universe_snapshot,
    _select_candidates,
)


def _master():
    return pd.DataFrame(
        {
            "ticker": ["1111", "2222", "3333"],
            "company_name": ["A", "B", "C"],
            "company_name_en": ["A", "B", "C"],
            "market_name": ["Prime", "Growth", "Standard"],
            "market_code": ["P", "G", "S"],
            "sector33_name": ["Tech", "Tech", "Chem"],
            "sector17_name": ["Info", "Info", "Materials"],
        }
    )


def _market():
    dates = pd.bdate_range("2026-01-01", periods=90)
    rows = []
    for code, base, vol in [
        ("1111", 1000.0, 200000),
        ("2222", 1500.0, 120000),
        ("3333", 800.0, 50000),
    ]:
        for i, d in enumerate(dates):
            rows.append(
                {
                    "trade_date": d,
                    "ticker": code,
                    "close": base * (1 + i * 0.001),
                    "volume": vol,
                    "per": 10 + i * 0.02 if code != "3333" else None,
                    "pbr": 1.2 + i * 0.002,
                    "market_cap": 100000 + i,
                }
            )
    return pd.DataFrame(rows)


def test_universe_snapshot_builds_liquidity_and_coarse_score():
    out = _build_universe_snapshot(
        _market(),
        _master(),
        min_turnover_yen=50_000_000,
    )

    assert set(out["ticker"]) == {"1111", "2222", "3333"}
    assert "coarse_score" in out.columns
    assert "prefilter_state" in out.columns
    assert out.loc[out["ticker"] == "1111", "liquidity_ok"].iloc[0]


def test_candidate_selector_pins_portfolio_and_reserves_growth():
    universe = _build_universe_snapshot(
        _market(),
        _master(),
        min_turnover_yen=50_000_000,
    )

    out = _select_candidates(
        universe,
        max_candidates=2,
        pinned_codes={"3333"},
    )

    assert "3333" in set(out["ticker"])
    assert len(out) == 2
    assert out["prefilter_rank"].tolist() == [1, 2]

import pandas as pd

from scripts.build_me_universe import (
    _build_universe_snapshot,
    _fetch_all_market_bars_by_date,
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


class _FakeClient:
    def __init__(self):
        self.calls = []

    def daily_bars(self, *, code, from_date="", to_date="", date=""):
        self.calls.append({
            "code": code,
            "from": from_date,
            "to": to_date,
            "date": date,
        })
        if date == "20261001":
            return pd.DataFrame([{"Date": date, "Code": "1111", "C": 1000, "Vo": 100}])
        if date == "20261002":
            return pd.DataFrame([{"Date": date, "Code": "2222", "C": 2000, "Vo": 200}])
        return pd.DataFrame()


def test_all_market_bars_are_fetched_by_date_only():
    client = _FakeClient()
    out = _fetch_all_market_bars_by_date(
        client,
        from_date=pd.Timestamp("2026-10-01").date(),
        to_date=pd.Timestamp("2026-10-02").date(),
    )

    assert len(out) == 2
    assert [x["date"] for x in client.calls] == ["20261001", "20261002"]
    assert all(x["code"] == "" for x in client.calls)
    assert all(x["from"] == "" and x["to"] == "" for x in client.calls)

import pandas as pd

from deep_fcf_top5 import build_deep_fcf_top5


def _row(**kwargs):
    row = {
        "trade_date": "2026-10-09",
        "ticker": "1111",
        "company_name": "Alpha",
        "fcf_ttm": 100.0,
        "fcf_ttm_complete": True,
        "fcf_per_share_3y_cagr_score": 80.0,
        "fcf_per_share_yoy_score": 70.0,
        "cfo_yoy_score": 75.0,
        "roic_score": 80.0,
        "fcf_margin_score": 85.0,
        "cash_conversion_score": 80.0,
        "net_cash_ratio_score": 70.0,
        "dilution_score": 85.0,
        "fcf_margin": 0.18,
        "cash_conversion": 0.95,
        "p_fcf": 14.0,
        "fcf_quality_grade": "B",
        "fcf_basis": "CFO_MINUS_CAPEX",
        "second_wave_state": "RE-WATCH READY",
        "sw_decision": "READY",
    }
    row.update(kwargs)
    return row


def test_deep_fcf_ranks_latest_complete_positive_fcf():
    frame = pd.DataFrame(
        [
            _row(ticker="1111", company_name="Alpha"),
            _row(
                ticker="2222",
                company_name="Beta",
                fcf_margin_score=95.0,
                cash_conversion_score=95.0,
                p_fcf=9.0,
            ),
            _row(
                ticker="3333",
                company_name="Negative",
                fcf_ttm=-10.0,
            ),
            _row(
                ticker="4444",
                company_name="Old",
                trade_date="2026-10-08",
                fcf_margin_score=100.0,
            ),
        ]
    )

    out = build_deep_fcf_top5(frame)

    assert list(out["ticker"]) == ["2222", "1111"]
    assert list(out["rank"]) == [1, 2]
    assert bool(out.iloc[0]["execution_ready"]) is True


def test_deep_fcf_does_not_bypass_entry_gate():
    frame = pd.DataFrame(
        [
            _row(
                ticker="5555",
                second_wave_state="EXP. DECELERATING",
                sw_decision="WATCH",
            )
        ]
    )

    out = build_deep_fcf_top5(frame)

    assert len(out) == 1
    assert bool(out.iloc[0]["execution_ready"]) is False
    assert "ENTRY GATE WAIT" in out.iloc[0]["reason"]


def test_deep_fcf_requires_minimum_component_coverage():
    frame = pd.DataFrame(
        [
            {
                "trade_date": "2026-10-09",
                "ticker": "6666",
                "company_name": "Sparse",
                "fcf_ttm": 100.0,
                "fcf_ttm_complete": True,
                "fcf_margin_score": 90.0,
                "cash_conversion_score": 90.0,
                "p_fcf": 10.0,
            }
        ]
    )

    out = build_deep_fcf_top5(frame, min_coverage=0.45)

    assert out.empty


def test_deep_fcf_allows_partial_but_reports_confidence():
    frame = pd.DataFrame(
        [
            _row(
                ticker="7777",
                fcf_per_share_3y_cagr_score=None,
                fcf_per_share_yoy_score=None,
                cfo_yoy_score=None,
            )
        ]
    )

    out = build_deep_fcf_top5(frame, min_coverage=0.45)

    assert len(out) == 1
    assert out.iloc[0]["deep_fcf_coverage"] >= 0.45
    assert out.iloc[0]["deep_fcf_confidence"] in {"LOW", "MEDIUM", "HIGH"}

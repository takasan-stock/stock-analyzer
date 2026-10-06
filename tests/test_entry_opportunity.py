from entry_opportunity import build_entry_opportunity


def _candidate(**overrides):
    base = {
        "source": "SHORT+ME",
        "signal_key": "CONFLUENCE|R-READY|EARLY",
        "source_score": 90,
        "adaptive_bonus": 3.0,
        "adaptive_confidence": "ADAPTIVE",
        "state_bonus": 2.0,
        "state_confidence": "WARMING",
        "trait_bonus": 1.0,
        "trait_confidence": "LOW SAMPLE",
    }
    base.update(overrides)
    return base


def test_ready_high_quality_becomes_entry_priority():
    out = build_entry_opportunity(
        _candidate(),
        {"status": "🟢 ENTRY READY", "score": 85},
    )
    assert out["opportunity_score"] >= 85
    assert out["opportunity_rating"] in {"A+", "S"}
    assert out["opportunity_action"] == "ENTRY PRIORITY"
    assert out["opportunity_coverage"] == 100


def test_wait_never_becomes_s_grade():
    out = build_entry_opportunity(
        _candidate(source_score=100),
        {"status": "🟡 WAIT", "score": 95},
    )
    assert out["opportunity_score"] <= 79
    assert out["opportunity_action"] in {"HIGH WATCH", "WATCH", "WAIT"}


def test_cancel_is_hard_capped():
    out = build_entry_opportunity(
        _candidate(source_score=100),
        {"status": "🔴 CANCEL", "score": 100},
    )
    assert out["opportunity_score"] <= 39
    assert out["opportunity_action"] == "AVOID"


def test_missing_intraday_reduces_coverage():
    out = build_entry_opportunity(
        _candidate(source_score=88),
        {"status": "⚪ NO DATA"},
    )
    assert out["opportunity_coverage"] == 55
    assert out["opportunity_action"] == "WAIT DATA"
    assert "Intraday未確定" in out["opportunity_reason"]

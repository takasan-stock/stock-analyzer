import json

import pandas as pd

from operational_health import build_operational_health


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False),
        encoding="utf-8",
    )


def test_health_is_ok_after_fresh_close_and_next_session_validation(tmp_path):
    me_dir = tmp_path / "multiple_expansion"
    _write_csv(
        me_dir / "me_screener_latest.csv",
        [{"ticker": "2670", "trade_date": "2026-10-09"}],
    )
    _write_json(
        me_dir / "me_screener_freshness_status.json",
        {
            "status": "FRESH",
            "stale": False,
            "actual_trade_date": "2026-10-09",
            "expected_trade_date": "2026-10-09",
        },
    )
    _write_json(
        tmp_path / "entry_hunter_validation_status.json",
        {
            "run_at": "2026-10-09T18:00:00+09:00",
            "validation_as_of": "2026-10-13T09:00:00",
            "me_promoted_count": 0,
            "me_watch_candidates": 1,
        },
    )
    _write_json(
        tmp_path / "short_cover_entry_status.json",
        {
            "run_at": "2026-10-09T10:30:00+09:00",
        },
    )

    out = build_operational_health(
        tmp_path,
        now="2026-10-10T18:00:00+09:00",
    )

    assert out["overall"] == "OK"
    assert out["expected_session"] == "2026-10-09"
    assert out["next_session"] == "2026-10-13"
    assert [x["status"] for x in out["rows"]] == ["OK", "OK", "IDLE"]


def test_health_alerts_when_me_trade_date_is_stale(tmp_path):
    me_dir = tmp_path / "multiple_expansion"
    _write_csv(
        me_dir / "me_screener_latest.csv",
        [{"ticker": "2670", "trade_date": "2026-10-08"}],
    )
    _write_json(
        tmp_path / "entry_hunter_validation_status.json",
        {
            "run_at": "2026-10-09T18:00:00+09:00",
            "validation_as_of": "2026-10-13T09:00:00",
        },
    )

    out = build_operational_health(
        tmp_path,
        now="2026-10-10T18:00:00+09:00",
    )

    assert out["overall"] == "ALERT"
    assert out["rows"][0]["status"] == "STALE"


def test_health_alerts_when_live_entry_hunter_has_not_updated_today(tmp_path):
    me_dir = tmp_path / "multiple_expansion"
    _write_csv(
        me_dir / "me_screener_latest.csv",
        [{"ticker": "2670", "trade_date": "2026-10-13"}],
    )
    _write_json(
        me_dir / "me_screener_freshness_status.json",
        {"status": "FRESH", "stale": False},
    )
    _write_json(
        tmp_path / "entry_hunter_validation_status.json",
        {
            "run_at": "2026-10-12T18:00:00+09:00",
            "validation_as_of": "2026-10-13T09:00:00",
        },
    )
    _write_json(
        tmp_path / "short_cover_entry_status.json",
        {"run_at": "2026-10-09T10:30:00+09:00"},
    )

    out = build_operational_health(
        tmp_path,
        now="2026-10-13T10:00:00+09:00",
    )

    assert out["overall"] == "ALERT"
    assert out["rows"][2]["status"] == "STALE"

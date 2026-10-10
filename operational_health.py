from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from tse_calendar import (
    is_tse_business_day,
    latest_completed_tse_session,
    next_tse_session,
)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype={"ticker": str})
    except Exception:
        return pd.DataFrame()


def _parse_ts(value: Any) -> pd.Timestamp | None:
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return None
    ts = pd.Timestamp(ts)
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Tokyo")
    return ts.tz_convert("Asia/Tokyo")


def _max_trade_date(frame: pd.DataFrame) -> pd.Timestamp | None:
    if frame.empty or "trade_date" not in frame.columns:
        return None
    vals = pd.to_datetime(frame["trade_date"], errors="coerce").dropna()
    if vals.empty:
        return None
    return pd.Timestamp(vals.max()).normalize()


def build_operational_health(
    base_dir: str | Path = "data",
    *,
    now: Any | None = None,
) -> dict[str, Any]:
    base = Path(base_dir)
    ref = (
        pd.Timestamp.now(tz="Asia/Tokyo")
        if now is None
        else pd.Timestamp(now)
    )
    if ref.tzinfo is None:
        ref = ref.tz_localize("Asia/Tokyo")
    else:
        ref = ref.tz_convert("Asia/Tokyo")

    me_dir = base / "multiple_expansion"
    freshness = _read_json(me_dir / "me_screener_freshness_status.json")
    me_latest = _read_csv(me_dir / "me_screener_latest.csv")
    validation = _read_json(base / "entry_hunter_validation_status.json")
    entry = _read_json(base / "short_cover_entry_status.json")

    expected_session = pd.Timestamp(
        latest_completed_tse_session(ref)
    ).normalize()
    next_session = pd.Timestamp(
        next_tse_session(expected_session)
    ).normalize()

    rows: list[dict[str, Any]] = []

    freshness_status = str(freshness.get("status", "") or "").upper()
    actual_me = _max_trade_date(me_latest)
    if freshness_status == "STALE" or bool(freshness.get("stale")):
        me_state = "STALE"
        me_detail = str(freshness.get("reason", "") or "日足データが古い状態です。")
    elif actual_me is None:
        me_state = "PENDING"
        me_detail = "MEスクリーナーの基準日を確認できません。"
    elif actual_me < expected_session:
        me_state = "STALE"
        me_detail = (
            f"ME基準日 {actual_me:%Y-%m-%d} / "
            f"期待 {expected_session:%Y-%m-%d}"
        )
    else:
        me_state = "OK"
        me_detail = f"ME基準日 {actual_me:%Y-%m-%d}"
    rows.append({
        "component": "ME Daily",
        "status": me_state,
        "detail": me_detail,
    })

    validation_run = _parse_ts(validation.get("run_at"))
    validation_as_of = pd.to_datetime(
        validation.get("validation_as_of"),
        errors="coerce",
    )
    if validation_run is None:
        val_state = "PENDING"
        val_detail = "次セッション検証結果がまだありません。"
    else:
        as_of_date = (
            None
            if pd.isna(validation_as_of)
            else pd.Timestamp(validation_as_of).normalize()
        )
        if as_of_date is not None and as_of_date >= next_session:
            val_state = "OK"
            val_detail = (
                f"次セッション {as_of_date:%Y-%m-%d} / "
                f"ME昇格 {int(validation.get('me_promoted_count', 0) or 0)}件 / "
                f"ME WATCH {int(validation.get('me_watch_candidates', 0) or 0)}件"
            )
        else:
            val_state = "WARN"
            val_detail = (
                "検証先セッションが古い可能性があります。"
                if as_of_date is None
                else (
                    f"検証先 {as_of_date:%Y-%m-%d} / "
                    f"次東証セッション {next_session:%Y-%m-%d}"
                )
            )
    rows.append({
        "component": "Next Session Validation",
        "status": val_state,
        "detail": val_detail,
    })

    entry_run = _parse_ts(entry.get("run_at"))
    live_window = (
        is_tse_business_day(ref)
        and (ref.hour * 60 + ref.minute) >= 9 * 60 + 15
        and (ref.hour * 60 + ref.minute) <= 11 * 60
    )
    if live_window:
        if entry_run is None or entry_run.date() != ref.date():
            entry_state = "STALE"
            entry_detail = "場中監視の当日更新が確認できません。"
        else:
            age_min = max(
                0.0,
                (ref - entry_run).total_seconds() / 60.0,
            )
            if age_min <= 20:
                entry_state = "OK"
            elif age_min <= 45:
                entry_state = "WARN"
            else:
                entry_state = "STALE"
            entry_detail = (
                f"最終更新 {entry_run:%H:%M} JST / "
                f"{age_min:.0f}分前"
            )
    else:
        if entry_run is None:
            entry_state = "PENDING"
            entry_detail = "Entry Hunterの更新履歴がありません。"
        else:
            entry_state = "IDLE"
            entry_detail = (
                f"市場時間外｜最終更新 {entry_run:%Y-%m-%d %H:%M} JST"
            )
    rows.append({
        "component": "Entry Hunter",
        "status": entry_state,
        "detail": entry_detail,
    })

    statuses = {row["status"] for row in rows}
    if "STALE" in statuses:
        overall = "ALERT"
    elif "WARN" in statuses:
        overall = "WARN"
    elif statuses <= {"OK", "IDLE"}:
        overall = "OK"
    else:
        overall = "PENDING"

    return {
        "overall": overall,
        "checked_at": ref.isoformat(),
        "expected_session": expected_session.strftime("%Y-%m-%d"),
        "next_session": next_session.strftime("%Y-%m-%d"),
        "rows": rows,
    }


def health_icon(status: str) -> str:
    return {
        "OK": "🟢",
        "IDLE": "⚪",
        "PENDING": "🟡",
        "WARN": "🟠",
        "STALE": "🔴",
        "ALERT": "🔴",
    }.get(str(status).upper(), "⚪")

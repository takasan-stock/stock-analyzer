from __future__ import annotations

from typing import Any

import pandas as pd

from entry_hunter_sources import select_me_entry_candidates
from entry_opportunity import build_entry_opportunity


COMMAND_CENTER_COLUMNS = [
    "rank",
    "ticker",
    "name",
    "opportunity_score",
    "opportunity_rating",
    "opportunity_action",
    "source",
    "signal_key",
    "entry_status",
    "coverage",
    "reason",
    "mode",
]


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _rating(score: float) -> str:
    if score >= 95:
        return "S"
    if score >= 85:
        return "A+"
    if score >= 78:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    return "C"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMAND_CENTER_COLUMNS)


def _from_entry_status(entry_status: dict[str, Any] | None) -> pd.DataFrame:
    if not isinstance(entry_status, dict):
        return _empty()

    rows = entry_status.get("rows", [])
    if not isinstance(rows, list) or not rows:
        return _empty()

    frame = pd.DataFrame(rows)
    if frame.empty or "ticker" not in frame.columns:
        return _empty()

    if "opportunity_score" not in frame.columns:
        return _empty()

    frame["opportunity_score"] = pd.to_numeric(
        frame["opportunity_score"], errors="coerce"
    )
    frame = frame.dropna(subset=["opportunity_score"]).copy()
    if frame.empty:
        return _empty()

    frame["ticker"] = frame["ticker"].astype(str).str.replace(".0", "", regex=False)
    frame["name"] = frame.get("name", pd.Series("", index=frame.index)).fillna("")
    frame["source"] = frame.get("source", pd.Series("", index=frame.index)).fillna("")
    frame["signal_key"] = frame.get(
        "signal_key", pd.Series("", index=frame.index)
    ).fillna("")
    frame["entry_status"] = frame.get(
        "status", pd.Series("", index=frame.index)
    ).fillna("")
    frame["opportunity_rating"] = frame.get(
        "opportunity_rating", pd.Series("", index=frame.index)
    ).fillna("")
    frame["opportunity_action"] = frame.get(
        "opportunity_action", pd.Series("", index=frame.index)
    ).fillna("")
    frame["coverage"] = pd.to_numeric(
        frame.get("opportunity_coverage", pd.Series(100, index=frame.index)),
        errors="coerce",
    ).fillna(100)
    frame["reason"] = frame.get(
        "opportunity_reason", pd.Series("", index=frame.index)
    ).fillna("")
    frame["mode"] = "LIVE"

    avoid = frame["opportunity_action"].astype(str).isin(
        {"AVOID", "CAUTION"}
    )
    frame = frame[~avoid].copy()
    if frame.empty:
        return _empty()

    frame["_status_priority"] = frame["entry_status"].map(
        {
            "🟢 ENTRY CONFIRMED": 5,
            "🟢 ENTRY READY": 4,
            "🟦 MONITORING": 3,
            "🟡 WAIT": 2,
            "⚪ NO DATA": 1,
        }
    ).fillna(0)

    frame = frame.sort_values(
        ["opportunity_score", "_status_priority", "ticker"],
        ascending=[False, False, True],
    )
    frame = frame.drop_duplicates("ticker", keep="first")
    return frame[
        [
            "ticker",
            "name",
            "opportunity_score",
            "opportunity_rating",
            "opportunity_action",
            "source",
            "signal_key",
            "entry_status",
            "coverage",
            "reason",
            "mode",
        ]
    ].reset_index(drop=True)


def _from_me_screener(
    me_screener: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    if me_screener is None or me_screener.empty:
        return _empty()

    candidates = select_me_entry_candidates(
        me_screener,
        as_of=as_of,
        max_calendar_days=5,
        limit=8,
    )
    if candidates.empty:
        return _empty()

    rows = []
    for _, candidate in candidates.iterrows():
        opportunity = build_entry_opportunity(
            candidate.to_dict(),
            {"status": "⚪ NO DATA"},
        )
        rows.append(
            {
                "ticker": _text(candidate.get("ticker")),
                "name": _text(candidate.get("name")),
                "opportunity_score": opportunity["opportunity_score"],
                "opportunity_rating": opportunity["opportunity_rating"],
                "opportunity_action": "PRE-MARKET WATCH",
                "source": _text(candidate.get("source")),
                "signal_key": _text(candidate.get("signal_key")),
                "entry_status": "⏳ PRE-MARKET",
                "coverage": opportunity["opportunity_coverage"],
                "reason": opportunity["opportunity_reason"],
                "mode": "PRE-MARKET",
            }
        )

    return pd.DataFrame(rows)


def build_daily_command_center(
    entry_status: dict[str, Any] | None,
    me_screener: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
    limit: int = 3,
) -> pd.DataFrame:
    """Return the day's highest-priority names for the dashboard.

    During the live monitoring window, Entry Hunter opportunity is preferred.
    Before/after live data exists, the command center falls back to the latest
    ME handoff candidates and clearly marks them PRE-MARKET with 55% coverage.
    """
    live = _from_entry_status(entry_status)
    source = live

    if source.empty:
        source = _from_me_screener(me_screener, as_of=as_of)

    if source.empty:
        return _empty()

    source = source.copy()
    source["opportunity_score"] = pd.to_numeric(
        source["opportunity_score"], errors="coerce"
    )
    source = source.dropna(subset=["opportunity_score"])
    if source.empty:
        return _empty()

    source = source.sort_values(
        ["opportunity_score", "ticker"],
        ascending=[False, True],
    ).head(max(1, int(limit))).reset_index(drop=True)

    source["rank"] = range(1, len(source) + 1)
    source["opportunity_rating"] = source.apply(
        lambda row: (
            _text(row.get("opportunity_rating"))
            or _rating(float(row["opportunity_score"]))
        ),
        axis=1,
    )

    return source[COMMAND_CENTER_COLUMNS].copy()

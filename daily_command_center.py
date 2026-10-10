from __future__ import annotations

from typing import Any

import pandas as pd

from tse_calendar import next_tse_session

from entry_hunter_sources import (
    select_me_entry_candidates,
    select_me_watch_candidates,
)
from entry_opportunity import build_entry_opportunity


COMMAND_CENTER_COLUMNS = [
    "rank",
    "previous_rank",
    "rank_change",
    "rank_trend",
    "ticker",
    "name",
    "opportunity_score",
    "opportunity_rating",
    "opportunity_action",
    "source",
    "signal_key",
    "entry_status",
    "coverage",
    "source_bonus",
    "setup_bonus",
    "trait_bonus",
    "fast_bonus",
    "adaptive_total",
    "adaptive_breakdown",
    "decision_card",
    "reason",
    "mode",
]

HISTORY_COLUMNS = [
    "snapshot_date",
    "snapshot_at",
    *COMMAND_CENTER_COLUMNS,
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




SCORE_BREAKDOWN_SPECS = [
    ("Source", "source_bonus", 8.0),
    ("Setup", "setup_bonus", 4.0),
    ("Trait", "trait_bonus", 2.5),
    ("Fast0D", "fast_bonus", 2.5),
]


def build_score_breakdown_rows(
    row: pd.Series | dict[str, Any],
) -> list[dict[str, Any]]:
    """Build compact normalized bars for the four adaptive contributions."""
    items: list[dict[str, Any]] = []
    for label, key, cap in SCORE_BREAKDOWN_SPECS:
        value = _num(row.get(key)) or 0.0
        value = max(-cap, min(cap, value))
        ratio = 0.0 if cap <= 0 else min(1.0, abs(value) / cap)
        items.append(
            {
                "label": label,
                "key": key,
                "value": round(float(value), 3),
                "cap": float(cap),
                "ratio": round(float(ratio), 4),
                "percent": round(float(ratio * 100.0), 1),
                "direction": (
                    "positive"
                    if value > 0
                    else ("negative" if value < 0 else "neutral")
                ),
            }
        )
    return items


def tradingview_url(ticker: Any) -> str:
    code = _text(ticker).replace(".T", "")
    return f"https://www.tradingview.com/chart/?symbol=TSE%3A{code}"



def execution_handoff_allowed(
    row: pd.Series | dict[str, Any],
) -> bool:
    """Whether Command Center may open Entry Hunter / Pre-Trade.

    WATCH ONLY is discovery, not execution. It must stay visible for review
    and TradingView inspection, but it cannot jump the maturity gates.
    """
    action = _text(row.get("opportunity_action")).upper()
    status = _text(row.get("entry_status")).upper()
    signal = _text(row.get("signal_key")).upper()

    if action == "WATCH ONLY":
        return False
    if status == "🟡 ME WATCH":
        return False
    if signal == "ME|WATCH":
        return False
    return True


def build_command_center_handoff(
    row: pd.Series | dict[str, Any],
) -> dict[str, Any]:
    return {
        "ticker": _text(row.get("ticker")),
        "name": _text(row.get("name")),
        "source": _text(row.get("source")),
        "signal_key": _text(row.get("signal_key")),
        "opportunity_score": _num(row.get("opportunity_score")),
        "opportunity_rating": _text(row.get("opportunity_rating")),
        "opportunity_action": _text(row.get("opportunity_action")),
        "opportunity_reason": _text(row.get("reason")),
        "opportunity_coverage": _num(row.get("coverage")),
        "source_bonus": _num(row.get("source_bonus")),
        "setup_bonus": _num(row.get("setup_bonus")),
        "trait_bonus": _num(row.get("trait_bonus")),
        "fast_bonus": _num(row.get("fast_bonus")),
        "adaptive_total": _num(row.get("adaptive_total")),
        "adaptive_breakdown": _text(row.get("adaptive_breakdown")),
        "decision_card": _text(row.get("decision_card")),
        "mode": _text(row.get("mode")),
    }


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


def _as_of_jst(value: Any | None) -> pd.Timestamp:
    ts = (
        pd.Timestamp.now(tz="Asia/Tokyo")
        if value is None
        else pd.Timestamp(value)
    )
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Tokyo")
    return ts.tz_convert("Asia/Tokyo")


def _as_of_ts(value: Any | None) -> pd.Timestamp:
    return _as_of_jst(value).tz_localize(None)


def resolve_command_center_mode(as_of: Any | None = None) -> str:
    """Resolve the dashboard mode in Japan time."""
    ts = _as_of_jst(as_of)
    if ts.weekday() >= 5:
        return "AFTER CLOSE"

    minutes = ts.hour * 60 + ts.minute
    if minutes < 9 * 60:
        return "PRE-MARKET"
    if minutes < 15 * 60 + 30:
        return "LIVE"
    return "AFTER CLOSE"


def _empty() -> pd.DataFrame:
    return pd.DataFrame(columns=COMMAND_CENTER_COLUMNS)


def _history_empty() -> pd.DataFrame:
    return pd.DataFrame(columns=HISTORY_COLUMNS)


def _is_entry_status_fresh(
    entry_status: dict[str, Any] | None,
    *,
    as_of: Any | None,
) -> bool:
    if not isinstance(entry_status, dict):
        return False
    run_at = pd.to_datetime(entry_status.get("run_at"), errors="coerce")
    if pd.isna(run_at):
        return False
    return _as_of_ts(run_at).normalize() == _as_of_ts(as_of).normalize()


def _from_entry_status(
    entry_status: dict[str, Any] | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    if not _is_entry_status_fresh(entry_status, as_of=as_of):
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
    for _col in [
        "source_bonus",
        "setup_bonus",
        "trait_bonus",
        "fast_bonus",
        "adaptive_total",
    ]:
        frame[_col] = pd.to_numeric(
            frame.get(_col, pd.Series(0.0, index=frame.index)),
            errors="coerce",
        ).fillna(0.0)
    frame["adaptive_breakdown"] = frame.get(
        "adaptive_breakdown", pd.Series("", index=frame.index)
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
            "source_bonus",
            "setup_bonus",
            "trait_bonus",
            "fast_bonus",
            "adaptive_total",
            "adaptive_breakdown",
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
    watch_only = False
    if candidates.empty:
        candidates = select_me_watch_candidates(
            me_screener,
            as_of=as_of,
            max_calendar_days=5,
            limit=8,
        )
        watch_only = not candidates.empty
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
                "opportunity_action": (
                    "WATCH ONLY" if watch_only else "PRE-MARKET WATCH"
                ),
                "source": _text(candidate.get("source")),
                "signal_key": _text(candidate.get("signal_key")),
                "entry_status": (
                    "🟡 ME WATCH" if watch_only else "⏳ PRE-MARKET"
                ),
                "coverage": opportunity["opportunity_coverage"],
                "source_bonus": opportunity.get("source_bonus", 0.0),
                "setup_bonus": opportunity.get("setup_bonus", 0.0),
                "trait_bonus": opportunity.get("trait_bonus", 0.0),
                "fast_bonus": opportunity.get("fast_bonus", 0.0),
                "adaptive_total": opportunity.get("adaptive_total", 0.0),
                "adaptive_breakdown": opportunity.get("adaptive_breakdown", ""),
                "reason": opportunity["opportunity_reason"],
                "mode": "PRE-MARKET",
            }
        )

    return pd.DataFrame(rows)


def build_decision_card(row: pd.Series | dict[str, Any]) -> str:
    source = _text(row.get("source"))
    signal_key = _text(row.get("signal_key"))
    status = _text(row.get("entry_status"))
    reason = _text(row.get("reason"))
    coverage = _num(row.get("coverage"))

    parts: list[str] = []
    if source == "SHORT+ME":
        parts.append("SHORT+ME合流")
    elif source == "ME HUNTER":
        parts.append("ME候補")
    elif source == "SHORT COVER":
        parts.append("Short Cover候補")
    elif source:
        parts.append(source)

    if signal_key:
        setup = signal_key
        setup = setup.replace("CONFLUENCE|", "")
        setup = setup.replace("ME|", "")
        setup = setup.replace("SC|", "")
        setup = setup.replace("|", "×")
        if setup and setup not in {"OTHER", "WATCH"}:
            parts.append(setup)

    if status == "🟢 ENTRY CONFIRMED":
        parts.append("寄り後CONFIRMED")
    elif status == "🟢 ENTRY READY":
        parts.append("寄り後READY")
    elif status == "🟦 MONITORING":
        parts.append("場中監視")
    elif status == "🟡 WAIT":
        parts.append("条件待ち")
    elif status == "⏳ PRE-MARKET":
        parts.append("寄り後確認待ち")
    elif status == "🟡 ME WATCH":
        parts.append("再加速待ち")

    evidence = []
    if "Source実績+" in reason:
        evidence.append("Source")
    if "Setup実績+" in reason:
        evidence.append("Setup")
    if "銘柄特性+" in reason:
        evidence.append("Trait")
    if evidence:
        parts.append("+".join(evidence) + "実績+")

    if coverage is not None and coverage < 100 and "寄り後確認待ち" not in parts:
        parts.append("Coverage待ち")

    return "｜".join(parts[:4]) if parts else (reason or "優先候補")


def normalize_command_center_history(
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return _history_empty()

    out = history.copy()
    for col in HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out["snapshot_date"] = pd.to_datetime(
        out["snapshot_date"], errors="coerce"
    ).dt.normalize()
    out["snapshot_at"] = pd.to_datetime(
        out["snapshot_at"], errors="coerce"
    )
    out["ticker"] = out["ticker"].astype(str).str.replace(".0", "", regex=False)
    out["rank"] = pd.to_numeric(out["rank"], errors="coerce")
    out["previous_rank"] = pd.to_numeric(out["previous_rank"], errors="coerce")
    out["rank_change"] = pd.to_numeric(out["rank_change"], errors="coerce")
    out["opportunity_score"] = pd.to_numeric(
        out["opportunity_score"], errors="coerce"
    )
    out = out.dropna(subset=["snapshot_date", "snapshot_at", "ticker", "rank"])
    return out[HISTORY_COLUMNS].sort_values(
        ["snapshot_at", "rank"],
        ascending=[False, True],
    ).reset_index(drop=True)


def _previous_snapshot_for_date(
    history: pd.DataFrame | None,
    *,
    as_of: Any | None,
) -> pd.DataFrame:
    hist = normalize_command_center_history(history)
    if hist.empty:
        return pd.DataFrame()

    ref_date = _as_of_ts(as_of).normalize()
    prior = hist[hist["snapshot_date"] < ref_date].copy()
    if prior.empty:
        return pd.DataFrame()

    latest_date = prior["snapshot_date"].max()
    prior = prior[prior["snapshot_date"] == latest_date].copy()
    latest_at = prior["snapshot_at"].max()
    prior = prior[prior["snapshot_at"] == latest_at].copy()
    return prior.sort_values("rank").reset_index(drop=True)


def attach_rank_change(
    current: pd.DataFrame,
    history: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    if current is None or current.empty:
        return _empty()

    out = current.copy()
    previous = _previous_snapshot_for_date(history, as_of=as_of)
    previous_map = {
        _text(row.get("ticker")): int(row["rank"])
        for _, row in previous.iterrows()
        if _text(row.get("ticker")) and pd.notna(row.get("rank"))
    }

    previous_ranks = []
    changes = []
    trends = []
    for _, row in out.iterrows():
        ticker = _text(row.get("ticker"))
        rank = int(row["rank"])
        previous_rank = previous_map.get(ticker)
        previous_ranks.append(previous_rank)
        if previous_rank is None:
            changes.append(None)
            trends.append("NEW")
            continue
        change = int(previous_rank - rank)
        changes.append(change)
        if change > 0:
            trends.append(f"↑{change}")
        elif change < 0:
            trends.append(f"↓{abs(change)}")
        else:
            trends.append("→")

    out["previous_rank"] = previous_ranks
    out["rank_change"] = changes
    out["rank_trend"] = trends
    out["decision_card"] = out.apply(build_decision_card, axis=1)
    return out[COMMAND_CENTER_COLUMNS].copy()


def update_command_center_history(
    history: pd.DataFrame | None,
    current: pd.DataFrame,
    *,
    snapshot_at: Any | None = None,
    max_rows: int = 1500,
) -> pd.DataFrame:
    base = normalize_command_center_history(history)
    if current is None or current.empty:
        return base

    ts = _as_of_ts(snapshot_at)
    incoming = current.copy()
    incoming["snapshot_date"] = ts.normalize()
    incoming["snapshot_at"] = ts
    for col in HISTORY_COLUMNS:
        if col not in incoming.columns:
            incoming[col] = None

    merged = pd.concat([incoming[HISTORY_COLUMNS], base], ignore_index=True)
    merged = normalize_command_center_history(merged)
    merged = merged.drop_duplicates(
        subset=["snapshot_at", "ticker", "rank"],
        keep="first",
    )
    return merged.head(max(1, int(max_rows))).reset_index(drop=True)


def build_daily_command_center(
    entry_status: dict[str, Any] | None,
    me_screener: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
    limit: int = 3,
    history: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Return the day's highest-priority names for the dashboard.

    During the live monitoring window, Entry Hunter opportunity is preferred.
    Before/after live data exists, the command center falls back to the latest
    ME handoff candidates and clearly marks them PRE-MARKET with 55% coverage.
    """
    live = _from_entry_status(entry_status, as_of=as_of)
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
    source["previous_rank"] = None
    source["rank_change"] = None
    source["rank_trend"] = "NEW"
    source["decision_card"] = source.apply(build_decision_card, axis=1)
    source["opportunity_rating"] = source.apply(
        lambda row: (
            _text(row.get("opportunity_rating"))
            or _rating(float(row["opportunity_score"]))
        ),
        axis=1,
    )

    source = source[COMMAND_CENTER_COLUMNS].copy()
    return attach_rank_change(source, history, as_of=as_of)

def _latest_history_snapshot(
    history: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
) -> pd.DataFrame:
    hist = normalize_command_center_history(history)
    if hist.empty:
        return _empty()

    ref = _as_of_ts(as_of)
    eligible = hist[hist["snapshot_at"] <= ref].copy()
    if eligible.empty:
        return _empty()

    latest_at = eligible["snapshot_at"].max()
    snap = eligible[eligible["snapshot_at"] == latest_at].copy()
    snap = snap.sort_values("rank").reset_index(drop=True)
    for col in COMMAND_CENTER_COLUMNS:
        if col not in snap.columns:
            snap[col] = None
    snap["mode"] = "AFTER CLOSE"
    return snap[COMMAND_CENTER_COLUMNS].copy()


def me_screener_trade_date(
    me_screener: pd.DataFrame | None,
) -> pd.Timestamp | None:
    if (
        me_screener is None
        or me_screener.empty
        or "trade_date" not in me_screener.columns
    ):
        return None
    values = pd.to_datetime(
        me_screener["trade_date"], errors="coerce"
    ).dropna()
    if values.empty:
        return None
    return pd.Timestamp(values.max()).normalize()


def build_command_center_session(
    entry_status: dict[str, Any] | None,
    me_screener: pd.DataFrame | None,
    *,
    as_of: Any | None = None,
    limit: int = 3,
    history: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Build PRE-MARKET, LIVE, or AFTER CLOSE dashboard payload."""
    ref = _as_of_ts(as_of)
    mode = resolve_command_center_mode(as_of)

    if mode == "PRE-MARKET":
        primary = build_daily_command_center(
            {},
            me_screener,
            as_of=ref,
            limit=limit,
            history=history,
        )
        return {
            "mode": mode,
            "headline": "🌅 PRE-MARKET｜今日狙う3銘柄",
            "primary": primary,
            "secondary": _empty(),
            "me_status": "PREVIOUS CLOSE",
            "me_trade_date": me_screener_trade_date(me_screener),
        }

    if mode == "LIVE":
        primary = build_daily_command_center(
            entry_status,
            me_screener,
            as_of=ref,
            limit=limit,
            history=history,
        )
        return {
            "mode": mode,
            "headline": "🔥 LIVE｜今すぐ見る3銘柄",
            "primary": primary,
            "secondary": _empty(),
            "me_status": "LIVE",
            "me_trade_date": me_screener_trade_date(me_screener),
        }

    today_review = build_daily_command_center(
        entry_status,
        pd.DataFrame(),
        as_of=ref,
        limit=limit,
        history=history,
    )
    if today_review.empty:
        today_review = _latest_history_snapshot(history, as_of=ref)

    next_ref = pd.Timestamp(
        next_tse_session(ref)
    ) + pd.Timedelta(hours=9)
    next_watch = build_daily_command_center(
        {},
        me_screener,
        as_of=next_ref,
        limit=limit,
        history=None,
    )
    if not next_watch.empty:
        next_watch = next_watch.copy()
        next_watch["mode"] = "NEXT SESSION"
        next_watch["opportunity_action"] = "NEXT SESSION WATCH"
        next_watch["entry_status"] = "🌅 NEXT SESSION"
        next_watch["decision_card"] = next_watch.apply(
            build_decision_card,
            axis=1,
        )

    trade_date = me_screener_trade_date(me_screener)
    me_status = (
        "UPDATED"
        if trade_date is not None and trade_date == ref.normalize()
        else "UPDATE PENDING"
    )

    return {
        "mode": mode,
        "headline": "🌙 AFTER CLOSE｜今日の結果",
        "primary": today_review,
        "secondary": next_watch,
        "secondary_headline": "🌅 NEXT SESSION WATCH｜明日の候補",
        "me_status": me_status,
        "me_trade_date": trade_date,
    }

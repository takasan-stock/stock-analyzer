from __future__ import annotations

import hashlib
from typing import Any

import pandas as pd


POSITIVE_STATES = {
    "RE-WATCH EARLY",
    "RE-WATCH READY",
    "RE-EXP",
}

POSITIVE_DECISIONS = {
    "WATCH",
    "READY",
    "PRIORITY WATCH",
    "ACTIVE",
}

EVENT_PRIORITY = {
    "RE-EXP CONFIRMED": 100,
    "PRIORITY WATCH": 95,
    "R-READY": 90,
    "NEW R-EARLY": 80,
    "TOP10 NEW": 70,
}


def _clean_ticker(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _event_id(
    *,
    market_date: str,
    ticker: str,
    event_type: str,
    previous_state: str,
    current_state: str,
    previous_decision: str,
    current_decision: str,
) -> str:
    raw = "|".join(
        [
            market_date,
            ticker,
            event_type,
            previous_state,
            current_state,
            previous_decision,
            current_decision,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:20]


def detect_screener_changes(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    *,
    top_n: int = 10,
    baseline_if_no_previous: bool = True,
) -> pd.DataFrame:
    """Detect only actionable ME screener changes.

    The detector intentionally avoids sending a daily "same score again" alert.
    It focuses on:
      - NEW R-EARLY
      - R-EARLY -> R-READY / newly discovered R-READY
      - R-READY/RE-WATCH -> RE-EXP
      - any new PRIORITY WATCH
      - new entry into the daily Top N

    On the first ever run, an empty previous snapshot is treated as a baseline
    by default so the user does not receive a wall of one-time alerts.
    """
    if current is None or current.empty:
        return pd.DataFrame()

    cur = current.copy()
    cur["ticker"] = cur["ticker"].map(_clean_ticker)
    cur = cur.drop_duplicates("ticker", keep="last")

    if previous is None or previous.empty:
        if baseline_if_no_previous:
            return pd.DataFrame()
        prev = pd.DataFrame(columns=cur.columns)
    else:
        prev = previous.copy()
        prev["ticker"] = prev["ticker"].map(_clean_ticker)
        prev = prev.drop_duplicates("ticker", keep="last")

    prev_map = {
        str(row["ticker"]): row
        for _, row in prev.iterrows()
        if _clean_ticker(row.get("ticker"))
    }

    market_date = ""
    if "trade_date" in cur.columns:
        dt = pd.to_datetime(cur["trade_date"], errors="coerce").max()
        if pd.notna(dt):
            market_date = pd.Timestamp(dt).strftime("%Y-%m-%d")

    events: list[dict[str, Any]] = []

    for _, row in cur.iterrows():
        ticker = _clean_ticker(row.get("ticker"))
        if not ticker:
            continue

        old = prev_map.get(ticker)
        prev_state = _text(old.get("second_wave_state")) if old is not None else ""
        cur_state = _text(row.get("second_wave_state"))
        prev_decision = _text(old.get("sw_decision")) if old is not None else ""
        cur_decision = _text(row.get("sw_decision"))

        cur_rank = _num(row.get("screen_rank"))
        prev_rank = _num(old.get("screen_rank")) if old is not None else None
        sw_score = _num(row.get("sw_score"))
        hist_edge = _num(row.get("hist_edge_score"))
        fcf_score = _num(row.get("fcf_engine_score"))

        emitted: list[str] = []

        def emit(event_type: str, severity: str, reason: str) -> None:
            if event_type in emitted:
                return
            emitted.append(event_type)
            events.append(
                {
                    "event_id": _event_id(
                        market_date=market_date,
                        ticker=ticker,
                        event_type=event_type,
                        previous_state=prev_state,
                        current_state=cur_state,
                        previous_decision=prev_decision,
                        current_decision=cur_decision,
                    ),
                    "market_date": market_date,
                    "ticker": ticker,
                    "company_name": _text(row.get("company_name")),
                    "market_name": _text(row.get("market_name")),
                    "event_type": event_type,
                    "severity": severity,
                    "previous_state": prev_state or "—",
                    "current_state": cur_state or "—",
                    "previous_decision": prev_decision or "—",
                    "current_decision": cur_decision or "—",
                    "previous_rank": prev_rank,
                    "current_rank": cur_rank,
                    "sw_score": sw_score,
                    "hist_edge_score": hist_edge,
                    "fcf_engine_score": fcf_score,
                    "mlp_c_core": _num(row.get("mlp_c_core")),
                    "me_velocity_pct": _num(row.get("me_velocity_pct")),
                    "me_acceleration_pct": _num(row.get("me_acceleration_pct")),
                    "screen_reason": _text(row.get("screen_reason")),
                    "change_reason": reason,
                }
            )

        # Strongest state progression first.
        if cur_state == "RE-EXP" and prev_state != "RE-EXP":
            emit(
                "RE-EXP CONFIRMED",
                "HIGH",
                "Second Wave re-expansion confirmed.",
            )
        elif cur_state == "RE-WATCH READY" and prev_state != "RE-WATCH READY":
            emit(
                "R-READY",
                "HIGH",
                "Re-acceleration setup advanced to READY.",
            )
        elif cur_state == "RE-WATCH EARLY" and prev_state != "RE-WATCH EARLY":
            emit(
                "NEW R-EARLY",
                "MEDIUM",
                "Early re-acceleration shape detected.",
            )

        if cur_decision == "PRIORITY WATCH" and prev_decision != "PRIORITY WATCH":
            emit(
                "PRIORITY WATCH",
                "HIGH",
                "Historical edge / fundamentals lifted the name to priority watch.",
            )

        current_is_candidate = cur_decision in POSITIVE_DECISIONS
        entered_top = cur_rank is not None and cur_rank <= top_n
        was_top = prev_rank is not None and prev_rank <= top_n

        if current_is_candidate and entered_top and not was_top:
            emit(
                "TOP10 NEW",
                "MEDIUM",
                f"New entry into daily Top {top_n}.",
            )

    if not events:
        return pd.DataFrame()

    out = pd.DataFrame(events)
    out["event_priority"] = out["event_type"].map(EVENT_PRIORITY).fillna(0)
    out = out.sort_values(
        ["event_priority", "sw_score", "current_rank", "ticker"],
        ascending=[False, False, True, True],
        na_position="last",
    ).reset_index(drop=True)
    return out


def compact_change_summary(events: pd.DataFrame) -> pd.DataFrame:
    """One row per ticker for the Streamlit top panel / email digest."""
    if events is None or events.empty:
        return pd.DataFrame()

    rows = []
    for ticker, g in events.groupby("ticker", sort=False):
        g = g.sort_values(
            ["event_priority", "sw_score"],
            ascending=[False, False],
            na_position="last",
        )
        first = g.iloc[0]
        rows.append(
            {
                "ticker": ticker,
                "company_name": first.get("company_name", ""),
                "events": " + ".join(g["event_type"].astype(str).tolist()),
                "current_state": first.get("current_state", ""),
                "current_decision": first.get("current_decision", ""),
                "current_rank": first.get("current_rank"),
                "sw_score": first.get("sw_score"),
                "hist_edge_score": first.get("hist_edge_score"),
                "fcf_engine_score": first.get("fcf_engine_score"),
                "mlp_c_core": first.get("mlp_c_core"),
                "me_velocity_pct": first.get("me_velocity_pct"),
                "me_acceleration_pct": first.get("me_acceleration_pct"),
                "screen_reason": first.get("screen_reason", ""),
                "max_priority": first.get("event_priority", 0),
            }
        )

    return pd.DataFrame(rows).sort_values(
        ["max_priority", "sw_score", "current_rank"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)

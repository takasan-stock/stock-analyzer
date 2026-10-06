from __future__ import annotations

from typing import Any

import pandas as pd


ENTRY_COLUMNS = [
    "ticker",
    "name",
    "alert_date",
    "alert_tier",
    "condition_version",
    "source",
    "source_detail",
    "source_score",
    "source_rank",
]


def _ticker(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".T"):
        text = text[:-2]
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_entry_candidates(
    frame: pd.DataFrame | None,
    *,
    default_source: str,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    out = frame.copy()

    if "ticker" not in out.columns:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    out["ticker"] = out["ticker"].map(_ticker)
    out = out[out["ticker"] != ""].copy()

    if "name" not in out.columns:
        if "company_name" in out.columns:
            out["name"] = out["company_name"]
        else:
            out["name"] = ""

    if "alert_date" not in out.columns:
        if "trade_date" in out.columns:
            out["alert_date"] = out["trade_date"]
        else:
            out["alert_date"] = pd.NaT

    out["alert_date"] = pd.to_datetime(
        out["alert_date"],
        errors="coerce",
    ).dt.normalize()

    if "alert_tier" not in out.columns:
        out["alert_tier"] = ""
    if "condition_version" not in out.columns:
        out["condition_version"] = ""
    if "source" not in out.columns:
        out["source"] = default_source
    out["source"] = out["source"].fillna(default_source).astype(str)

    if "source_detail" not in out.columns:
        out["source_detail"] = ""
    if "source_score" not in out.columns:
        out["source_score"] = 0.0
    if "source_rank" not in out.columns:
        out["source_rank"] = pd.NA

    out["source_score"] = pd.to_numeric(
        out["source_score"],
        errors="coerce",
    ).fillna(0.0)
    out["source_rank"] = pd.to_numeric(
        out["source_rank"],
        errors="coerce",
    )

    for col in ENTRY_COLUMNS:
        if col not in out.columns:
            out[col] = None

    return out[ENTRY_COLUMNS].copy()


def select_me_entry_candidates(
    screener: pd.DataFrame | None,
    *,
    as_of=None,
    max_calendar_days: int = 4,
    limit: int = 5,
) -> pd.DataFrame:
    """Select ME names that should be handed to next-session Entry Hunter.

    Conservative handoff rules:
      - RE-EXP / ACTIVE: always eligible
      - PRIORITY WATCH: eligible even while still decelerating
      - RE-WATCH READY / READY: eligible
      - R-EARLY only when it is already PRIORITY WATCH

    This keeps the intraday monitor focused on names where the daily
    multiple-expansion engine has already done most of the filtering.
    """
    if screener is None or screener.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    s = screener.copy()
    if "ticker" not in s.columns:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    s["ticker"] = s["ticker"].map(_ticker)
    s["trade_date"] = pd.to_datetime(
        s.get("trade_date"),
        errors="coerce",
    ).dt.normalize()
    s = s.dropna(subset=["trade_date"])
    if s.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    ref = pd.Timestamp(as_of if as_of is not None else pd.Timestamp.now()).normalize()
    s = s[s["trade_date"] < ref].copy()
    if s.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    latest = s["trade_date"].max()
    if (ref - latest).days > int(max_calendar_days):
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    s = s[s["trade_date"] == latest].copy()

    state = s.get(
        "second_wave_state",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)
    decision = s.get(
        "sw_decision",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)

    eligible = (
        state.eq("RE-EXP")
        | decision.eq("ACTIVE")
        | decision.eq("PRIORITY WATCH")
        | state.eq("RE-WATCH READY")
        | decision.eq("READY")
    )
    s = s[eligible].copy()
    if s.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    state = s.get(
        "second_wave_state",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)
    decision = s.get(
        "sw_decision",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)

    priority = pd.Series(0.0, index=s.index)
    priority.loc[state.eq("RE-EXP")] = 100.0
    priority.loc[decision.eq("ACTIVE")] = 98.0
    priority.loc[decision.eq("PRIORITY WATCH")] = 95.0
    priority.loc[state.eq("RE-WATCH READY")] = priority.loc[
        state.eq("RE-WATCH READY")
    ].clip(lower=90.0)
    priority.loc[decision.eq("READY")] = priority.loc[
        decision.eq("READY")
    ].clip(lower=88.0)

    sw = pd.to_numeric(
        s.get("sw_score", pd.Series(0, index=s.index)),
        errors="coerce",
    ).fillna(0.0)
    hist = pd.to_numeric(
        s.get("hist_edge_score", pd.Series(0, index=s.index)),
        errors="coerce",
    ).fillna(0.0)
    fcf = pd.to_numeric(
        s.get("fcf_engine_score", pd.Series(0, index=s.index)),
        errors="coerce",
    ).fillna(0.0)
    rank = pd.to_numeric(
        s.get("screen_rank", pd.Series(pd.NA, index=s.index)),
        errors="coerce",
    )

    s["source_score"] = (
        priority * 0.55
        + sw.clip(0, 100) * 0.25
        + hist.clip(0, 100) * 0.10
        + fcf.clip(0, 100) * 0.10
    ).clip(0, 100)
    s["source_rank"] = rank

    s["name"] = s.get(
        "company_name",
        pd.Series("", index=s.index),
    ).fillna("")
    s["alert_date"] = s["trade_date"]
    s["source"] = "ME HUNTER"
    s["condition_version"] = "ME-v0.9"

    details = []
    tiers = []
    for idx, row in s.iterrows():
        row_state = str(row.get("second_wave_state", "") or "")
        row_decision = str(row.get("sw_decision", "") or "")
        row_score = _num(row.get("sw_score"))
        row_hist = _num(row.get("hist_edge_score"))

        detail = f"{row_state} / {row_decision}"
        if row_score is not None:
            detail += f" / SW {row_score:.0f}"
        if row_hist is not None:
            detail += f" / Edge {row_hist:.0f}"
        details.append(detail)

        if row_state == "RE-EXP":
            tiers.append("🔥 ME RE-EXP")
        elif row_decision == "PRIORITY WATCH":
            tiers.append("🔥 ME PRIORITY")
        elif row_state == "RE-WATCH READY" or row_decision == "READY":
            tiers.append("🟢 ME R-READY")
        else:
            tiers.append("🟡 ME WATCH")

    s["source_detail"] = details
    s["alert_tier"] = tiers

    s = s.sort_values(
        ["source_score", "source_rank", "ticker"],
        ascending=[False, True, True],
        na_position="last",
    )

    return normalize_entry_candidates(
        s.head(int(limit)),
        default_source="ME HUNTER",
    )


def normalize_short_cover_candidates(
    candidates: pd.DataFrame | None,
) -> pd.DataFrame:
    if candidates is None or candidates.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    s = candidates.copy()
    if "source_score" not in s.columns:
        if "alert_score" in s.columns:
            s["source_score"] = pd.to_numeric(
                s["alert_score"],
                errors="coerce",
            ).fillna(0.0)
        else:
            s["source_score"] = 0.0

    if "source_rank" not in s.columns:
        s["source_rank"] = range(1, len(s) + 1)

    s["source"] = "SHORT COVER"
    s["source_detail"] = s.get(
        "alert_reason",
        pd.Series("", index=s.index),
    ).fillna("")

    return normalize_entry_candidates(
        s,
        default_source="SHORT COVER",
    )


def combine_entry_candidates(
    short_cover: pd.DataFrame | None,
    me_candidates: pd.DataFrame | None,
    *,
    limit: int = 8,
) -> pd.DataFrame:
    """Merge Entry Hunter sources and dedupe by ticker.

    When the same ticker is selected by both engines, it is promoted to
    SHORT+ME and receives a small confluence bonus rather than being monitored
    twice.
    """
    short_df = normalize_short_cover_candidates(short_cover)
    me_df = normalize_entry_candidates(
        me_candidates,
        default_source="ME HUNTER",
    )

    merged = pd.concat([short_df, me_df], ignore_index=True)
    if merged.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    rows = []
    for ticker, group in merged.groupby("ticker", sort=False):
        group = group.sort_values(
            ["source_score", "source_rank"],
            ascending=[False, True],
            na_position="last",
        )

        best = group.iloc[0].copy()
        sources = set(group["source"].astype(str))
        details = [
            x
            for x in group["source_detail"].astype(str).tolist()
            if x and x != "nan"
        ]
        tiers = [
            x
            for x in group["alert_tier"].astype(str).tolist()
            if x and x != "nan"
        ]

        if "SHORT COVER" in sources and "ME HUNTER" in sources:
            best["source"] = "SHORT+ME"
            best["source_score"] = min(
                100.0,
                float(group["source_score"].max()) + 8.0,
            )
            best["alert_tier"] = "🔥 CONFLUENCE"
        else:
            best["source"] = next(iter(sources)) if sources else ""

        if details:
            best["source_detail"] = " | ".join(dict.fromkeys(details))
        if not str(best.get("alert_tier", "") or "") and tiers:
            best["alert_tier"] = tiers[0]

        valid_dates = pd.to_datetime(
            group["alert_date"],
            errors="coerce",
        ).dropna()
        if not valid_dates.empty:
            best["alert_date"] = valid_dates.max()

        rows.append(best)

    out = pd.DataFrame(rows)
    source_bonus = out["source"].map(
        {
            "SHORT+ME": 10.0,
            "ME HUNTER": 5.0,
            "SHORT COVER": 0.0,
        }
    ).fillna(0.0)
    out["_sort_score"] = (
        pd.to_numeric(out["source_score"], errors="coerce").fillna(0.0)
        + source_bonus
    )

    out = out.sort_values(
        ["_sort_score", "source_rank", "ticker"],
        ascending=[False, True, True],
        na_position="last",
    ).head(int(limit))
    out = out.drop(columns=["_sort_score"])

    return normalize_entry_candidates(
        out,
        default_source="",
    ).reset_index(drop=True)

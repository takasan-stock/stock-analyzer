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
    "signal_key",
    "market_name",
    "market_cap",
    "realized_vol20_pct",
    "trait_market",
    "trait_size",
    "trait_vol",
    "adaptive_bonus",
    "adaptive_confidence",
    "state_bonus",
    "state_confidence",
    "trait_bonus",
    "trait_confidence",
    "fast_bonus",
    "fast_confidence",
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


def _clip(value: float, low: float, high: float) -> float:
    return max(low, min(high, float(value)))


def build_source_adjustments(
    source_summary: pd.DataFrame | None,
    *,
    min_samples: int = 5,
    full_samples: int = 20,
    max_bonus: float = 8.0,
) -> dict[str, dict[str, float | str]]:
    """Convert prospective live source performance into conservative bonuses.

    Safety rules:
      - no adaptive change before at least min_samples resolved 5D trades
      - shrink the effect toward zero until full_samples observations
      - cap any source adjustment to +/- max_bonus points
      - use 5D win rate, 5D average return and MFE/MAE; no optimizer

    With little or no data, the system behaves like the fixed v0.9 ranking.
    """
    sources = ["ME HUNTER", "SHORT COVER", "SHORT+ME"]
    result = {
        source: {
            "bonus": 0.0,
            "confidence": "BASE",
            "sample_5d": 0.0,
            "raw_edge": 0.0,
        }
        for source in sources
    }
    if source_summary is None or source_summary.empty:
        return result

    summary = source_summary.copy()
    if "source" not in summary.columns:
        return result
    summary["source"] = summary["source"].fillna("").astype(str).str.upper()

    for source in sources:
        rows = summary[summary["source"] == source]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        n = _num(row.get("sample_5d")) or 0.0
        if n < float(min_samples):
            result[source]["sample_5d"] = n
            result[source]["confidence"] = "DATA BUILDING"
            continue

        win5 = _num(row.get("win_5d"))
        avg5 = _num(row.get("avg_5d"))
        mfe = _num(row.get("avg_mfe_10d"))
        mae = _num(row.get("avg_mae_10d"))

        win_component = 0.0
        avg_component = 0.0
        rr_component = 0.0

        if win5 is not None:
            win_component = _clip((win5 - 50.0) / 20.0, -1.0, 1.0)
        if avg5 is not None:
            avg_component = _clip(avg5 / 5.0, -1.0, 1.0)
        if mfe is not None and mae is not None and abs(mae) >= 0.25:
            rr = mfe / abs(mae)
            rr_component = _clip((rr - 1.5) / 1.5, -1.0, 1.0)

        raw_edge = (
            win_component * 0.45
            + avg_component * 0.35
            + rr_component * 0.20
        )
        shrink = _clip(
            (n - float(min_samples) + 1.0)
            / max(1.0, float(full_samples - min_samples + 1)),
            0.0,
            1.0,
        )
        bonus = _clip(
            raw_edge * float(max_bonus) * shrink,
            -float(max_bonus),
            float(max_bonus),
        )

        if n >= full_samples:
            confidence = "ADAPTIVE"
        elif n >= 10:
            confidence = "WARMING"
        else:
            confidence = "LOW SAMPLE"

        result[source] = {
            "bonus": round(bonus, 3),
            "confidence": confidence,
            "sample_5d": n,
            "raw_edge": round(raw_edge, 4),
        }

    return result


def build_signal_adjustments(
    signal_summary: pd.DataFrame | None,
    *,
    min_samples: int = 5,
    full_samples: int = 20,
    max_bonus: float = 4.0,
) -> dict[str, dict[str, float | str]]:
    """Build smaller state-level bonuses on top of source-level weighting."""
    result: dict[str, dict[str, float | str]] = {}
    if signal_summary is None or signal_summary.empty:
        return result
    summary = signal_summary.copy()
    if "signal_key" not in summary.columns:
        return result
    summary["signal_key"] = summary["signal_key"].fillna("").astype(str)
    for _, row in summary.iterrows():
        key = str(row.get("signal_key", "") or "").strip()
        if not key:
            continue
        n = _num(row.get("sample_5d")) or 0.0
        bonus = 0.0
        confidence = "DATA BUILDING"
        if n >= float(min_samples):
            win5 = _num(row.get("win_5d"))
            avg5 = _num(row.get("avg_5d"))
            mfe = _num(row.get("avg_mfe_10d"))
            mae = _num(row.get("avg_mae_10d"))
            win_component = 0.0 if win5 is None else _clip((win5 - 50.0) / 20.0, -1.0, 1.0)
            avg_component = 0.0 if avg5 is None else _clip(avg5 / 5.0, -1.0, 1.0)
            rr_component = 0.0
            if mfe is not None and mae is not None and abs(mae) >= 0.25:
                rr_component = _clip(((mfe / abs(mae)) - 1.5) / 1.5, -1.0, 1.0)
            raw_edge = win_component * 0.45 + avg_component * 0.35 + rr_component * 0.20
            shrink = _clip(
                (n - float(min_samples) + 1.0) / max(1.0, float(full_samples - min_samples + 1)),
                0.0,
                1.0,
            )
            bonus = _clip(raw_edge * float(max_bonus) * shrink, -float(max_bonus), float(max_bonus))
            if n >= full_samples:
                confidence = "ADAPTIVE"
            elif n >= 10:
                confidence = "WARMING"
            else:
                confidence = "LOW SAMPLE"
        result[key] = {
            "bonus": round(bonus, 3),
            "confidence": confidence,
            "sample_5d": n,
        }
    return result


def classify_candidate_traits(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    market = out.get("market_name", pd.Series("", index=out.index)).fillna("").astype(str)
    out["trait_market"] = "MARKET|OTHER"
    out.loc[market.str.contains("prime|プライム", case=False, regex=True), "trait_market"] = "MARKET|PRIME"
    out.loc[market.str.contains("standard|スタンダード", case=False, regex=True), "trait_market"] = "MARKET|STANDARD"
    out.loc[market.str.contains("growth|グロース", case=False, regex=True), "trait_market"] = "MARKET|GROWTH"

    cap = pd.to_numeric(out.get("market_cap", pd.Series(pd.NA, index=out.index)), errors="coerce")
    out["trait_size"] = "SIZE|UNKNOWN"
    out.loc[cap < 30_000_000_000, "trait_size"] = "SIZE|MICRO"
    out.loc[(cap >= 30_000_000_000) & (cap < 100_000_000_000), "trait_size"] = "SIZE|SMALL"
    out.loc[(cap >= 100_000_000_000) & (cap < 500_000_000_000), "trait_size"] = "SIZE|MID"
    out.loc[(cap >= 500_000_000_000) & (cap < 2_000_000_000_000), "trait_size"] = "SIZE|LARGE"
    out.loc[cap >= 2_000_000_000_000, "trait_size"] = "SIZE|MEGA"

    vol = pd.to_numeric(out.get("realized_vol20_pct", pd.Series(pd.NA, index=out.index)), errors="coerce")
    out["trait_vol"] = "VOL|UNKNOWN"
    out.loc[vol < 25.0, "trait_vol"] = "VOL|LOW"
    out.loc[(vol >= 25.0) & (vol < 45.0), "trait_vol"] = "VOL|MID"
    out.loc[vol >= 45.0, "trait_vol"] = "VOL|HIGH"
    return out


def build_trait_adjustments(
    trait_summary: pd.DataFrame | None,
    *,
    min_samples: int = 8,
    full_samples: int = 30,
    max_axis_bonus: float = 1.0,
) -> dict[str, dict[str, float | str]]:
    """Build tiny per-trait bonuses; each axis is capped to avoid fragmentation."""
    result: dict[str, dict[str, float | str]] = {}
    if trait_summary is None or trait_summary.empty or "trait_key" not in trait_summary.columns:
        return result
    for _, row in trait_summary.iterrows():
        key = str(row.get("trait_key", "") or "").strip()
        if not key:
            continue
        n = _num(row.get("sample_5d")) or 0.0
        bonus = 0.0
        confidence = "DATA BUILDING"
        if n >= float(min_samples):
            win5 = _num(row.get("win_5d"))
            avg5 = _num(row.get("avg_5d"))
            win_component = 0.0 if win5 is None else _clip((win5 - 50.0) / 20.0, -1.0, 1.0)
            avg_component = 0.0 if avg5 is None else _clip(avg5 / 5.0, -1.0, 1.0)
            raw = win_component * 0.60 + avg_component * 0.40
            shrink = _clip((n - min_samples + 1.0) / max(1.0, full_samples - min_samples + 1.0), 0.0, 1.0)
            bonus = _clip(raw * max_axis_bonus * shrink, -max_axis_bonus, max_axis_bonus)
            if n >= full_samples:
                confidence = "ADAPTIVE"
            elif n >= 15:
                confidence = "WARMING"
            else:
                confidence = "LOW SAMPLE"
        result[key] = {"bonus": round(bonus, 3), "confidence": confidence, "sample_5d": n}
    return result


def enrich_candidate_metadata(
    frame: pd.DataFrame | None,
    universe_meta: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame() if frame is None else frame.copy()
    out = frame.copy()
    if universe_meta is None or universe_meta.empty or "ticker" not in universe_meta.columns:
        return out

    meta = universe_meta.copy()
    meta["ticker"] = meta["ticker"].map(_ticker)
    keep = [
        col
        for col in [
            "ticker",
            "market_name",
            "market_cap",
            "realized_vol20_pct",
        ]
        if col in meta.columns
    ]
    meta = meta[keep].drop_duplicates("ticker", keep="last")
    out["ticker"] = out["ticker"].map(_ticker)

    for col in ["market_name", "market_cap", "realized_vol20_pct"]:
        if col in out.columns:
            out = out.drop(columns=[col])

    return out.merge(meta, on="ticker", how="left")



def build_fast_feedback_adjustments(
    feedback_summary: pd.DataFrame | None,
) -> dict[str, dict[str, float | str]]:
    """Read after-close 0D feedback without re-fitting anything.

    The after-close layer is intentionally tiny and temporary in influence.
    It can nudge next-session ordering before 5D outcomes mature, but the
    slower source/setup/trait adaptive layers remain dominant.
    """
    result: dict[str, dict[str, float | str]] = {}
    if (
        feedback_summary is None
        or feedback_summary.empty
        or "dimension" not in feedback_summary.columns
        or "key" not in feedback_summary.columns
    ):
        return result

    for _, row in feedback_summary.iterrows():
        dimension = str(row.get("dimension", "") or "").strip().upper()
        key = str(row.get("key", "") or "").strip()
        if not dimension or not key:
            continue
        bonus = _num(row.get("fast_bonus")) or 0.0
        confidence = str(row.get("confidence", "BASE") or "BASE")
        samples = _num(row.get("sample_0d")) or 0.0
        result[f"{dimension}:{key}"] = {
            "bonus": float(bonus),
            "confidence": confidence,
            "sample_0d": samples,
        }
    return result


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
    if "signal_key" not in out.columns:
        out["signal_key"] = ""
    if "market_name" not in out.columns:
        out["market_name"] = ""
    if "market_cap" not in out.columns:
        out["market_cap"] = pd.NA
    if "realized_vol20_pct" not in out.columns:
        out["realized_vol20_pct"] = pd.NA
    if "trait_market" not in out.columns:
        out["trait_market"] = ""
    if "trait_size" not in out.columns:
        out["trait_size"] = ""
    if "trait_vol" not in out.columns:
        out["trait_vol"] = ""
    if "adaptive_bonus" not in out.columns:
        out["adaptive_bonus"] = 0.0
    if "adaptive_confidence" not in out.columns:
        out["adaptive_confidence"] = "BASE"
    if "state_bonus" not in out.columns:
        out["state_bonus"] = 0.0
    if "state_confidence" not in out.columns:
        out["state_confidence"] = "BASE"
    if "trait_bonus" not in out.columns:
        out["trait_bonus"] = 0.0
    if "trait_confidence" not in out.columns:
        out["trait_confidence"] = "BASE"
    if "fast_bonus" not in out.columns:
        out["fast_bonus"] = 0.0
    if "fast_confidence" not in out.columns:
        out["fast_confidence"] = "BASE"

    out["source_score"] = pd.to_numeric(
        out["source_score"],
        errors="coerce",
    ).fillna(0.0)
    out["source_rank"] = pd.to_numeric(
        out["source_rank"],
        errors="coerce",
    )
    out["adaptive_bonus"] = pd.to_numeric(
        out["adaptive_bonus"],
        errors="coerce",
    ).fillna(0.0)
    out["signal_key"] = out["signal_key"].fillna("").astype(str)
    out["market_name"] = out["market_name"].fillna("").astype(str)
    out["market_cap"] = pd.to_numeric(out["market_cap"], errors="coerce")
    out["realized_vol20_pct"] = pd.to_numeric(
        out["realized_vol20_pct"], errors="coerce"
    )
    out["trait_market"] = out["trait_market"].fillna("").astype(str)
    out["trait_size"] = out["trait_size"].fillna("").astype(str)
    out["trait_vol"] = out["trait_vol"].fillna("").astype(str)
    out["adaptive_confidence"] = (
        out["adaptive_confidence"].fillna("BASE").astype(str)
    )
    out["state_bonus"] = pd.to_numeric(
        out["state_bonus"],
        errors="coerce",
    ).fillna(0.0)
    out["state_confidence"] = (
        out["state_confidence"].fillna("BASE").astype(str)
    )
    out["trait_bonus"] = pd.to_numeric(
        out["trait_bonus"], errors="coerce"
    ).fillna(0.0)
    out["trait_confidence"] = (
        out["trait_confidence"].fillna("BASE").astype(str)
    )
    out["fast_bonus"] = pd.to_numeric(
        out["fast_bonus"], errors="coerce"
    ).fillna(0.0)
    out["fast_confidence"] = (
        out["fast_confidence"].fillna("BASE").astype(str)
    )

    for col in ENTRY_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out = classify_candidate_traits(out)
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
    signal_keys = []
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
            signal_keys.append("ME|RE-EXP")
        elif row_decision == "PRIORITY WATCH":
            tiers.append("🔥 ME PRIORITY")
            signal_keys.append("ME|PRIORITY")
        elif row_state == "RE-WATCH READY" or row_decision == "READY":
            tiers.append("🟢 ME R-READY")
            signal_keys.append("ME|R-READY")
        else:
            tiers.append("🟡 ME WATCH")
            signal_keys.append("ME|WATCH")

    s["source_detail"] = details
    s["alert_tier"] = tiers
    s["signal_key"] = signal_keys

    s = s.sort_values(
        ["source_score", "source_rank", "ticker"],
        ascending=[False, True, True],
        na_position="last",
    )

    return normalize_entry_candidates(
        s.head(int(limit)),
        default_source="ME HUNTER",
    )



def select_me_watch_candidates(
    screener: pd.DataFrame | None,
    *,
    as_of=None,
    max_calendar_days: int = 5,
    limit: int = 5,
) -> pd.DataFrame:
    """Return ME watch-only names without promoting them to Entry Hunter.

    This lane is intentionally separate from the Entry Hunter selector.
    Typical use is the Daily Command Center, where users should still see a
    decelerating second-wave setup even though the intraday Entry Hunter must
    wait for READY / PRIORITY / RE-EXP.

    Eligible watch-only rows:
      - latest completed ME date before as_of
      - second-wave candidate
      - WATCH decision
      - not already eligible for the Entry Hunter handoff
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

    ref = pd.Timestamp(
        as_of if as_of is not None else pd.Timestamp.now()
    ).normalize()
    s = s[s["trade_date"] < ref].copy()
    if s.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    latest = s["trade_date"].max()
    if (ref - latest).days > int(max_calendar_days):
        return pd.DataFrame(columns=ENTRY_COLUMNS)
    s = s[s["trade_date"] == latest].copy()

    decision = s.get(
        "sw_decision",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)
    candidate_type = s.get(
        "candidate_type",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)
    state = s.get(
        "second_wave_state",
        pd.Series("", index=s.index),
    ).fillna("").astype(str)

    mature = (
        state.eq("RE-EXP")
        | decision.eq("ACTIVE")
        | decision.eq("PRIORITY WATCH")
        | state.eq("RE-WATCH READY")
        | decision.eq("READY")
    )
    watch_only = (
        decision.eq("WATCH")
        & (
            candidate_type.eq("SECOND WAVE")
            | state.eq("EXP. DECELERATING")
            | state.eq("RE-WATCH EARLY")
        )
        & ~mature
    )
    s = s[watch_only].copy()
    if s.empty:
        return pd.DataFrame(columns=ENTRY_COLUMNS)

    sw = pd.to_numeric(
        s.get("sw_score", pd.Series(0, index=s.index)),
        errors="coerce",
    ).fillna(0.0)
    hist = pd.to_numeric(
        s.get("hist_edge_score", pd.Series(45, index=s.index)),
        errors="coerce",
    ).fillna(45.0)
    fcf = pd.to_numeric(
        s.get("fcf_engine_score", pd.Series(50, index=s.index)),
        errors="coerce",
    ).fillna(50.0)
    rank = pd.to_numeric(
        s.get("screen_rank", pd.Series(pd.NA, index=s.index)),
        errors="coerce",
    )

    s["source_score"] = (
        sw.clip(0, 100) * 0.70
        + hist.clip(0, 100) * 0.15
        + fcf.clip(0, 100) * 0.15
    ).clip(0, 100)
    s["source_rank"] = rank
    s["name"] = s.get(
        "company_name",
        pd.Series("", index=s.index),
    ).fillna("")
    s["alert_date"] = s["trade_date"]
    s["source"] = "ME HUNTER"
    s["condition_version"] = "ME-v0.9"
    s["alert_tier"] = "🟡 ME WATCH"
    s["signal_key"] = "ME|WATCH"

    details = []
    for _, row in s.iterrows():
        detail = (
            f"{str(row.get('second_wave_state', '') or '')} / "
            f"{str(row.get('sw_decision', '') or '')}"
        )
        route = str(row.get("re_route", "") or "")
        if route:
            detail += f" / {route}"
        details.append(detail)
    s["source_detail"] = details

    s = s.sort_values(
        ["source_score", "source_rank", "ticker"],
        ascending=[False, True, True],
        na_position="last",
    )
    return normalize_entry_candidates(
        s.head(max(1, int(limit))),
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

    phase = s.get("phase", pd.Series("", index=s.index)).fillna("").astype(str)
    s["signal_key"] = "SC|OTHER"
    s.loc[phase.eq("🚀 SQUEEZE"), "signal_key"] = "SC|SQUEEZE"
    s.loc[phase.eq("✅ COVER CONFIRMED"), "signal_key"] = "SC|CONFIRMED"
    s.loc[phase.eq("🔥 COVER EARLY"), "signal_key"] = "SC|EARLY"

    return normalize_entry_candidates(
        s,
        default_source="SHORT COVER",
    )


def combine_entry_candidates(
    short_cover: pd.DataFrame | None,
    me_candidates: pd.DataFrame | None,
    *,
    limit: int = 8,
    source_summary: pd.DataFrame | None = None,
    signal_summary: pd.DataFrame | None = None,
    trait_summary: pd.DataFrame | None = None,
    universe_meta: pd.DataFrame | None = None,
    fast_feedback_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Merge Entry Hunter sources and dedupe by ticker.

    When the same ticker is selected by both engines, it is promoted to
    SHORT+ME and receives a small confluence bonus rather than being monitored
    twice.
    """
    short_cover = enrich_candidate_metadata(short_cover, universe_meta)
    me_candidates = enrich_candidate_metadata(me_candidates, universe_meta)

    short_df = normalize_short_cover_candidates(short_cover)
    me_df = normalize_entry_candidates(
        me_candidates,
        default_source="ME HUNTER",
    )
    adjustments = build_source_adjustments(source_summary)
    signal_adjustments = build_signal_adjustments(signal_summary)
    trait_adjustments = build_trait_adjustments(trait_summary)
    fast_adjustments = build_fast_feedback_adjustments(
        fast_feedback_summary
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
        signal_keys = [
            x
            for x in group["signal_key"].astype(str).tolist()
            if x and x != "nan"
        ]

        if "SHORT COVER" in sources and "ME HUNTER" in sources:
            best["source"] = "SHORT+ME"
            adaptive = adjustments.get("SHORT+ME", {})
            adaptive_bonus = float(adaptive.get("bonus", 0.0) or 0.0)
            confluence_bonus = _clip(8.0 + adaptive_bonus, 3.0, 14.0)
            best["source_score"] = min(
                100.0,
                float(group["source_score"].max()) + confluence_bonus,
            )
            best["adaptive_bonus"] = adaptive_bonus
            best["adaptive_confidence"] = str(
                adaptive.get("confidence", "BASE")
            )
            best["alert_tier"] = "🔥 CONFLUENCE"
            me_key = next((x for x in signal_keys if x.startswith("ME|")), "ME|OTHER")
            sc_key = next((x for x in signal_keys if x.startswith("SC|")), "SC|OTHER")
            best["signal_key"] = f"CONFLUENCE|{me_key.split('|', 1)[1]}|{sc_key.split('|', 1)[1]}"
        else:
            best["source"] = next(iter(sources)) if sources else ""
            adaptive = adjustments.get(str(best["source"]), {})
            adaptive_bonus = float(adaptive.get("bonus", 0.0) or 0.0)
            best["source_score"] = _clip(
                float(best.get("source_score", 0.0) or 0.0) + adaptive_bonus,
                0.0,
                100.0,
            )
            best["adaptive_bonus"] = adaptive_bonus
            best["adaptive_confidence"] = str(
                adaptive.get("confidence", "BASE")
            )
            if signal_keys:
                best["signal_key"] = signal_keys[0]

        state_adaptive = signal_adjustments.get(
            str(best.get("signal_key", "") or ""),
            {},
        )
        state_bonus = float(state_adaptive.get("bonus", 0.0) or 0.0)
        best["state_bonus"] = state_bonus
        best["state_confidence"] = str(
            state_adaptive.get("confidence", "BASE")
        )
        best["source_score"] = _clip(
            float(best.get("source_score", 0.0) or 0.0) + state_bonus,
            0.0,
            100.0,
        )

        trait_total = 0.0
        trait_confs = []
        for trait_col in ["trait_market", "trait_size", "trait_vol"]:
            trait_key = str(best.get(trait_col, "") or "")
            adj = trait_adjustments.get(trait_key, {})
            trait_total += float(adj.get("bonus", 0.0) or 0.0)
            conf = str(adj.get("confidence", "") or "")
            if conf:
                trait_confs.append(conf)
        trait_total = _clip(trait_total, -2.5, 2.5)
        best["trait_bonus"] = trait_total
        best["trait_confidence"] = (
            "ADAPTIVE"
            if "ADAPTIVE" in trait_confs
            else ("WARMING" if "WARMING" in trait_confs else ("LOW SAMPLE" if "LOW SAMPLE" in trait_confs else "BASE"))
        )
        best["source_score"] = _clip(
            float(best.get("source_score", 0.0) or 0.0) + trait_total,
            0.0,
            100.0,
        )

        fast_total = 0.0
        fast_confs = []

        fast_keys = [
            ("SOURCE", str(best.get("source", "") or "")),
            ("SETUP", str(best.get("signal_key", "") or "")),
            ("MARKET", str(best.get("trait_market", "") or "")),
            ("SIZE", str(best.get("trait_size", "") or "")),
            ("VOL", str(best.get("trait_vol", "") or "")),
        ]
        for dimension, key in fast_keys:
            if not key:
                continue
            adj = fast_adjustments.get(f"{dimension}:{key}", {})
            fast_total += float(adj.get("bonus", 0.0) or 0.0)
            conf = str(adj.get("confidence", "") or "")
            if conf:
                fast_confs.append(conf)

        fast_total = _clip(fast_total, -2.5, 2.5)
        best["fast_bonus"] = fast_total
        best["fast_confidence"] = (
            "ADAPTIVE"
            if "ADAPTIVE" in fast_confs
            else (
                "WARMING"
                if "WARMING" in fast_confs
                else (
                    "LOW SAMPLE"
                    if "LOW SAMPLE" in fast_confs
                    else (
                        "DATA BUILDING"
                        if "DATA BUILDING" in fast_confs
                        else "BASE"
                    )
                )
            )
        )
        best["source_score"] = _clip(
            float(best.get("source_score", 0.0) or 0.0) + fast_total,
            0.0,
            100.0,
        )

        if details:
            best["source_detail"] = " | ".join(dict.fromkeys(details))
        adaptive_bonus = _num(best.get("adaptive_bonus")) or 0.0
        adaptive_conf = str(best.get("adaptive_confidence", "BASE") or "BASE")
        if abs(adaptive_bonus) >= 0.05:
            best["source_detail"] = (
                str(best.get("source_detail", "") or "")
                + f" | Source {adaptive_bonus:+.1f} ({adaptive_conf})"
            ).strip(" |")
        state_bonus = _num(best.get("state_bonus")) or 0.0
        state_conf = str(best.get("state_confidence", "BASE") or "BASE")
        if abs(state_bonus) >= 0.05:
            best["source_detail"] = (
                str(best.get("source_detail", "") or "")
                + f" | State {state_bonus:+.1f} ({state_conf})"
            ).strip(" |")
        trait_bonus = _num(best.get("trait_bonus")) or 0.0
        trait_conf = str(best.get("trait_confidence", "BASE") or "BASE")
        if abs(trait_bonus) >= 0.05:
            best["source_detail"] = (
                str(best.get("source_detail", "") or "")
                + f" | Trait {trait_bonus:+.1f} ({trait_conf})"
            ).strip(" |")
        fast_bonus = _num(best.get("fast_bonus")) or 0.0
        fast_conf = str(best.get("fast_confidence", "BASE") or "BASE")
        if abs(fast_bonus) >= 0.05:
            best["source_detail"] = (
                str(best.get("source_detail", "") or "")
                + f" | Fast0D {fast_bonus:+.1f} ({fast_conf})"
            ).strip(" |")
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

from __future__ import annotations

from pathlib import Path

import pandas as pd

from me_screener_alerts import detect_screener_changes
from multiple_expansion_screener import build_daily_screener, candidate_only


HISTORY_FILE = Path("data/multiple_expansion/mex_history.csv")
UNIVERSE_CANDIDATES_FILE = Path(
    "data/multiple_expansion/me_universe_candidates.csv"
)
OUT_FILE = Path("data/multiple_expansion/me_screener_latest.csv")
CANDIDATE_FILE = Path("data/multiple_expansion/me_screener_candidates.csv")
CHANGE_FILE = Path("data/multiple_expansion/me_screener_changes.csv")
CHANGE_HISTORY_FILE = Path(
    "data/multiple_expansion/me_screener_change_history.csv"
)


def _load_previous_snapshot() -> pd.DataFrame:
    if not OUT_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(OUT_FILE, dtype={"ticker": str})
    except Exception:
        return pd.DataFrame()


def _load_change_history() -> pd.DataFrame:
    if not CHANGE_HISTORY_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(
            CHANGE_HISTORY_FILE,
            dtype={"ticker": str, "event_id": str},
        )
    except Exception:
        return pd.DataFrame()


def _merge_universe_metadata(history: pd.DataFrame) -> pd.DataFrame:
    if not UNIVERSE_CANDIDATES_FILE.exists():
        return history

    try:
        meta = pd.read_csv(UNIVERSE_CANDIDATES_FILE, dtype={"ticker": str})
    except Exception:
        return history

    if meta.empty or "ticker" not in meta.columns:
        return history

    keep = [
        c
        for c in [
            "ticker",
            "company_name",
            "market_name",
            "sector33_name",
            "sector17_name",
            "prefilter_rank",
            "prefilter_state",
            "coarse_score",
            "coarse_mlp",
            "coarse_velocity_pct",
            "coarse_acceleration_pct",
            "coarse_curvature_pct",
            "avg_turnover20",
            "pinned",
        ]
        if c in meta.columns
    ]
    meta = meta[keep].drop_duplicates("ticker", keep="last")

    out = history.copy()
    out["ticker"] = out["ticker"].astype(str).str.replace(".0", "", regex=False)
    meta["ticker"] = meta["ticker"].astype(str).str.replace(".0", "", regex=False)
    return out.merge(meta, on="ticker", how="left")


def _write_changes(
    previous: pd.DataFrame,
    current: pd.DataFrame,
) -> pd.DataFrame:
    changes = detect_screener_changes(
        previous,
        current,
        top_n=10,
        baseline_if_no_previous=True,
    )

    CHANGE_FILE.parent.mkdir(parents=True, exist_ok=True)

    if changes.empty:
        pd.DataFrame(
            columns=[
                "event_id",
                "market_date",
                "ticker",
                "company_name",
                "event_type",
                "severity",
                "previous_state",
                "current_state",
                "previous_decision",
                "current_decision",
                "previous_rank",
                "current_rank",
                "sw_score",
                "hist_edge_score",
                "fcf_engine_score",
                "change_reason",
            ]
        ).to_csv(CHANGE_FILE, index=False)
        print("[ME] no actionable state changes")
        return changes

    changes.to_csv(CHANGE_FILE, index=False)

    old_history = _load_change_history()
    if old_history.empty:
        merged = changes.copy()
    else:
        merged = pd.concat([old_history, changes], ignore_index=True)
        if "event_id" in merged.columns:
            merged = merged.drop_duplicates("event_id", keep="last")

    if "market_date" in merged.columns:
        merged["market_date"] = pd.to_datetime(
            merged["market_date"], errors="coerce"
        )
        merged = merged.sort_values(
            ["market_date", "event_priority", "ticker"],
            ascending=[False, False, True],
            na_position="last",
        )
        merged["market_date"] = merged["market_date"].dt.strftime("%Y-%m-%d")

    # Keep the repository artifact compact while retaining enough history
    # for UI review and notification dedupe.
    merged.head(2500).to_csv(CHANGE_HISTORY_FILE, index=False)
    print(
        f"[ME] wrote {CHANGE_FILE} ({len(changes)} changes), "
        f"history={len(merged.head(2500))}"
    )
    return changes


def main() -> int:
    if not HISTORY_FILE.exists():
        raise SystemExit(f"Missing {HISTORY_FILE}. Run MEX build first.")

    previous = _load_previous_snapshot()

    history = pd.read_csv(HISTORY_FILE, dtype={"ticker": str})
    if history.empty:
        raise SystemExit("MEX history is empty.")

    history = _merge_universe_metadata(history)

    screener = build_daily_screener(history)
    candidates = candidate_only(screener)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Important: compare BEFORE overwriting the previous committed snapshot.
    changes = _write_changes(previous, screener)

    screener.to_csv(OUT_FILE, index=False)
    candidates.to_csv(CANDIDATE_FILE, index=False)

    print(f"[ME] wrote {OUT_FILE} ({len(screener)} rows)")
    print(f"[ME] wrote {CANDIDATE_FILE} ({len(candidates)} candidates)")

    if not changes.empty:
        show_change_cols = [
            c
            for c in [
                "event_type",
                "ticker",
                "company_name",
                "previous_state",
                "current_state",
                "current_decision",
                "current_rank",
                "sw_score",
            ]
            if c in changes.columns
        ]
        print("[ME] actionable changes:")
        print(changes[show_change_cols].head(20).to_string(index=False))

    if not candidates.empty:
        show_cols = [
            c
            for c in [
                "screen_rank",
                "ticker",
                "company_name",
                "market_name",
                "prefilter_rank",
                "second_wave_state",
                "sw_decision",
                "sw_score",
                "hist_edge_score",
                "fcf_engine_score",
                "screen_reason",
            ]
            if c in candidates.columns
        ]
        print("[ME] current ranking:")
        print(candidates[show_cols].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

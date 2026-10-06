from __future__ import annotations

from pathlib import Path

import pandas as pd

from multiple_expansion_screener import build_daily_screener, candidate_only


HISTORY_FILE = Path("data/multiple_expansion/mex_history.csv")
UNIVERSE_CANDIDATES_FILE = Path(
    "data/multiple_expansion/me_universe_candidates.csv"
)
OUT_FILE = Path("data/multiple_expansion/me_screener_latest.csv")
CANDIDATE_FILE = Path("data/multiple_expansion/me_screener_candidates.csv")


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


def main() -> int:
    if not HISTORY_FILE.exists():
        raise SystemExit(f"Missing {HISTORY_FILE}. Run MEX build first.")

    history = pd.read_csv(HISTORY_FILE, dtype={"ticker": str})
    if history.empty:
        raise SystemExit("MEX history is empty.")

    history = _merge_universe_metadata(history)

    screener = build_daily_screener(history)
    candidates = candidate_only(screener)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    screener.to_csv(OUT_FILE, index=False)
    candidates.to_csv(CANDIDATE_FILE, index=False)

    print(f"[ME] wrote {OUT_FILE} ({len(screener)} rows)")
    print(f"[ME] wrote {CANDIDATE_FILE} ({len(candidates)} candidates)")

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
        print(candidates[show_cols].head(20).to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

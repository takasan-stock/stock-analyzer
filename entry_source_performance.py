from __future__ import annotations

from typing import Any

import pandas as pd


CANDIDATE_HISTORY_COLUMNS = [
    "monitor_date",
    "ticker",
    "name",
    "source",
    "source_detail",
    "source_score",
    "source_rank",
    "signal_key",
    "trait_market",
    "trait_size",
    "trait_vol",
    "first_seen_at",
    "last_seen_at",
    "last_status",
    "best_entry_score",
    "ready_detected",
    "ready_detected_at",
]

PERFORMANCE_COLUMNS = [
    "market_date",
    "ticker",
    "name",
    "source",
    "source_detail",
    "signal_key",
    "trait_market",
    "trait_size",
    "trait_vol",
    "entry_price",
    "entry_score",
    "first_detected_at",
    "ret_1d",
    "ret_3d",
    "ret_5d",
    "ret_10d",
    "mfe_10d",
    "mae_10d",
    "outcome_status",
    "last_updated",
]


def _ticker(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".T"):
        text = text[:-2]
    if text.endswith(".0"):
        text = text[:-2]
    return text


def _source(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"SHORT+ME", "ME HUNTER", "SHORT COVER"}:
        return text
    if "SHORT" in text and "ME" in text:
        return "SHORT+ME"
    if "ME" in text:
        return "ME HUNTER"
    if "SHORT" in text:
        return "SHORT COVER"
    return "LEGACY"


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_candidate_history(
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=CANDIDATE_HISTORY_COLUMNS)

    out = history.copy()
    for col in CANDIDATE_HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out["monitor_date"] = pd.to_datetime(
        out["monitor_date"], errors="coerce"
    ).dt.normalize()
    out["ticker"] = out["ticker"].map(_ticker)
    out["source"] = out["source"].map(_source)
    out["signal_key"] = out["signal_key"].fillna("").astype(str)
    for trait_col in ["trait_market", "trait_size", "trait_vol"]:
        out[trait_col] = out[trait_col].fillna("").astype(str)
    for col in ["trait_market", "trait_size", "trait_vol"]:
        out[col] = out[col].fillna("").astype(str)
    out["first_seen_at"] = pd.to_datetime(
        out["first_seen_at"], errors="coerce"
    )
    out["last_seen_at"] = pd.to_datetime(
        out["last_seen_at"], errors="coerce"
    )
    out["ready_detected_at"] = pd.to_datetime(
        out["ready_detected_at"], errors="coerce"
    )
    out["source_score"] = pd.to_numeric(
        out["source_score"], errors="coerce"
    )
    out["source_rank"] = pd.to_numeric(
        out["source_rank"], errors="coerce"
    )
    out["best_entry_score"] = pd.to_numeric(
        out["best_entry_score"], errors="coerce"
    )
    out["ready_detected"] = (
        out["ready_detected"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )

    out = out.dropna(subset=["monitor_date"])
    out = out[out["ticker"] != ""].copy()
    out = out.sort_values(
        ["monitor_date", "ticker", "source", "last_seen_at"],
        ascending=[False, True, True, False],
        na_position="last",
    )
    out = out.drop_duplicates(
        subset=["monitor_date", "ticker", "source"],
        keep="first",
    )
    return out[CANDIDATE_HISTORY_COLUMNS].reset_index(drop=True)


def update_candidate_history(
    history: pd.DataFrame | None,
    candidates: pd.DataFrame | None,
    status_rows: list[dict[str, Any]] | None,
    *,
    observed_at: Any,
) -> pd.DataFrame:
    base = normalize_candidate_history(history)
    if candidates is None or candidates.empty:
        return base

    observed = pd.Timestamp(observed_at)
    if observed.tzinfo is not None:
        observed = observed.tz_localize(None)
    monitor_date = observed.normalize()

    status_map: dict[str, dict[str, Any]] = {}
    for item in status_rows or []:
        ticker = _ticker(item.get("ticker"))
        if ticker:
            status_map[ticker] = item

    rows = []
    for _, candidate in candidates.iterrows():
        ticker = _ticker(candidate.get("ticker"))
        if not ticker:
            continue

        source = _source(candidate.get("source"))
        status_item = status_map.get(ticker, {})
        status = str(status_item.get("status", "") or "")
        entry_score = _num(status_item.get("score"))

        mask = pd.Series(False, index=base.index)
        if not base.empty:
            mask = (
                (base["monitor_date"] == monitor_date)
                & (base["ticker"] == ticker)
                & (base["source"] == source)
            )

        if mask.any():
            idx = base.index[mask][0]
            base.at[idx, "last_seen_at"] = observed
            base.at[idx, "last_status"] = status or base.at[idx, "last_status"]
            base.at[idx, "source_detail"] = (
                str(candidate.get("source_detail", "") or "")
                or base.at[idx, "source_detail"]
            )
            base.at[idx, "signal_key"] = (
                str(candidate.get("signal_key", "") or "")
                or base.at[idx, "signal_key"]
            )
            for trait_col in ["trait_market", "trait_size", "trait_vol"]:
                base.at[idx, trait_col] = (
                    str(candidate.get(trait_col, "") or "")
                    or base.at[idx, trait_col]
                )
            source_score = _num(candidate.get("source_score"))
            if source_score is not None:
                base.at[idx, "source_score"] = source_score
            source_rank = _num(candidate.get("source_rank"))
            if source_rank is not None:
                base.at[idx, "source_rank"] = source_rank

            prior_best = _num(base.at[idx, "best_entry_score"])
            if entry_score is not None:
                if prior_best is None or entry_score > prior_best:
                    base.at[idx, "best_entry_score"] = entry_score

            if status == "🟢 ENTRY READY":
                base.at[idx, "ready_detected"] = True
                if pd.isna(base.at[idx, "ready_detected_at"]):
                    base.at[idx, "ready_detected_at"] = observed
            continue

        ready = status == "🟢 ENTRY READY"
        rows.append(
            {
                "monitor_date": monitor_date,
                "ticker": ticker,
                "name": str(candidate.get("name", "") or ""),
                "source": source,
                "source_detail": str(
                    candidate.get("source_detail", "") or ""
                ),
                "source_score": _num(candidate.get("source_score")),
                "source_rank": _num(candidate.get("source_rank")),
                "signal_key": str(candidate.get("signal_key", "") or ""),
                "trait_market": str(candidate.get("trait_market", "") or ""),
                "trait_size": str(candidate.get("trait_size", "") or ""),
                "trait_vol": str(candidate.get("trait_vol", "") or ""),
                "first_seen_at": observed,
                "last_seen_at": observed,
                "last_status": status,
                "best_entry_score": entry_score,
                "ready_detected": ready,
                "ready_detected_at": observed if ready else pd.NaT,
            }
        )

    if rows:
        base = pd.concat([base, pd.DataFrame(rows)], ignore_index=True)

    return normalize_candidate_history(base)


def normalize_performance(
    frame: pd.DataFrame | None,
) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame(columns=PERFORMANCE_COLUMNS)

    out = frame.copy()
    for col in PERFORMANCE_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out["market_date"] = pd.to_datetime(
        out["market_date"], errors="coerce"
    ).dt.normalize()
    out["ticker"] = out["ticker"].map(_ticker)
    out["source"] = out["source"].map(_source)
    out["signal_key"] = out["signal_key"].fillna("").astype(str)
    for col in [
        "entry_price",
        "entry_score",
        "ret_1d",
        "ret_3d",
        "ret_5d",
        "ret_10d",
        "mfe_10d",
        "mae_10d",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in ["first_detected_at", "last_updated"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    out = out.dropna(subset=["market_date"])
    out = out[out["ticker"] != ""].copy()
    out = out.sort_values(
        ["market_date", "ticker", "source", "first_detected_at"],
        ascending=[False, True, True, True],
        na_position="last",
    )
    out = out.drop_duplicates(
        subset=["market_date", "ticker", "source"],
        keep="first",
    )
    return out[PERFORMANCE_COLUMNS].reset_index(drop=True)


def _daily_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()

    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    idx = pd.to_datetime(out.index, errors="coerce")
    out = out.loc[~idx.isna()].copy()
    out.index = idx[~idx.isna()].normalize()
    return out.sort_index()


def build_ready_performance(
    notifications: pd.DataFrame | None,
    price_frames: dict[str, pd.DataFrame],
    *,
    prior: pd.DataFrame | None = None,
    updated_at: Any | None = None,
) -> pd.DataFrame:
    """Build forward performance from actual intraday ENTRY READY events.

    Entry price is the current price captured when ENTRY READY was first
    detected. Horizon returns use the close of the 1st/3rd/5th/10th trading
    session including the signal day. MFE/MAE use daily highs/lows through the
    first 10 trading sessions. This keeps ME, Short Cover and Confluence on the
    exact same entry convention.
    """
    base = normalize_performance(prior)
    if notifications is None or notifications.empty:
        return base

    ready = notifications.copy()
    ready = ready[
        ready.get("entry_status", pd.Series("", index=ready.index))
        .fillna("")
        .astype(str)
        .eq("🟢 ENTRY READY")
    ].copy()
    if ready.empty:
        return base

    ready["market_date"] = pd.to_datetime(
        ready["market_date"], errors="coerce"
    ).dt.normalize()
    ready["ticker"] = ready["ticker"].map(_ticker)
    ready["source"] = ready.get(
        "source", pd.Series("", index=ready.index)
    ).map(_source)
    ready["first_detected_at"] = pd.to_datetime(
        ready.get("first_detected_at"), errors="coerce"
    )
    ready = ready.sort_values("first_detected_at")
    ready = ready.drop_duplicates(
        subset=["market_date", "ticker", "source"],
        keep="first",
    )

    now_ts = pd.Timestamp(updated_at or pd.Timestamp.now())
    if now_ts.tzinfo is not None:
        now_ts = now_ts.tz_localize(None)

    rows: list[dict[str, Any]] = []

    for _, row in ready.iterrows():
        ticker = _ticker(row.get("ticker"))
        market_date = pd.to_datetime(
            row.get("market_date"), errors="coerce"
        )
        entry_price = _num(row.get("current_price"))
        if not ticker or pd.isna(market_date) or entry_price is None:
            continue
        if entry_price <= 0:
            continue

        frame = _daily_frame(price_frames.get(ticker))
        ret = {1: None, 3: None, 5: None, 10: None}
        mfe = None
        mae = None

        if not frame.empty:
            positions = [
                i
                for i, dt in enumerate(frame.index)
                if dt >= pd.Timestamp(market_date).normalize()
            ]
            if positions:
                entry_pos = positions[0]
                for horizon in (1, 3, 5, 10):
                    exit_pos = entry_pos + horizon - 1
                    if exit_pos < len(frame):
                        close_value = _num(
                            frame.iloc[exit_pos].get("Close")
                        )
                        if close_value is not None:
                            ret[horizon] = (
                                close_value / entry_price - 1.0
                            ) * 100.0

                end_pos = min(len(frame) - 1, entry_pos + 9)
                future = frame.iloc[entry_pos : end_pos + 1]
                if not future.empty:
                    high_col = (
                        future["High"]
                        if "High" in future.columns
                        else future.get("Close")
                    )
                    low_col = (
                        future["Low"]
                        if "Low" in future.columns
                        else future.get("Close")
                    )
                    if high_col is not None and high_col.notna().any():
                        mfe = (
                            float(high_col.max()) / entry_price - 1.0
                        ) * 100.0
                    if low_col is not None and low_col.notna().any():
                        mae = (
                            float(low_col.min()) / entry_price - 1.0
                        ) * 100.0

        if ret[10] is not None:
            outcome_status = "✅ 10D COMPLETE"
        elif ret[5] is not None:
            outcome_status = "📈 5D"
        elif ret[3] is not None:
            outcome_status = "📊 3D"
        elif ret[1] is not None:
            outcome_status = "🌱 1D"
        else:
            outcome_status = "⏳ TRACKING"

        rows.append(
            {
                "market_date": pd.Timestamp(market_date).normalize(),
                "ticker": ticker,
                "name": str(row.get("name", "") or ""),
                "source": _source(row.get("source")),
                "source_detail": str(
                    row.get("source_detail", "") or ""
                ),
                "signal_key": str(row.get("signal_key", "") or ""),
                "trait_market": str(row.get("trait_market", "") or ""),
                "trait_size": str(row.get("trait_size", "") or ""),
                "trait_vol": str(row.get("trait_vol", "") or ""),
                "entry_price": entry_price,
                "entry_score": _num(row.get("entry_score")),
                "first_detected_at": row.get("first_detected_at"),
                "ret_1d": ret[1],
                "ret_3d": ret[3],
                "ret_5d": ret[5],
                "ret_10d": ret[10],
                "mfe_10d": mfe,
                "mae_10d": mae,
                "outcome_status": outcome_status,
                "last_updated": now_ts,
            }
        )

    if rows:
        incoming = normalize_performance(pd.DataFrame(rows))
        base = normalize_performance(
            pd.concat([incoming, base], ignore_index=True)
        )

    return base


def _confidence(samples: int) -> str:
    if samples >= 30:
        return "STRONG"
    if samples >= 20:
        return "GOOD"
    if samples >= 10:
        return "MEDIUM"
    if samples >= 5:
        return "LOW"
    return "VERY LOW"


def summarize_source_performance(
    candidate_history: pd.DataFrame | None,
    performance: pd.DataFrame | None,
    notifications: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Compare ME, Short Cover and confluence with prospective live data."""
    candidates = normalize_candidate_history(candidate_history)
    perf = normalize_performance(performance)

    notifications = (
        pd.DataFrame()
        if notifications is None
        else notifications.copy()
    )
    if not notifications.empty:
        notifications["market_date"] = pd.to_datetime(
            notifications["market_date"], errors="coerce"
        ).dt.normalize()
        notifications["ticker"] = notifications["ticker"].map(_ticker)
        notifications["source"] = notifications.get(
            "source", pd.Series("", index=notifications.index)
        ).map(_source)

    sources = ["ME HUNTER", "SHORT COVER", "SHORT+ME"]
    rows: list[dict[str, Any]] = []

    for source in sources:
        cg = candidates[candidates["source"] == source].copy()
        pg = perf[perf["source"] == source].copy()

        candidate_n = int(len(cg))
        ready_n = int(cg["ready_detected"].fillna(False).sum())
        ready_rate = (
            ready_n / candidate_n * 100.0
            if candidate_n > 0
            else None
        )

        confirmed_n = 0
        weakening_n = 0
        exit_watch_n = 0
        if not notifications.empty:
            ng = notifications[
                notifications["source"] == source
            ].copy()
            for status, key in [
                ("🟢 ENTRY CONFIRMED", "confirmed"),
                ("🟡 WEAKENING", "weakening"),
                ("🔴 EXIT WATCH", "exit"),
            ]:
                unique_n = ng[
                    ng.get(
                        "entry_status",
                        pd.Series("", index=ng.index),
                    ).astype(str).eq(status)
                ][["market_date", "ticker"]].drop_duplicates().shape[0]
                if key == "confirmed":
                    confirmed_n = int(unique_n)
                elif key == "weakening":
                    weakening_n = int(unique_n)
                else:
                    exit_watch_n = int(unique_n)

        def avg(col: str):
            vals = pd.to_numeric(
                pg.get(col), errors="coerce"
            ).dropna()
            return float(vals.mean()) if not vals.empty else None

        def win(col: str):
            vals = pd.to_numeric(
                pg.get(col), errors="coerce"
            ).dropna()
            return (
                float((vals > 0).mean() * 100.0)
                if not vals.empty
                else None
            )

        sample_5d = int(
            pd.to_numeric(
                pg.get("ret_5d"), errors="coerce"
            ).notna().sum()
        )

        rows.append(
            {
                "source": source,
                "candidate_days": candidate_n,
                "entry_ready": ready_n,
                "entry_ready_rate": ready_rate,
                "entry_confirmed": confirmed_n,
                "confirmed_per_ready": (
                    confirmed_n / ready_n * 100.0
                    if ready_n > 0
                    else None
                ),
                "weakening": weakening_n,
                "exit_watch": exit_watch_n,
                "tracked_entries": int(len(pg)),
                "win_1d": win("ret_1d"),
                "win_3d": win("ret_3d"),
                "win_5d": win("ret_5d"),
                "win_10d": win("ret_10d"),
                "avg_1d": avg("ret_1d"),
                "avg_3d": avg("ret_3d"),
                "avg_5d": avg("ret_5d"),
                "avg_10d": avg("ret_10d"),
                "avg_mfe_10d": avg("mfe_10d"),
                "avg_mae_10d": avg("mae_10d"),
                "sample_5d": sample_5d,
                "confidence": _confidence(sample_5d),
            }
        )

    return pd.DataFrame(rows)


def summarize_signal_performance(
    candidate_history: pd.DataFrame | None,
    performance: pd.DataFrame | None,
) -> pd.DataFrame:
    """Compare prospective live performance by concrete setup/state."""
    candidates = normalize_candidate_history(candidate_history)
    perf = normalize_performance(performance)

    keys = sorted(
        {
            str(x)
            for x in pd.concat(
                [
                    candidates.get("signal_key", pd.Series(dtype=str)),
                    perf.get("signal_key", pd.Series(dtype=str)),
                ],
                ignore_index=True,
            )
            .fillna("")
            .astype(str)
            .tolist()
            if str(x).strip()
        }
    )

    rows = []
    for key in keys:
        cg = candidates[candidates["signal_key"] == key].copy()
        pg = perf[perf["signal_key"] == key].copy()

        candidate_n = int(len(cg))
        ready_n = int(cg["ready_detected"].fillna(False).sum())
        ready_rate = ready_n / candidate_n * 100.0 if candidate_n > 0 else None

        def avg(col: str):
            vals = pd.to_numeric(pg.get(col), errors="coerce").dropna()
            return float(vals.mean()) if not vals.empty else None

        def win(col: str):
            vals = pd.to_numeric(pg.get(col), errors="coerce").dropna()
            return float((vals > 0).mean() * 100.0) if not vals.empty else None

        sample_5d = int(pd.to_numeric(pg.get("ret_5d"), errors="coerce").notna().sum())

        rows.append(
            {
                "signal_key": key,
                "candidate_days": candidate_n,
                "entry_ready": ready_n,
                "entry_ready_rate": ready_rate,
                "tracked_entries": int(len(pg)),
                "win_5d": win("ret_5d"),
                "avg_5d": avg("ret_5d"),
                "avg_10d": avg("ret_10d"),
                "avg_mfe_10d": avg("mfe_10d"),
                "avg_mae_10d": avg("mae_10d"),
                "sample_5d": sample_5d,
                "confidence": _confidence(sample_5d),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "signal_key",
                "candidate_days",
                "entry_ready",
                "entry_ready_rate",
                "tracked_entries",
                "win_5d",
                "avg_5d",
                "avg_10d",
                "avg_mfe_10d",
                "avg_mae_10d",
                "sample_5d",
                "confidence",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["sample_5d", "avg_5d", "win_5d", "signal_key"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def summarize_trait_performance(
    candidate_history: pd.DataFrame | None,
    performance: pd.DataFrame | None,
) -> pd.DataFrame:
    """Aggregate prospective results by market, size and volatility traits."""
    candidates = normalize_candidate_history(candidate_history)
    perf = normalize_performance(performance)

    rows = []
    for trait_col in ["trait_market", "trait_size", "trait_vol"]:
        keys = sorted(
            {
                str(x)
                for x in pd.concat(
                    [
                        candidates.get(trait_col, pd.Series(dtype=str)),
                        perf.get(trait_col, pd.Series(dtype=str)),
                    ],
                    ignore_index=True,
                ).fillna("").astype(str)
                if str(x).strip() and "UNKNOWN" not in str(x) and "OTHER" not in str(x)
            }
        )

        for key in keys:
            cg = candidates[candidates[trait_col] == key].copy()
            pg = perf[perf[trait_col] == key].copy()
            candidate_n = int(len(cg))
            ready_n = int(cg["ready_detected"].fillna(False).sum())
            ready_rate = ready_n / candidate_n * 100.0 if candidate_n > 0 else None

            def avg(col: str):
                vals = pd.to_numeric(pg.get(col), errors="coerce").dropna()
                return float(vals.mean()) if not vals.empty else None

            def win(col: str):
                vals = pd.to_numeric(pg.get(col), errors="coerce").dropna()
                return float((vals > 0).mean() * 100.0) if not vals.empty else None

            sample_5d = int(pd.to_numeric(pg.get("ret_5d"), errors="coerce").notna().sum())

            rows.append(
                {
                    "trait_axis": trait_col.replace("trait_", "").upper(),
                    "trait_key": key,
                    "candidate_days": candidate_n,
                    "entry_ready": ready_n,
                    "entry_ready_rate": ready_rate,
                    "tracked_entries": int(len(pg)),
                    "win_5d": win("ret_5d"),
                    "avg_5d": avg("ret_5d"),
                    "avg_10d": avg("ret_10d"),
                    "avg_mfe_10d": avg("mfe_10d"),
                    "avg_mae_10d": avg("mae_10d"),
                    "sample_5d": sample_5d,
                    "confidence": _confidence(sample_5d),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "trait_axis", "trait_key", "candidate_days", "entry_ready",
                "entry_ready_rate", "tracked_entries", "win_5d", "avg_5d",
                "avg_10d", "avg_mfe_10d", "avg_mae_10d", "sample_5d",
                "confidence",
            ]
        )

    return pd.DataFrame(rows).sort_values(
        ["trait_axis", "sample_5d", "avg_5d", "trait_key"],
        ascending=[True, False, False, True],
        na_position="last",
    ).reset_index(drop=True)

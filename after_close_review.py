from __future__ import annotations

import re
from typing import Any

import pandas as pd


DAILY_REVIEW_COLUMNS = [
    "review_date",
    "ticker",
    "name",
    "source",
    "signal_key",
    "trait_market",
    "trait_size",
    "trait_vol",
    "entry_price",
    "close_price",
    "close_return_pct",
    "mfe_pct",
    "mae_pct",
    "latest_status",
    "carryover",
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
        out = float(value)
        if not pd.notna(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _daily_frame(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    idx = pd.to_datetime(out.index, errors="coerce")
    keep = ~idx.isna()
    out = out.loc[keep].copy()
    out.index = idx[keep].normalize()
    return out.sort_index()


def _latest_status_for_ticker(rows: pd.DataFrame) -> str:
    if rows.empty:
        return ""
    work = rows.copy()
    work["_dt"] = pd.to_datetime(
        work.get("first_detected_at"),
        errors="coerce",
    )
    work = work.sort_values("_dt", na_position="first")
    return str(work.iloc[-1].get("entry_status", "") or "")


def _status_priority(status: str) -> int:
    return {
        "🟢 ENTRY CONFIRMED": 5,
        "🟢 ENTRY READY": 4,
        "🟦 MONITORING": 3,
        "🟡 WEAKENING": 2,
        "🔴 EXIT WATCH": 1,
        "🔴 CANCEL": 0,
    }.get(status, 0)


def _carryover_from_status(status: str) -> bool:
    return status in {
        "🟢 ENTRY CONFIRMED",
        "🟢 ENTRY READY",
        "🟦 MONITORING",
    }


def build_after_close_review(
    notifications: pd.DataFrame | None,
    daily_prices: dict[str, pd.DataFrame],
    *,
    review_date: Any,
    top_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Build a same-day Entry Hunter review from actual ENTRY READY prices.

    Returns use the signal's first ENTRY READY price as the denominator and
    the official daily close/high/low for the review date as the outcome.
    This makes the after-close panel independent from intraday follow-up
    polling frequency.
    """
    date = pd.Timestamp(review_date).normalize()
    notes = pd.DataFrame() if notifications is None else notifications.copy()

    if notes.empty:
        return {
            "review_date": date.strftime("%Y-%m-%d"),
            "ready_count": 0,
            "confirmed_count": 0,
            "weakening_count": 0,
            "exit_watch_count": 0,
            "carryover_count": 0,
            "top3_avg_return_pct": None,
            "avg_close_return_pct": None,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "price_coverage": 0,
            "tracked_entries": 0,
            "rows": [],
        }

    notes["market_date"] = pd.to_datetime(
        notes.get("market_date"),
        errors="coerce",
    ).dt.normalize()
    notes["first_detected_at"] = pd.to_datetime(
        notes.get("first_detected_at"),
        errors="coerce",
    )
    notes["ticker"] = notes.get(
        "ticker",
        pd.Series("", index=notes.index),
    ).map(_ticker)
    notes = notes[notes["market_date"] == date].copy()

    if notes.empty:
        return build_after_close_review(
            pd.DataFrame(),
            daily_prices,
            review_date=date,
            top_tickers=top_tickers,
        )

    ready = notes[
        notes.get(
            "entry_status",
            pd.Series("", index=notes.index),
        ).astype(str).eq("🟢 ENTRY READY")
    ].copy()
    ready = ready.sort_values("first_detected_at")
    ready = ready.drop_duplicates("ticker", keep="first")

    confirmed_count = int(
        notes[
            notes["entry_status"].astype(str).eq("🟢 ENTRY CONFIRMED")
        ]["ticker"].nunique()
    )
    weakening_count = int(
        notes[
            notes["entry_status"].astype(str).eq("🟡 WEAKENING")
        ]["ticker"].nunique()
    )
    exit_watch_count = int(
        notes[
            notes["entry_status"].astype(str).eq("🔴 EXIT WATCH")
        ]["ticker"].nunique()
    )

    rows: list[dict[str, Any]] = []
    for _, entry in ready.iterrows():
        ticker = _ticker(entry.get("ticker"))
        entry_price = _num(entry.get("current_price"))
        if not ticker or entry_price is None or entry_price <= 0:
            continue

        ticker_rows = notes[notes["ticker"] == ticker].copy()
        latest_status = _latest_status_for_ticker(ticker_rows)
        frame = _daily_frame(daily_prices.get(ticker))

        close_price = None
        close_return = None
        mfe = None
        mae = None

        if not frame.empty and date in frame.index:
            bar = frame.loc[date]
            if isinstance(bar, pd.DataFrame):
                bar = bar.iloc[-1]

            close_price = _num(bar.get("Close"))
            high_price = _num(bar.get("High"))
            low_price = _num(bar.get("Low"))

            if close_price is not None:
                close_return = (close_price / entry_price - 1.0) * 100.0
            if high_price is not None:
                mfe = (high_price / entry_price - 1.0) * 100.0
            if low_price is not None:
                mae = (low_price / entry_price - 1.0) * 100.0

        rows.append(
            {
                "review_date": date.strftime("%Y-%m-%d"),
                "ticker": ticker,
                "name": str(entry.get("name", "") or ""),
                "source": str(entry.get("source", "") or ""),
                "signal_key": str(entry.get("signal_key", "") or ""),
                "trait_market": str(entry.get("trait_market", "") or ""),
                "trait_size": str(entry.get("trait_size", "") or ""),
                "trait_vol": str(entry.get("trait_vol", "") or ""),
                "entry_price": entry_price,
                "close_price": close_price,
                "close_return_pct": close_return,
                "mfe_pct": mfe,
                "mae_pct": mae,
                "latest_status": latest_status,
                "carryover": _carryover_from_status(latest_status),
            }
        )

    detail = pd.DataFrame(rows, columns=DAILY_REVIEW_COLUMNS)
    if detail.empty:
        return {
            "review_date": date.strftime("%Y-%m-%d"),
            "ready_count": int(ready["ticker"].nunique()),
            "confirmed_count": confirmed_count,
            "weakening_count": weakening_count,
            "exit_watch_count": exit_watch_count,
            "carryover_count": 0,
            "top3_avg_return_pct": None,
            "avg_close_return_pct": None,
            "avg_mfe_pct": None,
            "avg_mae_pct": None,
            "price_coverage": 0,
            "tracked_entries": 0,
            "rows": [],
        }

    for col in ["close_return_pct", "mfe_pct", "mae_pct"]:
        detail[col] = pd.to_numeric(detail[col], errors="coerce")

    priced = detail["close_return_pct"].notna()

    top_set = {
        _ticker(x)
        for x in (top_tickers or [])
        if _ticker(x)
    }
    top_returns = detail[
        detail["ticker"].isin(top_set)
    ]["close_return_pct"].dropna()

    def mean_or_none(series: pd.Series):
        values = pd.to_numeric(series, errors="coerce").dropna()
        return float(values.mean()) if not values.empty else None

    return {
        "review_date": date.strftime("%Y-%m-%d"),
        "ready_count": int(ready["ticker"].nunique()),
        "confirmed_count": confirmed_count,
        "weakening_count": weakening_count,
        "exit_watch_count": exit_watch_count,
        "carryover_count": int(detail["carryover"].fillna(False).sum()),
        "top3_avg_return_pct": mean_or_none(top_returns),
        "avg_close_return_pct": mean_or_none(detail["close_return_pct"]),
        "avg_mfe_pct": mean_or_none(detail["mfe_pct"]),
        "avg_mae_pct": mean_or_none(detail["mae_pct"]),
        "price_coverage": int(priced.sum()),
        "tracked_entries": int(len(detail)),
        "rows": detail.to_dict(orient="records"),
    }


def normalize_after_close_history(
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    if history is None or history.empty:
        return pd.DataFrame(columns=DAILY_REVIEW_COLUMNS)

    out = history.copy()
    for col in DAILY_REVIEW_COLUMNS:
        if col not in out.columns:
            out[col] = None

    out["review_date"] = pd.to_datetime(
        out["review_date"], errors="coerce"
    ).dt.normalize()
    out["ticker"] = out["ticker"].map(_ticker)
    for col in [
        "entry_price",
        "close_price",
        "close_return_pct",
        "mfe_pct",
        "mae_pct",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    for col in [
        "source",
        "signal_key",
        "trait_market",
        "trait_size",
        "trait_vol",
        "latest_status",
    ]:
        out[col] = out[col].fillna("").astype(str)

    out["carryover"] = (
        out["carryover"]
        .fillna(False)
        .astype(str)
        .str.lower()
        .isin({"true", "1", "yes"})
    )
    out = out.dropna(subset=["review_date"])
    out = out[out["ticker"] != ""].copy()
    out = out.sort_values(
        ["review_date", "ticker"],
        ascending=[False, True],
    )
    out = out.drop_duplicates(
        subset=["review_date", "ticker"],
        keep="first",
    )
    return out[DAILY_REVIEW_COLUMNS].reset_index(drop=True)


def update_after_close_history(
    history: pd.DataFrame | None,
    review_rows: pd.DataFrame | None,
    *,
    max_rows: int = 5000,
) -> pd.DataFrame:
    base = normalize_after_close_history(history)
    incoming = normalize_after_close_history(review_rows)
    if incoming.empty:
        return base

    merged = pd.concat([incoming, base], ignore_index=True)
    merged = normalize_after_close_history(merged)
    return merged.head(max(1, int(max_rows))).reset_index(drop=True)


def _feedback_confidence(samples: int) -> str:
    if samples >= 30:
        return "ADAPTIVE"
    if samples >= 15:
        return "WARMING"
    if samples >= 8:
        return "LOW SAMPLE"
    return "DATA BUILDING"


def _feedback_bonus(
    group: pd.DataFrame,
    *,
    cap: float,
    min_samples: int = 8,
    full_samples: int = 30,
) -> tuple[float, str, int, float | None, float | None, float | None]:
    returns = pd.to_numeric(
        group["close_return_pct"], errors="coerce"
    ).dropna()
    n = int(len(returns))
    if n == 0:
        return 0.0, "DATA BUILDING", 0, None, None, None

    win_rate = float((returns > 0).mean() * 100.0)
    avg_return = float(returns.mean())
    mfe = pd.to_numeric(group["mfe_pct"], errors="coerce").dropna()
    mae = pd.to_numeric(group["mae_pct"], errors="coerce").dropna()
    avg_mfe = float(mfe.mean()) if not mfe.empty else None
    avg_mae = float(mae.mean()) if not mae.empty else None

    if n < min_samples:
        return 0.0, "DATA BUILDING", n, win_rate, avg_return, avg_mfe

    win_component = max(-1.0, min(1.0, (win_rate - 50.0) / 20.0))
    return_component = max(-1.0, min(1.0, avg_return / 3.0))

    rr_component = 0.0
    if (
        avg_mfe is not None
        and avg_mae is not None
        and abs(avg_mae) >= 0.25
    ):
        rr = avg_mfe / abs(avg_mae)
        rr_component = max(-1.0, min(1.0, (rr - 1.5) / 1.5))

    raw = (
        win_component * 0.45
        + return_component * 0.35
        + rr_component * 0.20
    )
    shrink = max(
        0.0,
        min(
            1.0,
            (n - min_samples + 1.0)
            / max(1.0, full_samples - min_samples + 1.0),
        ),
    )
    bonus = max(-cap, min(cap, raw * cap * shrink))
    return (
        round(float(bonus), 3),
        _feedback_confidence(n),
        n,
        win_rate,
        avg_return,
        avg_mfe,
    )


def summarize_after_close_feedback(
    history: pd.DataFrame | None,
) -> pd.DataFrame:
    """Create fast next-session feedback by source/setup/traits.

    This is deliberately weaker than 5-day adaptive learning. It is meant to
    nudge tomorrow's priority after the same-day close, not replace the
    slower forward-return evidence.
    """
    hist = normalize_after_close_history(history)
    columns = [
        "dimension",
        "key",
        "sample_0d",
        "win_0d",
        "avg_0d",
        "avg_mfe_0d",
        "fast_bonus",
        "confidence",
    ]
    if hist.empty:
        return pd.DataFrame(columns=columns)

    specs = [
        ("SOURCE", "source", 1.5),
        ("SETUP", "signal_key", 0.75),
        ("MARKET", "trait_market", 0.25),
        ("SIZE", "trait_size", 0.25),
        ("VOL", "trait_vol", 0.25),
    ]

    rows = []
    for dimension, col, cap in specs:
        if col not in hist.columns:
            continue
        values = hist[col].fillna("").astype(str)
        for key in sorted(x for x in values.unique() if x):
            if "UNKNOWN" in key or key.endswith("|OTHER"):
                continue
            group = hist[values == key].copy()
            (
                bonus,
                confidence,
                n,
                win_rate,
                avg_return,
                avg_mfe,
            ) = _feedback_bonus(group, cap=cap)
            rows.append(
                {
                    "dimension": dimension,
                    "key": key,
                    "sample_0d": n,
                    "win_0d": win_rate,
                    "avg_0d": avg_return,
                    "avg_mfe_0d": avg_mfe,
                    "fast_bonus": bonus,
                    "confidence": confidence,
                }
            )

    if not rows:
        return pd.DataFrame(columns=columns)

    return pd.DataFrame(rows, columns=columns).sort_values(
        ["dimension", "sample_0d", "fast_bonus", "key"],
        ascending=[True, False, False, True],
    ).reset_index(drop=True)

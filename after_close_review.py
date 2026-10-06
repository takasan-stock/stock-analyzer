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

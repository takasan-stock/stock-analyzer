from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from after_close_review import (
    build_after_close_review,
    normalize_after_close_history,
    summarize_after_close_feedback,
    update_after_close_history,
)


DATA_DIR = ROOT / "data"
NOTIFICATION_FILE = DATA_DIR / "short_cover_entry_notifications.csv"
COMMAND_CENTER_FILE = DATA_DIR / "daily_command_center_latest.csv"
OUT_JSON = DATA_DIR / "after_close_review_latest.json"
OUT_CSV = DATA_DIR / "after_close_review_latest.csv"
HISTORY_CSV = DATA_DIR / "after_close_review_history.csv"
FEEDBACK_CSV = DATA_DIR / "after_close_feedback_summary.csv"



def _load_history() -> pd.DataFrame:
    if not HISTORY_CSV.exists():
        return normalize_after_close_history(None)
    try:
        return normalize_after_close_history(
            pd.read_csv(
                HISTORY_CSV,
                dtype={"ticker": str},
                encoding="utf-8-sig",
            )
        )
    except Exception:
        return normalize_after_close_history(None)


def _load_notifications() -> pd.DataFrame:
    if not NOTIFICATION_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(
            NOTIFICATION_FILE,
            dtype={"ticker": str},
            encoding="utf-8-sig",
        )
    except Exception:
        return pd.DataFrame()


def _load_top_tickers() -> list[str]:
    if not COMMAND_CENTER_FILE.exists():
        return []
    try:
        frame = pd.read_csv(
            COMMAND_CENTER_FILE,
            dtype={"ticker": str},
            encoding="utf-8-sig",
        )
    except Exception:
        return []

    if frame.empty or "ticker" not in frame.columns:
        return []

    if "rank" in frame.columns:
        frame["rank"] = pd.to_numeric(frame["rank"], errors="coerce")
        frame = frame.sort_values("rank", na_position="last")

    return (
        frame["ticker"]
        .dropna()
        .astype(str)
        .str.replace(".0", "", regex=False)
        .head(3)
        .tolist()
    )


def _download_daily(ticker: str) -> pd.DataFrame:
    symbol = ticker if "." in ticker else f"{ticker}.T"
    frame = yf.download(
        symbol,
        period="10d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if isinstance(frame.columns, pd.MultiIndex):
        frame.columns = frame.columns.get_level_values(0)
    return frame


def main() -> int:
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    review_date = now.tz_localize(None).normalize()

    notifications = _load_notifications()
    top_tickers = _load_top_tickers()

    today_tickers: list[str] = []
    if not notifications.empty and "market_date" in notifications.columns:
        market_dates = pd.to_datetime(
            notifications["market_date"],
            errors="coerce",
        ).dt.normalize()
        today = notifications.loc[
            market_dates == review_date,
            "ticker",
        ]
        today_tickers = (
            today.dropna()
            .astype(str)
            .str.replace(".0", "", regex=False)
            .unique()
            .tolist()
        )

    price_frames: dict[str, pd.DataFrame] = {}
    for ticker in today_tickers:
        try:
            price_frames[str(ticker)] = _download_daily(str(ticker))
        except Exception as exc:
            print(
                f"[AFTER-CLOSE] {ticker}: "
                f"{type(exc).__name__}: {exc}"
            )

    review = build_after_close_review(
        notifications,
        price_frames,
        review_date=review_date,
        top_tickers=top_tickers,
    )
    review["built_at"] = now.isoformat()

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            review,
            ensure_ascii=False,
            indent=2,
            default=str,
        ),
        encoding="utf-8",
    )

    rows = review.get("rows", [])
    review_rows = pd.DataFrame(rows)
    review_rows.to_csv(
        OUT_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    history = update_after_close_history(
        _load_history(),
        review_rows,
    )
    history.to_csv(
        HISTORY_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    feedback = summarize_after_close_feedback(history)
    feedback.to_csv(
        FEEDBACK_CSV,
        index=False,
        encoding="utf-8-sig",
    )

    print(
        "[AFTER-CLOSE] "
        f"date={review.get('review_date')} "
        f"ready={review.get('ready_count')} "
        f"confirmed={review.get('confirmed_count')} "
        f"avg_return={review.get('avg_close_return_pct')} "
        f"top3_avg={review.get('top3_avg_return_pct')}"
    )
    print(f"[AFTER-CLOSE] wrote {OUT_JSON}")
    print(f"[AFTER-CLOSE] wrote {OUT_CSV}")
    print(f"[AFTER-CLOSE] wrote {HISTORY_CSV}")
    print(f"[AFTER-CLOSE] wrote {FEEDBACK_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

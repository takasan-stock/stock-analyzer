from __future__ import annotations

from datetime import date, datetime, time, timedelta
from typing import Any

import jpholiday
import pandas as pd


TSE_CLOSE = time(15, 30)


def _as_jst(value: Any | None = None) -> pd.Timestamp:
    ts = (
        pd.Timestamp.now(tz="Asia/Tokyo")
        if value is None
        else pd.Timestamp(value)
    )
    if ts.tzinfo is None:
        return ts.tz_localize("Asia/Tokyo")
    return ts.tz_convert("Asia/Tokyo")


def is_tse_business_day(value: date | datetime | pd.Timestamp) -> bool:
    d = pd.Timestamp(value).date()
    if d.weekday() >= 5:
        return False
    if d.month == 12 and d.day == 31:
        return False
    if d.month == 1 and d.day in {1, 2, 3}:
        return False
    return not jpholiday.is_holiday(d)


def previous_tse_session(value: date | datetime | pd.Timestamp) -> date:
    d = pd.Timestamp(value).date() - timedelta(days=1)
    while not is_tse_business_day(d):
        d -= timedelta(days=1)
    return d


def next_tse_session(value: date | datetime | pd.Timestamp) -> date:
    d = pd.Timestamp(value).date() + timedelta(days=1)
    while not is_tse_business_day(d):
        d += timedelta(days=1)
    return d


def latest_completed_tse_session(
    value: Any | None = None,
    *,
    close_time: time = TSE_CLOSE,
) -> date:
    ts = _as_jst(value)
    d = ts.date()

    if is_tse_business_day(d) and ts.time() >= close_time:
        return d

    cursor = d
    if is_tse_business_day(cursor) and ts.time() < close_time:
        cursor -= timedelta(days=1)

    while not is_tse_business_day(cursor):
        cursor -= timedelta(days=1)
    return cursor


def next_tse_session_from_now(value: Any | None = None) -> date:
    latest = latest_completed_tse_session(value)
    return next_tse_session(latest)

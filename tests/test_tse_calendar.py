from datetime import datetime
from zoneinfo import ZoneInfo

from tse_calendar import (
    is_tse_business_day,
    latest_completed_tse_session,
    next_tse_session,
)


def test_next_tse_session_skips_weekend_and_sports_day_2026():
    assert next_tse_session("2026-10-09") == datetime(2026, 10, 13).date()


def test_latest_completed_session_on_sunday_is_previous_friday():
    ts = datetime(2026, 10, 11, 18, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert latest_completed_tse_session(ts) == datetime(2026, 10, 9).date()


def test_latest_completed_session_before_close_uses_previous_session():
    ts = datetime(2026, 10, 9, 14, 0, tzinfo=ZoneInfo("Asia/Tokyo"))
    assert latest_completed_tse_session(ts) == datetime(2026, 10, 8).date()


def test_year_end_exchange_closures_are_not_business_days():
    assert is_tse_business_day("2026-12-31") is False
    assert is_tse_business_day("2027-01-02") is False
    assert is_tse_business_day("2027-01-03") is False

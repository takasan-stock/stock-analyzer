from __future__ import annotations

import datetime
from typing import Any, Iterable

from stock_action_status import StockActionStatus, evaluate_stock_action_status


STAGE_PRIORITY = {
    "EVENT_CHECK": 0,
    "PRETRADE": 1,
    "ENTRY_MONITOR": 2,
    "MONITOR": 3,
    "RESEARCH": 4,
    "DATA_CHECK": 5,
}


def _ticker(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".0"):
        text = text[:-2]
    if text.upper().endswith(".T"):
        text = text[:-2]
    return text.strip()


def _parse_date(value: Any) -> datetime.date | None:
    text = str(value or "").strip()
    if not text:
        return None
    for candidate in (text[:10], text):
        try:
            return datetime.date.fromisoformat(candidate)
        except ValueError:
            continue
    return None


def _latest_by_ticker(
    rows: Iterable[dict[str, Any]],
    *,
    ticker_key: str,
    date_keys: tuple[str, ...],
) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    latest_stamp: dict[str, str] = {}

    for raw in rows:
        row = dict(raw or {})
        code = _ticker(row.get(ticker_key))
        if not code:
            continue

        stamp = ""
        for key in date_keys:
            value = str(row.get(key, "") or "").strip()
            if value:
                stamp = value
                break

        if code not in latest or stamp >= latest_stamp.get(code, ""):
            latest[code] = row
            latest_stamp[code] = stamp

    return latest


def _status_row(
    *,
    ticker: str,
    name: str,
    portfolio_row: dict[str, Any],
    me_row: dict[str, Any],
    short_row: dict[str, Any],
    live_entry: dict[str, Any],
    today: datetime.date | None,
) -> dict[str, Any]:
    earnings_date = (
        _parse_date(portfolio_row.get("決算予定日"))
        or _parse_date(portfolio_row.get("次回決算予定日"))
        or _parse_date(portfolio_row.get("次回決算"))
    )

    result: StockActionStatus = evaluate_stock_action_status(
        portfolio_row=portfolio_row,
        me_row=me_row,
        short_row=short_row,
        live_entry=live_entry,
        earnings_date=earnings_date,
        today=today,
    )

    lead_reason = result.reasons[0] if result.reasons else ""
    lead_blocker = result.blockers[0] if result.blockers else ""

    return {
        "ticker": ticker,
        "name": name,
        "portfolio_status": str(portfolio_row.get("ステータス", "") or ""),
        "stage": result.stage,
        "stage_label": result.label,
        "stage_icon": result.icon,
        "readiness": result.readiness_score,
        "confidence": result.confidence,
        "summary": result.summary,
        "lead_reason": lead_reason,
        "lead_blocker": lead_blocker,
        "next_page": result.next_page,
        "next_label": result.next_label,
        "next_icon": result.next_icon,
        "mex_score": me_row.get("mex_score"),
        "sw_decision": me_row.get("sw_decision"),
        "entry_status": live_entry.get("status") or short_row.get("alert_tier"),
    }


def build_daily_priority_board(
    *,
    portfolio_rows: Iterable[dict[str, Any]],
    me_rows: Iterable[dict[str, Any]] = (),
    short_rows: Iterable[dict[str, Any]] = (),
    live_entry_rows: Iterable[dict[str, Any]] = (),
    today: datetime.date | None = None,
    include_passed: bool = False,
) -> list[dict[str, Any]]:
    """Build the V3 daily review queue from already-produced snapshots.

    Rows are ordered by workflow urgency first, then Research Readiness.
    This does not fetch market data or create a buy/sell ranking.
    """

    me_map = _latest_by_ticker(
        me_rows,
        ticker_key="ticker",
        date_keys=("trade_date", "data_asof", "available_date"),
    )
    short_map = _latest_by_ticker(
        short_rows,
        ticker_key="ticker",
        date_keys=("alert_date", "last_updated"),
    )
    live_map = _latest_by_ticker(
        live_entry_rows,
        ticker_key="ticker",
        date_keys=("updated_at", "run_at", "timestamp"),
    )

    output: list[dict[str, Any]] = []
    seen: set[str] = set()

    for raw in portfolio_rows:
        portfolio_row = dict(raw or {})
        code = _ticker(portfolio_row.get("ティッカー") or portfolio_row.get("ticker"))
        if not code or code in seen:
            continue
        seen.add(code)

        portfolio_status = str(portfolio_row.get("ステータス", "") or "").strip()
        if not include_passed and portfolio_status == "見送り":
            continue

        name = str(
            portfolio_row.get("銘柄名")
            or me_map.get(code, {}).get("company_name")
            or ""
        ).strip()

        output.append(
            _status_row(
                ticker=code,
                name=name,
                portfolio_row=portfolio_row,
                me_row=me_map.get(code, {}),
                short_row=short_map.get(code, {}),
                live_entry=live_map.get(code, {}),
                today=today,
            )
        )

    output.sort(
        key=lambda row: (
            STAGE_PRIORITY.get(str(row.get("stage", "")), 99),
            -int(row.get("readiness", 0) or 0),
            str(row.get("ticker", "")),
        )
    )

    for index, row in enumerate(output, start=1):
        row["rank"] = index

    return output

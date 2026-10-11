from __future__ import annotations

import json
import os
import re
from typing import Any

import pandas as pd

from deep_fcf_top5 import build_deep_fcf_top5, deep_fcf_strength
from portfolio_quality import normalize_ticker_code


ME_PATH = "data/multiple_expansion/me_screener_latest.csv"
ME_UNIVERSE_PATH = "data/multiple_expansion/me_universe_snapshot.csv"
ENTRY_VALIDATION_PATH = "data/entry_hunter_validation_status.json"


def load_me_screener(path: str = ME_PATH) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path, dtype={"ticker": str})
    except Exception:
        pass
    return pd.DataFrame()


def load_me_universe_snapshot(path: str = ME_UNIVERSE_PATH) -> pd.DataFrame:
    try:
        if os.path.exists(path):
            return pd.read_csv(path, dtype={"ticker": str})
    except Exception:
        pass
    return pd.DataFrame()


def load_entry_validation(path: str = ENTRY_VALIDATION_PATH) -> dict[str, Any]:
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8-sig") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else {}
    except Exception:
        pass
    return {}


def _latest_me_rows(me_screener: pd.DataFrame | None) -> pd.DataFrame:
    if me_screener is None or me_screener.empty or "ticker" not in me_screener.columns:
        return pd.DataFrame()

    me = me_screener.copy()
    me["ticker"] = me["ticker"].map(normalize_ticker_code)

    if "trade_date" in me.columns:
        me["trade_date"] = pd.to_datetime(me["trade_date"], errors="coerce")
        latest = me["trade_date"].max()
        if pd.notna(latest):
            me = me[me["trade_date"] == latest].copy()

    return me.drop_duplicates(subset=["ticker"], keep="first")


def _compact_me_label(row: pd.Series) -> str:
    decision = str(row.get("sw_decision", "") or "").strip()
    state = str(row.get("second_wave_state", "") or "").strip()

    ignored = {"", "UNAVAILABLE", "NONE", "NAN"}
    if decision.upper() not in ignored:
        return decision
    if state.upper() not in ignored:
        return state
    return "—"



BASELINE_ME_DISPLAY = {
    "BASE": "⚪ 通常",
    "RE-ACCEL WATCH": "🟡 再加速監視",
    "RE-READY PROXY": "🟠 再拡大準備",
    "EXPANDING PROXY": "🟢 拡大中",
}


def baseline_me_display_label(value: Any) -> str:
    text = str(value or "").strip().upper()
    if text in {"", "—", "UNAVAILABLE", "NONE", "NAN"}:
        return "—"
    return BASELINE_ME_DISPLAY.get(text, f"⚪ {str(value).strip()}")


def baseline_score_display_label(value: Any) -> str:
    text = str(value or "").strip()
    if not text or text == "—":
        return "—"

    low_coverage = "LOW COVERAGE" in text.upper()
    score = _parse_opportunity(text)
    if score is None:
        return "⚠ 低信頼" if low_coverage else text

    score_text = f"{score:.0f}"
    if low_coverage:
        return f"⚠ {score_text} 低信頼"
    if score >= 75:
        return f"🟢 {score_text}"
    if score >= 65:
        return f"🟡 {score_text}"
    if score >= 60:
        distance = max(0, 65 - int(round(score)))
        return f"🔵 {score_text}｜P3まであと{distance}"
    return f"⚪ {score_text}"

def build_portfolio_signal_overlay(
    portfolio: pd.DataFrame | None,
    me_screener: pd.DataFrame | None,
    entry_validation: dict[str, Any] | None,
    me_universe: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """
    Add read-only derived columns to portfolio rows.

    These columns never belong to portfolio_data.csv; they are a live overlay:
      - ME状態
      - Deep FCF
      - Entry状態
      - Opportunity
      - Baseline ME
      - Baseline Score
    """
    if portfolio is None:
        return pd.DataFrame()

    out = portfolio.copy()
    if out.empty:
        for col in ["ME状態", "Deep FCF", "Entry状態", "Opportunity"]:
            out[col] = []
        return out

    if "ティッカー" not in out.columns:
        out["ティッカー"] = ""

    out["_ticker_norm"] = out["ティッカー"].map(normalize_ticker_code)

    latest_me = _latest_me_rows(me_screener)
    me_map: dict[str, str] = {}
    if not latest_me.empty:
        me_map = {
            normalize_ticker_code(row["ticker"]): _compact_me_label(row)
            for _, row in latest_me.iterrows()
        }

    deep = build_deep_fcf_top5(
        me_screener,
        limit=999,
        min_coverage=0.60,
        min_score=55.0,
        min_quality_grade="C",
    )
    deep_map: dict[str, str] = {}
    if not deep.empty:
        for _, row in deep.iterrows():
            ticker = normalize_ticker_code(row.get("ticker"))
            try:
                score = float(row.get("deep_fcf_score"))
                deep_map[ticker] = f"{score:.0f} {deep_fcf_strength(score)}"
            except (TypeError, ValueError):
                deep_map[ticker] = "—"

    baseline_state_map: dict[str, str] = {}
    baseline_score_map: dict[str, str] = {}
    if me_universe is not None and not me_universe.empty and "ticker" in me_universe.columns:
        universe = me_universe.copy()
        universe["ticker"] = universe["ticker"].map(normalize_ticker_code)
        if "trade_date" in universe.columns:
            universe["trade_date"] = pd.to_datetime(
                universe["trade_date"], errors="coerce"
            )
            latest_universe_date = universe["trade_date"].max()
            if pd.notna(latest_universe_date):
                universe = universe[
                    universe["trade_date"] == latest_universe_date
                ].copy()

        universe = universe.drop_duplicates("ticker", keep="first")
        for _, row in universe.iterrows():
            ticker = normalize_ticker_code(row.get("ticker"))
            if not ticker:
                continue
            state = str(row.get("prefilter_state", "") or "").strip()
            score = pd.to_numeric(
                pd.Series([row.get("coarse_score")]),
                errors="coerce",
            ).iloc[0]
            eligible = bool(row.get("eligible", False))
            if state:
                baseline_state_map[ticker] = state
            if pd.notna(score):
                suffix = "" if eligible else " / LOW COVERAGE"
                baseline_score_map[ticker] = f"{float(score):.0f}{suffix}"

    validation = entry_validation or {}
    entry_map: dict[str, str] = {}
    opp_map: dict[str, str] = {}
    for row in validation.get("rows", []) or []:
        if not isinstance(row, dict):
            continue
        ticker = normalize_ticker_code(row.get("ticker"))
        if not ticker:
            continue
        status = str(row.get("entry_status", "") or "").strip()
        if status:
            entry_map[ticker] = status
        score = row.get("opportunity_score")
        rating = str(row.get("opportunity_rating", "") or "").strip()
        try:
            score_num = float(score)
            opp_map[ticker] = (
                f"{rating} {score_num:.0f}" if rating else f"{score_num:.0f}"
            )
        except (TypeError, ValueError):
            if rating:
                opp_map[ticker] = rating

    watch_tickers = {
        normalize_ticker_code(x)
        for x in (validation.get("me_watch_tickers", []) or [])
    }

    out["ME状態"] = out["_ticker_norm"].map(me_map).fillna("—")
    out["Deep FCF"] = out["_ticker_norm"].map(deep_map).fillna("—")
    out["Entry状態"] = out["_ticker_norm"].map(entry_map).fillna("—")
    out["Opportunity"] = out["_ticker_norm"].map(opp_map).fillna("—")
    out["Baseline ME"] = out["_ticker_norm"].map(baseline_state_map).fillna("—")
    out["Baseline Score"] = out["_ticker_norm"].map(baseline_score_map).fillna("—")
    out["Baseline ME表示"] = out["Baseline ME"].map(baseline_me_display_label)
    out["Baseline Score表示"] = out["Baseline Score"].map(baseline_score_display_label)

    me_watch_mask = (
        out["_ticker_norm"].isin(watch_tickers)
        & out["Entry状態"].eq("—")
    )
    out.loc[me_watch_mask, "Entry状態"] = "🟡 ME WATCH"

    out = out.drop(columns=["_ticker_norm"])
    return apply_portfolio_priority(out)



def _parse_opportunity(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text or text == "—":
        return None
    m = __import__("re").search(r"(-?\d+(?:\.\d+)?)", text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except ValueError:
        return None


def portfolio_priority_label(row: pd.Series | dict[str, Any]) -> tuple[str, int]:
    """
    Display-only priority for deciding what to inspect first.

    This does not open entry gates or modify trading scores.

    P1: Entry READY, or Opportunity >= 70
    P2: ME READY / PRIORITY WATCH / ACTIVE, or WAIT + Opportunity >= 55
    P3: ME WATCH, Opportunity 40-54, or qualifying Deep FCF
    P4: evaluated but low-priority
    未評価: no ME / Deep FCF / Entry / Opportunity data
    """
    entry = str(row.get("Entry状態", "") or "").upper()
    me = str(row.get("ME状態", "") or "").upper()
    deep = str(row.get("Deep FCF", "") or "").upper()
    opp_text = str(row.get("Opportunity", "") or "").strip()
    baseline_state = str(row.get("Baseline ME", "") or "").upper()
    baseline_score_text = str(row.get("Baseline Score", "") or "").strip()
    opp = _parse_opportunity(opp_text)
    baseline_score = _parse_opportunity(baseline_score_text)

    empty_markers = {"", "—", "UNAVAILABLE", "NONE", "NAN"}
    has_any_signal_data = (
        entry not in empty_markers
        or me not in empty_markers
        or deep not in empty_markers
        or opp_text.upper() not in empty_markers
        or baseline_state not in empty_markers
        or baseline_score_text.upper() not in empty_markers
    )
    if not has_any_signal_data:
        return "⚪ 未評価", 5

    if "READY" in entry:
        return "🔴 P1 今すぐ確認", 1

    if opp is not None and opp >= 70:
        return "🔴 P1 今すぐ確認", 1

    if me in {"READY", "PRIORITY WATCH", "ACTIVE"}:
        return "🟠 P2 優先監視", 2

    if "WAIT" in entry and opp is not None and opp >= 55:
        return "🟠 P2 優先監視", 2

    if "ME WATCH" in entry:
        return "🟡 P3 通常監視", 3

    if opp is not None and opp >= 40:
        return "🟡 P3 通常監視", 3

    if deep != "—" and any(tag in deep for tag in ("STRONG", "GOOD", "SOLID", "PASS")):
        return "🟡 P3 通常監視", 3

    # Baseline ME is lightweight all-market screening only.
    # It can raise visibility to P3 but must never create P1/P2.
    if baseline_state in {"RE-READY PROXY", "EXPANDING PROXY", "RE-ACCEL WATCH"}:
        return "🟡 P3 通常監視", 3

    if baseline_score is not None and baseline_score >= 65:
        return "🟡 P3 通常監視", 3

    return "⚪ P4 待機", 4


def apply_portfolio_priority(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None:
        return pd.DataFrame()
    out = frame.copy()
    if out.empty:
        out["今日の優先度"] = []
        out["_priority_rank"] = []
        return out

    labels = out.apply(portfolio_priority_label, axis=1)
    out["今日の優先度"] = [x[0] for x in labels]
    out["_priority_rank"] = [x[1] for x in labels]
    return out

def split_news_by_age(
    news_items: list[dict[str, Any]] | None,
    *,
    recent_days: int = 30,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    recent: list[dict[str, Any]] = []
    older: list[dict[str, Any]] = []

    threshold_hours = max(1, int(recent_days)) * 24
    for item in news_items or []:
        hours = item.get("hours_ago")
        try:
            hours_num = float(hours)
        except (TypeError, ValueError):
            hours_num = None

        if hours_num is not None and hours_num <= threshold_hours:
            recent.append(item)
        else:
            older.append(item)

    return recent, older

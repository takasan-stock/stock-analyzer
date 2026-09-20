from __future__ import annotations

import math
from typing import Any

import pandas as pd


JOURNAL_COLUMNS = [
    "journal_id", "plan_id", "updated_at", "status",
    "ticker", "name", "source", "plan_confirmed_at",
    "verdict", "total_score", "setup_score", "entry_score", "risk_score",
    "stage", "rs_proxy", "volume_ratio", "ema20_gap_pct",
    "hunter_status", "hunter_score",
    "planned_entry", "planned_stop", "planned_target", "planned_rr",
    "planned_shares", "planned_max_loss",
    "actual_entry_date", "actual_entry", "actual_stop", "actual_target",
    "actual_shares", "entry_slippage_pct", "risk_per_share", "risk_amount",
    "actual_exit_date", "actual_exit", "fees",
    "pnl", "return_pct", "r_multiple", "holding_days",
    "exit_reason", "skip_reason", "followed_plan", "rule_break", "memo",
]


def _num(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        text = str(value).replace(",", "").replace("円", "").strip()
        if text == "":
            return None
        return float(text)
    except (TypeError, ValueError):
        return None


def _date(value) -> pd.Timestamp | None:
    try:
        ts = pd.to_datetime(value, errors="coerce")
        if pd.isna(ts):
            return None
        return pd.Timestamp(ts).normalize()
    except Exception:
        return None


def score_bucket(value) -> str:
    score = _num(value)
    if score is None:
        return "未取得"
    if score >= 85:
        return "85-100"
    if score >= 75:
        return "75-84"
    if score >= 65:
        return "65-74"
    if score >= 50:
        return "50-64"
    return "<50"


def calc_trade_outcome(
    *,
    planned_entry,
    actual_entry,
    actual_stop,
    shares,
    actual_exit=None,
    fees=0.0,
    entry_date=None,
    exit_date=None,
) -> dict[str, Any]:
    """Calculate realized trade metrics using the actual execution risk.

    R multiple is based on (actual_entry - actual_stop) * shares.
    It therefore measures the risk actually accepted at entry rather than the
    original plan's theoretical risk.
    """
    p_entry = _num(planned_entry)
    a_entry = _num(actual_entry)
    a_stop = _num(actual_stop)
    qty = _num(shares)
    a_exit = _num(actual_exit)
    fees_v = max(0.0, _num(fees) or 0.0)

    entry_slippage_pct = None
    if p_entry is not None and p_entry > 0 and a_entry is not None:
        entry_slippage_pct = (a_entry / p_entry - 1.0) * 100.0

    risk_per_share = None
    risk_amount = None
    if (
        a_entry is not None and a_entry > 0
        and a_stop is not None and a_stop > 0
        and a_stop < a_entry
        and qty is not None and qty > 0
    ):
        risk_per_share = a_entry - a_stop
        risk_amount = risk_per_share * qty

    pnl = None
    return_pct = None
    r_multiple = None
    if (
        a_entry is not None and a_entry > 0
        and a_exit is not None and a_exit > 0
        and qty is not None and qty > 0
    ):
        pnl = (a_exit - a_entry) * qty - fees_v
        invested = a_entry * qty
        if invested > 0:
            return_pct = pnl / invested * 100.0
        if risk_amount is not None and risk_amount > 0:
            r_multiple = pnl / risk_amount

    holding_days = None
    d1 = _date(entry_date)
    d2 = _date(exit_date)
    if d1 is not None and d2 is not None and d2 >= d1:
        holding_days = int((d2 - d1).days)

    return {
        "entry_slippage_pct": entry_slippage_pct,
        "risk_per_share": risk_per_share,
        "risk_amount": risk_amount,
        "pnl": pnl,
        "return_pct": return_pct,
        "r_multiple": r_multiple,
        "holding_days": holding_days,
    }


def normalize_journal(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame(columns=JOURNAL_COLUMNS)
    out = df.copy()
    for col in JOURNAL_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    return out[JOURNAL_COLUMNS].fillna("")


def upsert_journal_record(
    df: pd.DataFrame | None,
    record: dict[str, Any],
) -> pd.DataFrame:
    out = normalize_journal(df)
    plan_id = str(record.get("plan_id", "") or "").strip()
    if not plan_id:
        raise ValueError("plan_id is required")

    row = {col: record.get(col, "") for col in JOURNAL_COLUMNS}
    if plan_id in set(out["plan_id"].astype(str)):
        idx = out.index[out["plan_id"].astype(str) == plan_id][-1]
        for col, value in row.items():
            out.at[idx, col] = value
    else:
        out = pd.concat([out, pd.DataFrame([row])], ignore_index=True)

    return normalize_journal(out)


def closed_trades(df: pd.DataFrame | None) -> pd.DataFrame:
    out = normalize_journal(df)
    if out.empty:
        return out

    status = out["status"].astype(str)
    mask = status.eq("CLOSED")
    result = out[mask].copy()
    if result.empty:
        return result

    for col in [
        "total_score", "setup_score", "entry_score", "risk_score",
        "planned_rr", "planned_shares", "planned_max_loss",
        "actual_entry", "actual_stop", "actual_target", "actual_shares",
        "entry_slippage_pct", "risk_per_share", "risk_amount",
        "actual_exit", "fees", "pnl", "return_pct", "r_multiple",
        "holding_days",
    ]:
        result[col] = pd.to_numeric(result[col], errors="coerce")
    return result


def build_summary(df: pd.DataFrame | None) -> dict[str, Any]:
    trades = closed_trades(df)
    if trades.empty:
        return {
            "trades": 0,
            "wins": 0,
            "win_rate": None,
            "total_pnl": 0.0,
            "avg_pnl": None,
            "avg_return_pct": None,
            "avg_r": None,
            "median_r": None,
            "profit_factor": None,
            "avg_holding_days": None,
            "plan_follow_rate": None,
        }

    pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
    r = pd.to_numeric(trades["r_multiple"], errors="coerce").dropna()
    returns = pd.to_numeric(trades["return_pct"], errors="coerce").dropna()
    hold = pd.to_numeric(trades["holding_days"], errors="coerce").dropna()

    wins = int((pnl > 0).sum())
    count = int(len(trades))
    gross_profit = float(pnl[pnl > 0].sum()) if not pnl.empty else 0.0
    gross_loss_abs = abs(float(pnl[pnl < 0].sum())) if not pnl.empty else 0.0
    if gross_loss_abs > 0:
        profit_factor = gross_profit / gross_loss_abs
    elif gross_profit > 0:
        profit_factor = math.inf
    else:
        profit_factor = None

    followed = trades["followed_plan"].astype(str).str.lower()
    follow_mask = followed.isin({"true", "1", "yes", "y", "はい", "プラン通り"})
    known_follow = followed.ne("")
    plan_follow_rate = (
        float(follow_mask[known_follow].mean() * 100.0)
        if known_follow.any()
        else None
    )

    return {
        "trades": count,
        "wins": wins,
        "win_rate": wins / count * 100.0 if count else None,
        "total_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
        "avg_pnl": float(pnl.mean()) if not pnl.empty else None,
        "avg_return_pct": float(returns.mean()) if not returns.empty else None,
        "avg_r": float(r.mean()) if not r.empty else None,
        "median_r": float(r.median()) if not r.empty else None,
        "profit_factor": profit_factor,
        "avg_holding_days": float(hold.mean()) if not hold.empty else None,
        "plan_follow_rate": plan_follow_rate,
    }


def build_group_summary(
    df: pd.DataFrame | None,
    group_col: str,
) -> pd.DataFrame:
    trades = closed_trades(df)
    columns = [
        group_col, "trades", "wins", "win_rate",
        "avg_r", "median_r", "avg_return_pct", "total_pnl",
    ]
    if trades.empty or group_col not in trades.columns:
        return pd.DataFrame(columns=columns)

    if group_col == "score_bucket":
        trades = trades.copy()
        trades["score_bucket"] = trades["total_score"].map(score_bucket)

    rows = []
    for key, group in trades.groupby(group_col, dropna=False):
        pnl = pd.to_numeric(group["pnl"], errors="coerce").dropna()
        r = pd.to_numeric(group["r_multiple"], errors="coerce").dropna()
        ret = pd.to_numeric(group["return_pct"], errors="coerce").dropna()
        count = len(group)
        wins = int((pnl > 0).sum())
        rows.append({
            group_col: str(key) if str(key).strip() else "未取得",
            "trades": count,
            "wins": wins,
            "win_rate": wins / count * 100.0 if count else None,
            "avg_r": float(r.mean()) if not r.empty else None,
            "median_r": float(r.median()) if not r.empty else None,
            "avg_return_pct": float(ret.mean()) if not ret.empty else None,
            "total_pnl": float(pnl.sum()) if not pnl.empty else 0.0,
        })

    out = pd.DataFrame(rows)
    if group_col == "score_bucket" and not out.empty:
        order = {"85-100": 0, "75-84": 1, "65-74": 2, "50-64": 3, "<50": 4, "未取得": 5}
        out["_order"] = out[group_col].map(order).fillna(99)
        out = out.sort_values("_order").drop(columns="_order")
    elif not out.empty:
        out = out.sort_values(["trades", "avg_r"], ascending=[False, False], na_position="last")

    return out.reset_index(drop=True)

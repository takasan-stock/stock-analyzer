from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


DEFAULT_HORIZONS = (20, 60, 120, 250)


def _num(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def mex_bucket(score) -> str:
    value = _num(score)
    if value is None:
        return "UNSCORED"
    if value >= 90:
        return "90-100"
    if value >= 80:
        return "80-89"
    if value >= 70:
        return "70-79"
    if value >= 60:
        return "60-69"
    return "<60"


def _max_drawdown_from_path(path: pd.Series) -> float | None:
    s = pd.to_numeric(path, errors="coerce").dropna()
    if s.empty:
        return None
    wealth = s / float(s.iloc[0])
    peak = wealth.cummax()
    dd = wealth / peak - 1.0
    return float(dd.min())


def _excursions_from_entry(path: pd.Series) -> tuple[float | None, float | None]:
    s = pd.to_numeric(path, errors="coerce").dropna()
    if s.empty:
        return None, None
    entry = float(s.iloc[0])
    if entry <= 0:
        return None, None
    rel = s / entry - 1.0
    return float(rel.max()), float(rel.min())


def build_mex_event_study(
    history: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    event_col: str = "ignition_event",
    min_score: float | None = None,
) -> pd.DataFrame:
    """Create one row per confirmed MEX event with forward-return outcomes.

    Uses trading-row offsets within each ticker; no calendar-day interpolation.
    If topix_close is available, excess returns use the same row dates.
    """
    if history is None or history.empty:
        return pd.DataFrame()

    required = {"ticker", "trade_date", "adj_close", event_col}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(
            "history missing required columns: "
            + ", ".join(sorted(missing))
        )

    horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    df = history.copy()
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce")
    df["adj_close"] = pd.to_numeric(df["adj_close"], errors="coerce")
    if "topix_close" in df.columns:
        df["topix_close"] = pd.to_numeric(df["topix_close"], errors="coerce")
    else:
        df["topix_close"] = pd.NA
    df = df.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    rows = []
    for ticker, group in df.groupby("ticker", sort=False):
        g = group.reset_index(drop=True)
        events = g[g[event_col].fillna(False).astype(bool)]

        for idx in events.index:
            event = g.loc[idx]
            score = _num(event.get("mex_score"))
            if min_score is not None and (
                score is None or score < min_score
            ):
                continue

            entry = _num(event.get("adj_close"))
            if entry is None or entry <= 0:
                continue

            topix_entry = _num(event.get("topix_close"))
            row = {
                "ticker": ticker,
                "event_date": event["trade_date"],
                "event_type": event_col,
                "mex_score": score,
                "mex_bucket": mex_bucket(score),
                "fcf_engine_score": _num(event.get("fcf_engine_score")),
                "mlp_c_core": _num(event.get("mlp_c_core")),
                "mlp_z": _num(event.get("mlp_z")),
                "mlp_velocity_20": _num(event.get("mlp_velocity_20")),
                "mlp_acceleration": _num(event.get("mlp_acceleration")),
                "normalization_gap": _num(event.get("normalization_gap")),
                "state": event.get("state_confirmed", event.get("state_raw")),
                "fcf_basis": event.get("fcf_basis"),
                "data_quality": event.get("data_quality"),
            }

            for horizon in horizons:
                end_idx = idx + horizon
                ret = excess = None
                mfe = mae = max_dd = None

                if end_idx < len(g):
                    end_price = _num(g.loc[end_idx, "adj_close"])
                    if end_price is not None and end_price > 0:
                        ret = end_price / entry - 1.0

                    topix_end = _num(g.loc[end_idx, "topix_close"])
                    if (
                        ret is not None
                        and topix_entry is not None
                        and topix_entry > 0
                        and topix_end is not None
                        and topix_end > 0
                    ):
                        topix_ret = topix_end / topix_entry - 1.0
                        excess = ret - topix_ret

                    path = g.loc[idx:end_idx, "adj_close"]
                    mfe, mae = _excursions_from_entry(path)
                    max_dd = _max_drawdown_from_path(path)

                row[f"ret_{horizon}d"] = ret
                row[f"excess_{horizon}d"] = excess
                row[f"mfe_{horizon}d"] = mfe
                row[f"mae_{horizon}d"] = mae
                row[f"max_dd_{horizon}d"] = max_dd

            rows.append(row)

    return pd.DataFrame(rows)


def summarize_mex_event_study(
    events: pd.DataFrame,
    *,
    horizons: Iterable[int] = DEFAULT_HORIZONS,
    group_col: str = "mex_bucket",
) -> pd.DataFrame:
    if events is None or events.empty:
        return pd.DataFrame()

    horizons = tuple(sorted({int(h) for h in horizons if int(h) > 0}))
    if group_col not in events.columns:
        raise ValueError(f"{group_col} not found in event study")

    bucket_order = {
        "90-100": 0,
        "80-89": 1,
        "70-79": 2,
        "60-69": 3,
        "<60": 4,
        "UNSCORED": 5,
    }

    rows = []
    for group_value, g in events.groupby(group_col, dropna=False):
        for horizon in horizons:
            ret_col = f"ret_{horizon}d"
            excess_col = f"excess_{horizon}d"
            mfe_col = f"mfe_{horizon}d"
            mae_col = f"mae_{horizon}d"
            dd_col = f"max_dd_{horizon}d"

            ret = pd.to_numeric(g.get(ret_col), errors="coerce").dropna()
            excess = pd.to_numeric(g.get(excess_col), errors="coerce").dropna()
            mfe = pd.to_numeric(g.get(mfe_col), errors="coerce").dropna()
            mae = pd.to_numeric(g.get(mae_col), errors="coerce").dropna()
            dd = pd.to_numeric(g.get(dd_col), errors="coerce").dropna()

            rows.append({
                group_col: group_value,
                "horizon": horizon,
                "signals": int(len(g)),
                "completed": int(len(ret)),
                "mean_return": float(ret.mean()) if not ret.empty else None,
                "median_return": float(ret.median()) if not ret.empty else None,
                "positive_hit_rate": (
                    float((ret > 0).mean()) if not ret.empty else None
                ),
                "mean_excess": (
                    float(excess.mean()) if not excess.empty else None
                ),
                "median_excess": (
                    float(excess.median()) if not excess.empty else None
                ),
                "topix_hit_rate": (
                    float((excess > 0).mean()) if not excess.empty else None
                ),
                "median_mfe": float(mfe.median()) if not mfe.empty else None,
                "median_mae": float(mae.median()) if not mae.empty else None,
                "median_max_dd": float(dd.median()) if not dd.empty else None,
            })

    out = pd.DataFrame(rows)
    if group_col == "mex_bucket":
        out["_order"] = out[group_col].map(bucket_order).fillna(99)
        out = out.sort_values(["horizon", "_order"]).drop(columns="_order")
    else:
        out = out.sort_values(["horizon", group_col])
    return out.reset_index(drop=True)


def monotonicity_report(
    summary: pd.DataFrame,
    *,
    horizon: int = 120,
    metric: str = "median_excess",
) -> dict:
    """Check whether higher score buckets broadly outperform lower buckets."""
    if summary is None or summary.empty:
        return {
            "status": "NO_DATA",
            "pairs": 0,
            "ordered_pairs": 0,
            "ratio": None,
        }

    bucket_order = ["90-100", "80-89", "70-79", "60-69", "<60"]
    s = summary[summary["horizon"] == int(horizon)].copy()
    if s.empty or metric not in s.columns:
        return {
            "status": "NO_DATA",
            "pairs": 0,
            "ordered_pairs": 0,
            "ratio": None,
        }

    values = {}
    for bucket in bucket_order:
        row = s[s["mex_bucket"] == bucket]
        if row.empty:
            continue
        value = _num(row.iloc[0].get(metric))
        if value is not None:
            values[bucket] = value

    pairs = ordered = 0
    available = [b for b in bucket_order if b in values]
    for left, right in zip(available[:-1], available[1:]):
        pairs += 1
        if values[left] >= values[right]:
            ordered += 1

    ratio = ordered / pairs if pairs else None
    if ratio is None:
        status = "INSUFFICIENT"
    elif ratio >= 0.75:
        status = "PASS"
    elif ratio >= 0.50:
        status = "MIXED"
    else:
        status = "FAIL"

    return {
        "status": status,
        "pairs": pairs,
        "ordered_pairs": ordered,
        "ratio": ratio,
        "horizon": int(horizon),
        "metric": metric,
        "values": values,
    }

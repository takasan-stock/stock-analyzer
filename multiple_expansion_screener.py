from __future__ import annotations

import math
from typing import Any

import pandas as pd


DEFAULT_CONFIG = {
    "re_min_mlp": 1.07,
    "reset_mlp": 1.03,
    "re_accel_min_pct": 0.10,
    "re_curvature_min_pct": 0.10,
    "re_min_pullback": 0.025,
    "re_min_decel_bars": 5,
    "ready_velocity_pct": -3.0,
    "advance_gate_days": 120,
}

SECOND_WAVE_STATES = {
    "EXP. DECELERATING",
    "RE-WATCH EARLY",
    "RE-WATCH READY",
    "RE-EXP",
}

SCREEN_PRIORITY = {
    "PRIORITY WATCH": 5,
    "READY": 4,
    "ACTIVE": 4,
    "WATCH": 3,
    "FIRST WAVE": 2,
    "PASS": 0,
}


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _pct_score(value: float | None, anchors: list[tuple[float, float]]) -> float:
    if value is None:
        return 50.0
    pts = sorted(anchors)
    if value <= pts[0][0]:
        return pts[0][1]
    if value >= pts[-1][0]:
        return pts[-1][1]
    for (x0, y0), (x1, y1) in zip(pts[:-1], pts[1:]):
        if x0 <= value <= x1:
            if x1 == x0:
                return y1
            ratio = (value - x0) / (x1 - x0)
            return y0 + ratio * (y1 - y0)
    return 50.0


def add_reacceleration_engine(
    history: pd.DataFrame,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Add a conservative second-wave / re-expansion state machine.

    The thresholds mirror the TradingView v0.8.x family:
    - expansion reference MLP >= 1.07
    - minimum pullback 2.5%
    - at least 5 deceleration bars
    - re-acceleration / curvature minimum +0.10 percentage points
    - R-READY velocity threshold -3.0%
    - 120 trading-day advance gate after a RE-EXP event

    This intentionally suppresses repeated R-READY signals while the
    advance gate is locked. That makes the daily screener favor genuinely
    re-armed second-wave setups instead of repeatedly ranking one old trend.
    """
    if history is None or history.empty:
        return pd.DataFrame() if history is None else history.copy()

    cfg = {**DEFAULT_CONFIG, **(config or {})}
    out = history.copy()
    required = {"ticker", "trade_date", "mlp_c_core"}
    missing = required - set(out.columns)
    if missing:
        raise ValueError(
            "history missing required columns: " + ", ".join(sorted(missing))
        )

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    groups: list[pd.DataFrame] = []
    for _, g in out.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)
        core = pd.to_numeric(g["mlp_c_core"], errors="coerce")

        if "mlp_velocity_pct_20" in g.columns:
            velocity_pct = (
                pd.to_numeric(g["mlp_velocity_pct_20"], errors="coerce") * 100.0
            )
        else:
            velocity_pct = (core / core.shift(20) - 1.0) * 100.0

        if "mlp_acceleration" in g.columns:
            acceleration_pct = (
                pd.to_numeric(g["mlp_acceleration"], errors="coerce") * 100.0
            )
        else:
            v_log = (core / core.shift(20)).map(
                lambda x: math.log(x)
                if pd.notna(x) and x > 0
                else float("nan")
            )
            acceleration_pct = (v_log - v_log.shift(10)) * 100.0

        slope10 = (core / core.shift(10) - 1.0) * 100.0
        curvature_pct = slope10 - slope10.shift(10)

        g["me_velocity_pct"] = velocity_pct
        g["me_acceleration_pct"] = acceleration_pct
        g["me_slope10_pct"] = slope10
        g["me_curvature_pct"] = curvature_pct

        had_expansion = False
        peak_mlp: float | None = None
        decel_bars = 0
        engine = "IDLE"
        last_reexp_idx: int | None = None
        previous_state = ""

        states: list[str] = []
        engines: list[str] = []
        routes: list[str] = []
        pullbacks: list[float | None] = []
        gate_progress: list[int | None] = []
        state_changed: list[bool] = []
        reexp_events: list[bool] = []

        for i, row in g.iterrows():
            mlp = _num(row.get("mlp_c_core"))
            vel = _num(velocity_pct.iloc[i])
            acc = _num(acceleration_pct.iloc[i])
            curv = _num(curvature_pct.iloc[i])

            if (
                mlp is not None
                and vel is not None
                and mlp >= cfg["re_min_mlp"]
                and vel > 0
            ):
                had_expansion = True
                peak_mlp = mlp if peak_mlp is None else max(peak_mlp, mlp)

            if had_expansion and mlp is not None and peak_mlp is not None:
                peak_mlp = max(peak_mlp, mlp)
                pullback = max(0.0, 1.0 - (mlp / peak_mlp))
            else:
                pullback = None

            if had_expansion and vel is not None and vel < 0:
                decel_bars += 1
            elif vel is not None and vel >= 0:
                decel_bars = 0

            bars_since_reexp = (
                None if last_reexp_idx is None else i - last_reexp_idx
            )
            gate_locked = (
                bars_since_reexp is not None
                and bars_since_reexp <= int(cfg["advance_gate_days"])
            )

            if gate_locked:
                route = "LOCKED"
            elif last_reexp_idx is not None:
                route = "RE-ARMED"
            else:
                route = "OPEN"

            arm_ok = (
                had_expansion
                and pullback is not None
                and pullback >= float(cfg["re_min_pullback"])
                and decel_bars >= int(cfg["re_min_decel_bars"])
            )
            if gate_locked:
                engine = "LOCKED"
            elif arm_ok:
                engine = "ARMED"
            elif engine == "LOCKED" and not gate_locked:
                engine = "IDLE"

            first_wave_state = str(row.get("state_confirmed") or row.get("state_raw") or "")
            state = first_wave_state or "UNAVAILABLE"

            if mlp is not None and mlp >= 1.28:
                state = "OVERHEATED"
            elif had_expansion and vel is not None and vel <= 0:
                state = "EXP. DECELERATING"

            re_shape_ok = (
                engine == "ARMED"
                and acc is not None
                and curv is not None
                and acc >= float(cfg["re_accel_min_pct"])
                and curv >= float(cfg["re_curvature_min_pct"])
            )

            if re_shape_ok and vel is not None:
                if (
                    vel > 0
                    and mlp is not None
                    and mlp >= float(cfg["reset_mlp"])
                ):
                    state = "RE-EXP"
                elif vel >= float(cfg["ready_velocity_pct"]):
                    state = "RE-WATCH READY"
                else:
                    state = "RE-WATCH EARLY"

            # Do not issue a second-wave state while the advance gate is locked.
            if gate_locked and state in {"RE-WATCH EARLY", "RE-WATCH READY", "RE-EXP"}:
                state = "EXP. DECELERATING"

            changed = state != previous_state
            reexp_event = state == "RE-EXP" and previous_state != "RE-EXP"
            if reexp_event:
                last_reexp_idx = i
                route = "LOCKED"
                engine = "LOCKED"

            states.append(state)
            engines.append(engine)
            routes.append(route)
            pullbacks.append(pullback)
            gate_progress.append(bars_since_reexp)
            state_changed.append(changed)
            reexp_events.append(reexp_event)
            previous_state = state

        g["second_wave_state"] = states
        g["re_engine"] = engines
        g["re_route"] = routes
        g["mlp_pullback"] = pullbacks
        g["advance_gate_progress"] = gate_progress
        g["second_wave_state_changed"] = state_changed
        g["reexp_event_v09"] = reexp_events
        groups.append(g)

    return pd.concat(groups, ignore_index=True)


def add_historical_edge(history: pd.DataFrame) -> pd.DataFrame:
    """Attach resolved forward-performance stats for second-wave events."""
    if history is None or history.empty:
        return pd.DataFrame() if history is None else history.copy()
    if "second_wave_state" not in history.columns:
        raise ValueError("second_wave_state is required")

    out = history.copy()
    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    summary_by_ticker: dict[str, dict[str, float | int | str | None]] = {}

    for ticker, g in out.groupby("ticker", sort=False):
        g = g.copy().reset_index(drop=True)
        price = pd.to_numeric(
            g.get(
                "adj_close",
                pd.Series(pd.NA, index=g.index, dtype="Float64"),
            ),
            errors="coerce",
        )
        bench = pd.to_numeric(
            g.get(
                "topix_close",
                pd.Series(pd.NA, index=g.index, dtype="Float64"),
            ),
            errors="coerce",
        )
        state = g["second_wave_state"].astype(str)
        changed = state.ne(state.shift(1))

        event_mask = changed & state.isin(
            {"RE-WATCH EARLY", "RE-WATCH READY", "RE-EXP"}
        )
        event_indices = list(g.index[event_mask])

        resolved_60: list[dict[str, float]] = []
        for i in event_indices:
            if i + 60 >= len(g):
                continue
            p0 = _num(price.iloc[i])
            p60 = _num(price.iloc[i + 60])
            if p0 is None or p60 is None or p0 <= 0:
                continue
            ret60 = p60 / p0 - 1.0

            excess60 = ret60
            b0 = _num(bench.iloc[i]) if len(bench) else None
            b60 = _num(bench.iloc[i + 60]) if len(bench) else None
            if b0 is not None and b60 is not None and b0 > 0:
                excess60 = ret60 - (b60 / b0 - 1.0)

            window = price.iloc[i + 1 : i + 61].dropna()
            mfe = (
                float(window.max()) / p0 - 1.0
                if not window.empty
                else float("nan")
            )
            mae = (
                float(window.min()) / p0 - 1.0
                if not window.empty
                else float("nan")
            )
            resolved_60.append(
                {
                    "ret60": ret60,
                    "excess60": excess60,
                    "mfe60": mfe,
                    "mae60": mae,
                }
            )

        n = len(resolved_60)
        if n:
            ret = pd.Series([x["ret60"] for x in resolved_60], dtype=float)
            excess = pd.Series([x["excess60"] for x in resolved_60], dtype=float)
            mfe = pd.Series([x["mfe60"] for x in resolved_60], dtype=float)
            mae = pd.Series([x["mae60"] for x in resolved_60], dtype=float)
            win = float((ret > 0).mean())
            avg = float(ret.mean())
            avg_excess = float(excess.mean())
            avg_mfe = float(mfe.mean())
            avg_mae = float(mae.mean())
            payoff = (
                abs(avg_mfe / avg_mae)
                if avg_mae < 0
                else None
            )
            edge_score = _clip(
                50.0
                + avg_excess * 180.0
                + (win - 0.50) * 40.0
                + max(-10.0, min(10.0, avg * 50.0))
            )
        else:
            win = avg = avg_excess = avg_mfe = avg_mae = None
            payoff = None
            edge_score = 45.0

        if n >= 8:
            confidence = "STRONG"
        elif n >= 5:
            confidence = "MEDIUM"
        elif n >= 3:
            confidence = "LOW"
        else:
            confidence = "VERY LOW"

        summary_by_ticker[str(ticker)] = {
            "hist_edge_n": n,
            "hist_60d_win": win,
            "hist_60d_avg": avg,
            "hist_60d_excess": avg_excess,
            "hist_60d_mfe": avg_mfe,
            "hist_60d_mae": avg_mae,
            "hist_mfe_mae": payoff,
            "hist_edge_score": edge_score,
            "hist_confidence": confidence,
        }

    for col in [
        "hist_edge_n",
        "hist_60d_win",
        "hist_60d_avg",
        "hist_60d_excess",
        "hist_60d_mfe",
        "hist_60d_mae",
        "hist_mfe_mae",
        "hist_edge_score",
        "hist_confidence",
    ]:
        out[col] = out["ticker"].astype(str).map(
            {k: v[col] for k, v in summary_by_ticker.items()}
        )

    return out


def build_daily_screener(history: pd.DataFrame) -> pd.DataFrame:
    """Create the daily 'multiple expansion may start soon' ranking."""
    if history is None or history.empty:
        return pd.DataFrame()

    enriched = add_reacceleration_engine(history)
    enriched = add_historical_edge(enriched)

    latest = (
        enriched.sort_values(["ticker", "trade_date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )

    state_base = {
        "RE-EXP": 92.0,
        "RE-WATCH READY": 82.0,
        "RE-WATCH EARLY": 68.0,
        "EXP. DECELERATING": 52.0,
        "IGNITION": 85.0,
        "PREP": 65.0,
        "SPRING": 58.0,
    }

    scores: list[float] = []
    decisions: list[str] = []
    candidate_types: list[str] = []
    reasons: list[str] = []
    fundamental_labels: list[str] = []

    for _, row in latest.iterrows():
        state = str(row.get("second_wave_state") or "UNAVAILABLE")
        base = state_base.get(state, 25.0)
        fcf = _num(row.get("fcf_engine_score"))
        hist = _num(row.get("hist_edge_score"))
        accel = _num(row.get("me_acceleration_pct"))
        confirm = _num(row.get("market_confirm_count"))

        fcf_score = 50.0 if fcf is None else _clip(fcf)
        hist_score = 45.0 if hist is None else _clip(hist)
        accel_score = _pct_score(
            accel,
            [
                (-1.0, 0.0),
                (0.0, 45.0),
                (0.10, 60.0),
                (0.50, 80.0),
                (1.00, 100.0),
            ],
        )
        market_score = _clip((confirm or 0.0) * 25.0)

        score = _clip(
            base * 0.40
            + fcf_score * 0.20
            + hist_score * 0.20
            + accel_score * 0.15
            + market_score * 0.05
        )

        if fcf is None:
            fundamental = "UNKNOWN"
        elif fcf >= 70:
            fundamental = "STRONG"
        elif fcf >= 55:
            fundamental = "MIXED"
        else:
            fundamental = "WEAK"

        is_second_wave = state in SECOND_WAVE_STATES
        is_first_wave = state in {"IGNITION", "PREP", "SPRING"}

        if state == "RE-EXP" and score >= 65:
            decision = "ACTIVE"
        elif is_second_wave and score >= 70:
            decision = "PRIORITY WATCH"
        elif state == "RE-WATCH READY" and score >= 55:
            decision = "READY"
        elif state in {"RE-WATCH EARLY", "EXP. DECELERATING"} and score >= 45:
            decision = "WATCH"
        elif is_first_wave and score >= 55:
            decision = "FIRST WAVE"
        else:
            decision = "PASS"

        candidate_type = (
            "SECOND WAVE"
            if is_second_wave
            else ("FIRST WAVE" if is_first_wave else "OTHER")
        )

        reason_parts = [state]
        if fundamental == "STRONG":
            reason_parts.append("FCF STRONG")
        if (_num(row.get("hist_edge_n")) or 0) < 3:
            reason_parts.append("LOW SAMPLE")
        elif (_num(row.get("hist_edge_score")) or 0) >= 60:
            reason_parts.append("HIST EDGE")
        if accel is not None and accel >= 0.10:
            reason_parts.append("ACCEL+")
        if str(row.get("re_route") or "") == "LOCKED":
            reason_parts.append("GATE LOCKED")

        scores.append(score)
        decisions.append(decision)
        candidate_types.append(candidate_type)
        reasons.append(" / ".join(reason_parts))
        fundamental_labels.append(fundamental)

    latest["sw_score"] = scores
    latest["sw_decision"] = decisions
    latest["candidate_type"] = candidate_types
    latest["screen_reason"] = reasons
    latest["fundamental_label"] = fundamental_labels
    latest["screen_priority"] = latest["sw_decision"].map(SCREEN_PRIORITY).fillna(0)

    latest = latest.sort_values(
        ["screen_priority", "sw_score", "hist_edge_score", "mex_score", "ticker"],
        ascending=[False, False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    latest["screen_rank"] = range(1, len(latest) + 1)

    return latest


def candidate_only(screener: pd.DataFrame) -> pd.DataFrame:
    if screener is None or screener.empty:
        return pd.DataFrame() if screener is None else screener.copy()
    return screener[
        screener["sw_decision"].isin(
            {"PRIORITY WATCH", "READY", "ACTIVE", "WATCH", "FIRST WAVE"}
        )
    ].copy()

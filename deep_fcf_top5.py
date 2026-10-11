from __future__ import annotations

import math
from typing import Any

import pandas as pd


DEEP_FCF_WEIGHTS = {
    "fcf_per_share_3y_cagr_score": 20.0,
    "fcf_per_share_yoy_score": 15.0,
    "cfo_yoy_score": 15.0,
    "roic_score": 15.0,
    "fcf_margin_score": 15.0,
    "cash_conversion_score": 10.0,
    "net_cash_ratio_score": 5.0,
    "dilution_score": 5.0,
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


def _valuation_score(p_fcf: Any) -> float | None:
    value = _num(p_fcf)
    if value is None or value <= 0:
        return None
    if value <= 10:
        return 100.0
    if value <= 15:
        return 85.0
    if value <= 20:
        return 70.0
    if value <= 30:
        return 55.0
    if value <= 40:
        return 40.0
    if value <= 60:
        return 20.0
    return 5.0


def _execution_ready(row: pd.Series) -> bool:
    state = str(row.get("second_wave_state") or "")
    decision = str(row.get("sw_decision") or "")
    return (
        state == "RE-EXP"
        or decision == "ACTIVE"
        or decision == "PRIORITY WATCH"
        or state == "RE-WATCH READY"
        or decision == "READY"
    )


def _confidence(coverage: float) -> str:
    if coverage >= 0.75:
        return "HIGH"
    if coverage >= 0.60:
        return "MEDIUM"
    return "LOW"


def build_deep_fcf_top5(
    screener: pd.DataFrame | None,
    *,
    limit: int = 5,
    min_coverage: float = 0.45,
) -> pd.DataFrame:
    """Rank latest ME names by cash-generation quality without bypassing entry gates.

    Deep FCF intentionally uses the point-in-time financial components already
    produced by the ME pipeline. It is a research-priority ranking, not an
    execution signal. Entry Hunter / Pre-Trade remain gated by the existing
    READY / PRIORITY / RE-EXP maturity rules.
    """
    columns = [
        "rank",
        "ticker",
        "company_name",
        "deep_fcf_score",
        "deep_fcf_coverage",
        "deep_fcf_confidence",
        "fcf_ttm",
        "fcf_margin",
        "cash_conversion",
        "p_fcf",
        "fcf_quality_grade",
        "fcf_basis",
        "second_wave_state",
        "sw_decision",
        "execution_ready",
        "reason",
    ]
    if screener is None or screener.empty or "ticker" not in screener.columns:
        return pd.DataFrame(columns=columns)

    out = screener.copy()
    if "trade_date" in out.columns:
        out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
        out = out.dropna(subset=["trade_date"])
        if out.empty:
            return pd.DataFrame(columns=columns)
        latest = out["trade_date"].max()
        out = out[out["trade_date"] == latest].copy()

    out["ticker"] = (
        out["ticker"].astype(str).str.replace(".0", "", regex=False).str.zfill(4)
    )
    fcf_ttm = pd.to_numeric(out.get("fcf_ttm"), errors="coerce")
    fcf_complete = out.get(
        "fcf_ttm_complete",
        pd.Series(True, index=out.index),
    ).fillna(False).astype(bool)
    out = out[(fcf_ttm > 0) & fcf_complete].copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    deep_scores: list[float | None] = []
    coverages: list[float] = []
    confidences: list[str] = []
    valuation_scores: list[float | None] = []

    total_weight = sum(DEEP_FCF_WEIGHTS.values())

    for _, row in out.iterrows():
        used = 0.0
        weighted = 0.0
        for col, weight in DEEP_FCF_WEIGHTS.items():
            value = _num(row.get(col))
            if value is None:
                continue
            used += weight
            weighted += max(0.0, min(100.0, value)) * weight

        coverage = used / total_weight if total_weight else 0.0
        core = weighted / used if used > 0 else None
        val = _valuation_score(row.get("p_fcf"))
        if core is None or coverage < float(min_coverage):
            deep = None
        elif val is None:
            deep = core
        else:
            deep = core * 0.85 + val * 0.15

        deep_scores.append(deep)
        coverages.append(coverage)
        confidences.append(_confidence(coverage))
        valuation_scores.append(val)

    out["deep_fcf_score"] = deep_scores
    out["deep_fcf_coverage"] = coverages
    out["deep_fcf_confidence"] = confidences
    out["_valuation_score"] = valuation_scores
    out = out.dropna(subset=["deep_fcf_score"]).copy()
    if out.empty:
        return pd.DataFrame(columns=columns)

    out["execution_ready"] = out.apply(_execution_ready, axis=1)

    reasons: list[str] = []
    for _, row in out.iterrows():
        bits: list[str] = []
        margin = _num(row.get("fcf_margin"))
        conversion = _num(row.get("cash_conversion"))
        p_fcf = _num(row.get("p_fcf"))
        quality = str(row.get("fcf_quality_grade") or "")
        if margin is not None:
            bits.append(f"FCF Margin {margin * 100:.1f}%")
        if conversion is not None:
            bits.append(f"Cash Conv {conversion:.2f}x")
        if p_fcf is not None:
            bits.append(f"P/FCF {p_fcf:.1f}x")
        if quality:
            bits.append(f"Quality {quality}")
        bits.append(
            "ENTRY GATE OPEN" if bool(row.get("execution_ready")) else "ENTRY GATE WAIT"
        )
        reasons.append(" / ".join(bits))
    out["reason"] = reasons

    out = out.sort_values(
        ["deep_fcf_score", "deep_fcf_coverage", "_valuation_score", "ticker"],
        ascending=[False, False, False, True],
        na_position="last",
    ).head(max(1, int(limit))).copy()
    out["rank"] = range(1, len(out) + 1)

    for col in columns:
        if col not in out.columns:
            out[col] = None
    return out[columns].reset_index(drop=True)

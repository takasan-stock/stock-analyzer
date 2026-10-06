from __future__ import annotations

from typing import Any, Mapping


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        out = float(value)
        if out != out:
            return None
        return out
    except (TypeError, ValueError):
        return None


def _clip(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _learning_confidence(candidate: Mapping[str, Any]) -> str:
    labels = {
        str(candidate.get("adaptive_confidence", "") or "").upper(),
        str(candidate.get("state_confidence", "") or "").upper(),
        str(candidate.get("trait_confidence", "") or "").upper(),
    }
    if "ADAPTIVE" in labels:
        return "GOOD"
    if "WARMING" in labels:
        return "MEDIUM"
    if "LOW SAMPLE" in labels:
        return "LOW"
    if "DATA BUILDING" in labels:
        return "BUILDING"
    return "BASE"


def _rating(score: float) -> str:
    if score >= 95:
        return "S"
    if score >= 85:
        return "A+"
    if score >= 78:
        return "A"
    if score >= 70:
        return "B+"
    if score >= 60:
        return "B"
    return "C"


def build_entry_opportunity(
    candidate: Mapping[str, Any],
    entry_snapshot: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Unify candidate quality and intraday execution into one score.

    The candidate score already contains the ME / Short Cover source quality,
    source-level adaptive evidence, setup-level evidence, and small trait
    adjustments. This function deliberately does not re-score those inputs.
    It only combines that pre-market candidate quality with the live Entry
    Hunter execution score and then applies conservative state gates.

    Result:
      - opportunity_score: 0..100
      - opportunity_rating: S / A+ / A / B+ / B / C
      - opportunity_action: practical next action
      - opportunity_reason: short explainable summary
      - opportunity_coverage: 55 or 100
      - learning_confidence: maturity of adaptive evidence
    """
    entry = entry_snapshot or {}

    candidate_score = _num(candidate.get("source_score"))
    if candidate_score is None:
        candidate_score = 0.0
    candidate_score = _clip(candidate_score)

    entry_score = _num(entry.get("score"))
    status = str(entry.get("status", "⚪ NO DATA") or "⚪ NO DATA")

    if entry_score is None:
        base = candidate_score
        coverage = 55
    else:
        entry_score = _clip(entry_score)
        base = candidate_score * 0.60 + entry_score * 0.40
        coverage = 100

    if status == "🟢 ENTRY CONFIRMED":
        score = base + 6.0
    elif status == "🟢 ENTRY READY":
        score = base + 4.0
    elif status == "🟦 MONITORING":
        score = base + 1.0
    elif status == "🟡 WAIT":
        score = min(base - 3.0, 79.0)
    elif status == "🟡 WEAKENING":
        score = min(base - 10.0, 59.0)
    elif status in {"🔴 CANCEL", "🔴 EXIT WATCH"}:
        score = min(base - 20.0, 39.0)
    else:
        score = min(base - 8.0, 74.0)
        coverage = min(coverage, 55)

    score = _clip(score)
    rating = _rating(score)

    if status in {"🔴 CANCEL", "🔴 EXIT WATCH"}:
        action = "AVOID"
    elif status == "🟡 WEAKENING":
        action = "CAUTION"
    elif status == "⚪ NO DATA":
        action = "WAIT DATA"
    elif status == "🟢 ENTRY CONFIRMED":
        action = "PRIORITY CONFIRMED" if score >= 85 else "CONFIRMED"
    elif status == "🟢 ENTRY READY":
        if score >= 85:
            action = "ENTRY PRIORITY"
        elif score >= 78:
            action = "ENTRY READY"
        else:
            action = "WATCH"
    elif status == "🟡 WAIT":
        if score >= 78:
            action = "HIGH WATCH"
        elif score >= 70:
            action = "WATCH"
        else:
            action = "WAIT"
    else:
        action = "WATCH" if score >= 70 else "WAIT"

    reasons: list[str] = []
    source = str(candidate.get("source", "") or "")
    signal_key = str(candidate.get("signal_key", "") or "")

    if source == "SHORT+ME":
        reasons.append("SHORT+ME")
    elif source:
        reasons.append(source)

    if signal_key:
        reasons.append(signal_key)

    if candidate_score >= 90:
        reasons.append("候補品質STRONG")
    elif candidate_score >= 80:
        reasons.append("候補品質GOOD")

    if status == "🟢 ENTRY READY":
        reasons.append("寄り後READY")
    elif status == "🟢 ENTRY CONFIRMED":
        reasons.append("寄り後CONFIRMED")
    elif status == "🟡 WAIT":
        reasons.append("ENTRY条件待ち")
    elif status in {"🔴 CANCEL", "🔴 EXIT WATCH"}:
        reasons.append("初動崩れ")
    elif status == "🟡 WEAKENING":
        reasons.append("勢い低下")

    source_bonus = _num(candidate.get("adaptive_bonus")) or 0.0
    state_bonus = _num(candidate.get("state_bonus")) or 0.0
    trait_bonus = _num(candidate.get("trait_bonus")) or 0.0

    if source_bonus >= 0.5:
        reasons.append("Source実績+")
    elif source_bonus <= -0.5:
        reasons.append("Source実績-")

    if state_bonus >= 0.5:
        reasons.append("Setup実績+")
    elif state_bonus <= -0.5:
        reasons.append("Setup実績-")

    if trait_bonus >= 0.5:
        reasons.append("銘柄特性+")
    elif trait_bonus <= -0.5:
        reasons.append("銘柄特性-")

    if coverage < 100:
        reasons.append("Intraday未確定")

    return {
        "opportunity_score": round(score, 1),
        "opportunity_rating": rating,
        "opportunity_action": action,
        "opportunity_reason": " / ".join(reasons),
        "opportunity_coverage": int(coverage),
        "candidate_score": round(candidate_score, 1),
        "live_entry_score": None if entry_score is None else round(entry_score, 1),
        "learning_confidence": _learning_confidence(candidate),
    }

from __future__ import annotations

import datetime
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StockActionStatus:
    stage: str
    label: str
    icon: str
    readiness_score: int
    confidence: str
    summary: str
    reasons: tuple[str, ...]
    blockers: tuple[str, ...]
    next_page: str
    next_label: str
    next_icon: str


def _num(value: Any) -> float | None:
    try:
        if value is None:
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _text(value: Any) -> str:
    return str(value or "").strip()


def _pct_like(value: Any) -> float | None:
    if value is None:
        return None

    text = str(value).strip()
    has_percent = text.endswith("%")
    if has_percent:
        text = text[:-1].strip()

    try:
        num = float(text.replace(",", ""))
    except (TypeError, ValueError):
        return None

    if has_percent:
        return num

    # Accept either 0.25 or 25.0 style inputs.
    if -1.5 <= num <= 1.5:
        return num * 100.0
    return num


def evaluate_stock_action_status(
    *,
    portfolio_row: dict[str, Any] | None = None,
    me_row: dict[str, Any] | None = None,
    short_row: dict[str, Any] | None = None,
    live_entry: dict[str, Any] | None = None,
    earnings_date: datetime.date | None = None,
    today: datetime.date | None = None,
) -> StockActionStatus:
    """Summarize research/monitoring maturity and the next screen to inspect.

    This is deliberately not a buy/sell recommendation. The readiness score
    measures how much supporting evidence is present across growth, FCF/ME,
    entry/flow and event-risk checks.
    """

    portfolio_row = portfolio_row or {}
    me_row = me_row or {}
    short_row = short_row or {}
    live_entry = live_entry or {}
    today = today or datetime.date.today()

    score = 0.0
    reasons: list[str] = []
    blockers: list[str] = []

    # Growth / portfolio context: 20 points.
    sales_cagr = _pct_like(portfolio_row.get("売上5y CAGR"))
    if sales_cagr is not None:
        if sales_cagr >= 15:
            score += 12
            reasons.append(f"売上5y CAGR {sales_cagr:.1f}%")
        elif sales_cagr >= 5:
            score += 7
            reasons.append(f"売上5y CAGR {sales_cagr:.1f}%")
        else:
            score += 3
            blockers.append("成長率が低め")
    else:
        blockers.append("売上成長率が未登録")

    status = _text(portfolio_row.get("ステータス"))
    if status in {"保有中", "打診買い", "監視中"}:
        score += 4

    investor_memo = _text(portfolio_row.get("投資家メモ"))
    if investor_memo:
        score += 4

    # FCF / quality: 25 points.
    fcf_score = _num(me_row.get("fcf_engine_score"))
    fcf_coverage = _num(me_row.get("fcf_engine_coverage"))
    quality = _text(me_row.get("fcf_quality_grade")).upper()

    if fcf_score is not None:
        if fcf_score >= 70:
            score += 12
            reasons.append(f"FCF Engine {fcf_score:.0f}")
        elif fcf_score >= 55:
            score += 8
            reasons.append(f"FCF Engine {fcf_score:.0f}")
        else:
            score += 3
            blockers.append("FCF Engineが弱め")
    else:
        blockers.append("FCF評価が未取得")

    if fcf_coverage is not None:
        coverage_pct = fcf_coverage * 100 if fcf_coverage <= 1.5 else fcf_coverage
        if coverage_pct >= 80:
            score += 6
        elif coverage_pct >= 60:
            score += 4
        else:
            score += 1
            blockers.append("FCFデータカバレッジが低い")

    if quality in {"A", "B"}:
        score += 7
        reasons.append(f"FCF Quality {quality}")
    elif quality == "C":
        score += 4
    elif quality:
        blockers.append(f"FCF Quality {quality}")

    # ME / trend structure: 25 points.
    mex_score = _num(me_row.get("mex_score"))
    sw_decision = _text(me_row.get("sw_decision")).upper()
    me_state = _text(me_row.get("second_wave_state") or me_row.get("state_confirmed")).upper()
    rs_rating = _num(me_row.get("rs_rating"))

    if mex_score is not None:
        if mex_score >= 75:
            score += 10
            reasons.append(f"MEX {mex_score:.0f}")
        elif mex_score >= 60:
            score += 7
            reasons.append(f"MEX {mex_score:.0f}")
        elif mex_score >= 45:
            score += 4
        else:
            score += 1

    if sw_decision in {"ACTIVE", "READY", "PRIORITY WATCH"}:
        score += 8
        reasons.append(f"ME {sw_decision}")
    elif sw_decision in {"WATCH", "FIRST WAVE"}:
        score += 5
    elif sw_decision == "PASS":
        blockers.append("ME判定はPASS")

    if any(token in me_state for token in ("READY", "RE-EXP", "EXPANSION", "IGNITION", "SPRING")):
        score += 4
    elif me_state and "UNAVAILABLE" in me_state:
        blockers.append("ME状態の判定材料不足")

    if rs_rating is not None:
        if rs_rating >= 80:
            score += 3
        elif rs_rating >= 65:
            score += 2

    # Entry / flow: 20 points.
    live_status = _text(live_entry.get("status")).upper()
    alert_tier = _text(short_row.get("alert_tier"))
    alert_score = _num(short_row.get("alert_score"))
    phase = _text(short_row.get("phase"))

    if any(token in live_status for token in ("READY", "CONFIRMED")):
        score += 12
        reasons.append(f"Entry {live_status}")
    elif any(token in live_status for token in ("PREP", "WATCH", "WAIT")):
        score += 6

    if "A+" in alert_tier:
        score += 6
        reasons.append("Short Cover A+")
    elif "A " in alert_tier or alert_tier.startswith("🔥 A"):
        score += 4
    elif "B" in alert_tier:
        score += 2

    if alert_score is not None and alert_score >= 75:
        score += 2

    if phase and any(token in phase for token in ("SQUEEZE", "CONFIRMED", "EARLY")):
        reasons.append(phase)

    # Event risk can cap action maturity.
    earnings_days: int | None = None
    if earnings_date:
        earnings_days = (earnings_date - today).days
        if 0 <= earnings_days <= 2:
            blockers.insert(0, f"決算まで{earnings_days}日")
        elif 3 <= earnings_days <= 7:
            blockers.append(f"決算まで{earnings_days}日")

    score = max(0, min(100, int(round(score))))

    # Confidence is based on coverage rather than bullishness.
    coverage_signals = 0
    coverage_signals += 1 if sales_cagr is not None else 0
    coverage_signals += 1 if fcf_score is not None else 0
    coverage_signals += 1 if mex_score is not None or sw_decision else 0
    coverage_signals += 1 if live_status or alert_tier else 0
    coverage_signals += 1 if earnings_date is not None else 0

    if coverage_signals >= 5:
        confidence = "HIGH"
    elif coverage_signals >= 3:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # Event risk overrides the next action regardless of score.
    if earnings_days is not None and 0 <= earnings_days <= 2:
        return StockActionStatus(
            stage="EVENT_CHECK",
            label="決算イベント確認",
            icon="📅",
            readiness_score=score,
            confidence=confidence,
            summary="決算が直前です。エントリー判断より先にイベントリスクを確認します。",
            reasons=tuple(reasons[:5]),
            blockers=tuple(blockers[:5]),
            next_page="pages/16_Stock_Detail.py",
            next_label="材料・イベントを確認",
            next_icon="📰",
        )

    entry_ready = any(token in live_status for token in ("READY", "CONFIRMED"))
    me_ready = sw_decision in {"ACTIVE", "READY"} or (
        mex_score is not None and mex_score >= 70
    )

    if entry_ready and score >= 60:
        stage = "PRETRADE"
        label = "購入前チェックへ"
        icon = "🛡️"
        summary = "Entry条件が進んでいます。発注前のリスク・損切り・決算確認へ進む段階です。"
        next_page = "pages/9_Pre_Trade_Check.py"
        next_label = "購入前チェックを開く"
        next_icon = "🛡️"
    elif me_ready and score >= 45:
        stage = "ENTRY_MONITOR"
        label = "ENTRY監視"
        icon = "🎯"
        summary = "ME条件が進んでいます。Entry Hunterで買い位置と場中状態を追う段階です。"
        next_page = "pages/8_Short_Cover_Hunter.py"
        next_label = "Entry Hunterを開く"
        next_icon = "🎯"
    elif mex_score is not None or sw_decision:
        stage = "MONITOR"
        label = "監視継続"
        icon = "👀"
        summary = "候補として監視を継続し、MEの再加速や出来高・RSの改善を待つ段階です。"
        next_page = "pages/14_Multiple_Expansion_Hunter.py"
        next_label = "ME Hunterで確認"
        next_icon = "🚀"
    elif coverage_signals <= 2:
        stage = "DATA_CHECK"
        label = "データ確認"
        icon = "🧩"
        summary = "判定材料が不足しています。基本情報・財務・分析レポートを先に補完します。"
        next_page = "dashboard_app.py"
        next_label = "AI分析・登録データを確認"
        next_icon = "🤖"
    else:
        stage = "RESEARCH"
        label = "研究優先"
        icon = "🔎"
        summary = "まだ実行段階ではありません。FCF・成長・材料を深掘りして監視条件を固めます。"
        next_page = "dashboard_app.py"
        next_label = "AI分析で深掘り"
        next_icon = "🤖"

    return StockActionStatus(
        stage=stage,
        label=label,
        icon=icon,
        readiness_score=score,
        confidence=confidence,
        summary=summary,
        reasons=tuple(reasons[:5]),
        blockers=tuple(blockers[:5]),
        next_page=next_page,
        next_label=next_label,
        next_icon=next_icon,
    )

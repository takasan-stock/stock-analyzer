from __future__ import annotations

import math
from typing import Any

import pandas as pd


def _num(value) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def _clean_ohlcv(frame: pd.DataFrame | None) -> pd.DataFrame:
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = out.columns.get_level_values(0)
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(set(out.columns)):
        return pd.DataFrame()
    out = out.sort_index()
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["Close"])


def _return_pct(series: pd.Series, lookback: int) -> float | None:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if len(s) <= lookback:
        return None
    start = _num(s.iloc[-lookback - 1])
    end = _num(s.iloc[-1])
    if start is None or end is None or start <= 0:
        return None
    return (end / start - 1.0) * 100.0


def calc_rs_proxy(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
) -> dict[str, float | None]:
    """Return a 0-100 market-relative strength proxy.

    This is intentionally called a proxy, not an IBD percentile RS Rating.
    The score compares 3-month and 6-month returns with a benchmark.
    """
    d = _clean_ohlcv(daily)
    b = _clean_ohlcv(benchmark)
    if d.empty:
        return {"rs_proxy": None, "stock_ret63": None, "stock_ret126": None,
                "bench_ret63": None, "bench_ret126": None}

    s63 = _return_pct(d["Close"], 63)
    s126 = _return_pct(d["Close"], 126)
    b63 = _return_pct(b["Close"], 63) if not b.empty else None
    b126 = _return_pct(b["Close"], 126) if not b.empty else None

    if s63 is None and s126 is None:
        score = None
    else:
        ex63 = (s63 or 0.0) - (b63 or 0.0)
        ex126 = (s126 or 0.0) - (b126 or 0.0)
        # 0% excess return -> 50. Strong positive/negative excess saturates gradually.
        weighted_excess = ex63 * 0.60 + ex126 * 0.40
        score = max(0.0, min(100.0, 50.0 + weighted_excess * 1.8))
        score = round(score, 1)

    return {
        "rs_proxy": score,
        "stock_ret63": s63,
        "stock_ret126": s126,
        "bench_ret63": b63,
        "bench_ret126": b126,
    }


def build_technical_snapshot(
    daily: pd.DataFrame,
    benchmark: pd.DataFrame | None = None,
) -> dict[str, Any]:
    d = _clean_ohlcv(daily)
    if len(d) < 30:
        return {"data_ok": False, "reason": "日足データ不足"}

    close = d["Close"]
    high = d["High"]
    low = d["Low"]
    volume = d["Volume"]

    ema20 = close.ewm(span=20, adjust=False).mean()
    sma50 = close.rolling(50).mean()
    sma150 = close.rolling(150).mean()
    sma200 = close.rolling(200).mean()

    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low).abs(),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    atr14 = tr.rolling(14).mean()

    price = _num(close.iloc[-1])
    open_px = _num(d["Open"].iloc[-1])
    prev = _num(close.iloc[-2]) if len(close) >= 2 else None
    ema20_v = _num(ema20.iloc[-1])
    sma50_v = _num(sma50.iloc[-1])
    sma150_v = _num(sma150.iloc[-1])
    sma200_v = _num(sma200.iloc[-1])
    atr_v = _num(atr14.iloc[-1])

    high52 = _num(high.tail(252).max())
    low52 = _num(low.tail(252).min())
    prior20_high = _num(high.shift(1).tail(20).max())
    swing10 = _num(low.shift(1).tail(10).min())

    avg_vol20 = _num(volume.shift(1).tail(20).mean())
    latest_vol = _num(volume.iloc[-1])
    vol_ratio = (
        latest_vol / avg_vol20
        if latest_vol is not None and avg_vol20 not in (None, 0)
        else None
    )

    gap_pct = (
        (open_px / prev - 1.0) * 100.0
        if open_px is not None and prev not in (None, 0)
        else None
    )
    change_pct = (
        (price / prev - 1.0) * 100.0
        if price is not None and prev not in (None, 0)
        else None
    )
    ema20_gap_pct = (
        (price / ema20_v - 1.0) * 100.0
        if price is not None and ema20_v not in (None, 0)
        else None
    )

    sma200_20ago = _num(sma200.iloc[-21]) if len(sma200) >= 21 else None
    sma200_rising = bool(
        sma200_v is not None and sma200_20ago is not None and sma200_v > sma200_20ago
    )

    stage2 = bool(
        all(x is not None for x in [price, sma50_v, sma150_v, sma200_v, high52, low52])
        and price > sma50_v > sma150_v > sma200_v
        and sma200_rising
        and price >= low52 * 1.30
        and price >= high52 * 0.75
    )
    stage4 = bool(
        price is not None
        and sma50_v is not None
        and sma200_v is not None
        and price < sma50_v < sma200_v
    )
    if stage2:
        stage = "🟢 Stage 2"
    elif stage4:
        stage = "🔴 Stage 4"
    else:
        stage = "🟡 Transition"

    breakout20 = bool(
        price is not None and prior20_high is not None and price > prior20_high
    )
    pullback20 = bool(
        len(d) >= 2
        and price is not None
        and ema20_v is not None
        and _num(low.iloc[-1]) is not None
        and low.iloc[-1] <= ema20_v * 1.015
        and price >= ema20_v
        and price >= open_px
    )

    avg_turnover20 = (
        _num((close.shift(1).tail(20) * volume.shift(1).tail(20)).mean())
        if len(d) >= 21
        else None
    )

    rs = calc_rs_proxy(d, benchmark)

    stop_suggestions = {}
    if price is not None:
        if swing10 is not None and swing10 < price:
            stop_suggestions["直近10日安値"] = round(swing10 * 0.995, 1)
        if ema20_v is not None and ema20_v < price:
            stop_suggestions["20EMA下"] = round(ema20_v * 0.985, 1)
        if atr_v is not None and atr_v > 0:
            atr_stop = price - atr_v * 1.5
            if atr_stop > 0 and atr_stop < price:
                stop_suggestions["ATR 1.5倍"] = round(atr_stop, 1)

    return {
        "data_ok": True,
        "price": price,
        "open": open_px,
        "prev_close": prev,
        "change_pct": change_pct,
        "gap_pct": gap_pct,
        "ema20": ema20_v,
        "sma50": sma50_v,
        "sma150": sma150_v,
        "sma200": sma200_v,
        "sma200_rising": sma200_rising,
        "atr14": atr_v,
        "ema20_gap_pct": ema20_gap_pct,
        "high52": high52,
        "low52": low52,
        "prior20_high": prior20_high,
        "swing10_low": swing10,
        "breakout20": breakout20,
        "pullback20": pullback20,
        "volume_ratio": vol_ratio,
        "avg_turnover20": avg_turnover20,
        "stage2": stage2,
        "stage4": stage4,
        "stage": stage,
        "stop_suggestions": stop_suggestions,
        **rs,
    }


def calc_risk_reward(entry, stop, target) -> dict[str, float | None]:
    entry = _num(entry)
    stop = _num(stop)
    target = _num(target)
    if entry is None or entry <= 0:
        return {"rr": None, "risk_per_share": None, "reward_per_share": None,
                "risk_pct": None, "reward_pct": None}

    risk = entry - stop if stop is not None else None
    reward = target - entry if target is not None else None
    rr = (
        reward / risk
        if risk is not None and reward is not None and risk > 0 and reward > 0
        else None
    )
    return {
        "rr": rr,
        "risk_per_share": risk,
        "reward_per_share": reward,
        "risk_pct": (risk / entry * 100.0) if risk is not None and risk > 0 else None,
        "reward_pct": (reward / entry * 100.0) if reward is not None and reward > 0 else None,
    }


def calc_position_size(
    capital,
    risk_percent,
    entry,
    stop,
    lot_size: int = 100,
) -> dict[str, float | int | None]:
    capital = _num(capital)
    risk_percent = _num(risk_percent)
    entry = _num(entry)
    stop = _num(stop)
    if (
        capital is None or capital <= 0
        or risk_percent is None or risk_percent <= 0
        or entry is None or entry <= 0
        or stop is None or stop <= 0
        or stop >= entry
    ):
        return {
            "max_loss": None,
            "raw_shares": None,
            "shares": 0,
            "position_value": 0.0,
            "actual_max_loss": 0.0,
        }

    max_loss = capital * risk_percent / 100.0
    per_share = entry - stop
    raw = max_loss / per_share
    lot = max(1, int(lot_size))
    shares = int(math.floor(raw / lot) * lot)
    position_value = shares * entry
    actual_max_loss = shares * per_share

    return {
        "max_loss": max_loss,
        "raw_shares": raw,
        "shares": shares,
        "position_value": position_value,
        "actual_max_loss": actual_max_loss,
    }


def _entry_hunter_points(entry_hunter: dict | None) -> tuple[float, list[str], list[str]]:
    if not entry_hunter:
        return 0.0, [], ["Entry Hunter未連携"]

    status = str(entry_hunter.get("status", ""))
    score = _num(entry_hunter.get("score"))
    positive, risks = [], []

    if "ENTRY READY" in status:
        return 8.0, ["Entry Hunter READY"], risks
    if "CANCEL" in status:
        return 0.0, positive, ["Entry Hunter CANCEL"]
    if "WAIT" in status:
        if score is not None and score >= 50:
            return 4.0, ["Entry Hunter条件形成中"], risks
        return 2.0, positive, ["Entry Hunter待ち"]
    return 0.0, positive, ["Entry Hunterデータ不足"]


def evaluate_pretrade(
    technical: dict,
    *,
    entry,
    stop,
    target,
    capital,
    risk_percent: float = 1.0,
    earnings_days: int | None = None,
    entry_hunter: dict | None = None,
    pattern_flags: dict[str, bool] | None = None,
    lot_size: int = 100,
) -> dict[str, Any]:
    """Score a long-side pre-trade check.

    Scores are execution-support heuristics, not investment recommendations.
    Missing inputs do not receive full points.
    """
    if not technical or not technical.get("data_ok"):
        return {
            "total_score": 0.0,
            "setup_score": 0.0,
            "entry_score": 0.0,
            "risk_score": 0.0,
            "verdict": "⚪ NO DATA",
            "blocked": True,
            "block_reasons": ["日足データ不足"],
            "positive": [],
            "risks": ["日足データ不足"],
        }

    entry = _num(entry)
    stop = _num(stop)
    target = _num(target)
    pattern_flags = pattern_flags or {}
    positive: list[str] = []
    risks: list[str] = []
    blocks: list[str] = []

    rr_info = calc_risk_reward(entry, stop, target)
    pos = calc_position_size(capital, risk_percent, entry, stop, lot_size=lot_size)

    # ---------------- Setup Quality / 40 ----------------
    setup = 0.0
    if technical.get("stage2"):
        setup += 10
        positive.append("Stage 2")
    elif technical.get("stage4"):
        risks.append("Stage 4")
    else:
        risks.append("Stage移行期")

    price = _num(technical.get("price"))
    ema20 = _num(technical.get("ema20"))
    sma50 = _num(technical.get("sma50"))
    sma200 = _num(technical.get("sma200"))
    high52 = _num(technical.get("high52"))

    if price is not None and ema20 is not None and price > ema20:
        setup += 4
        positive.append("20EMA上")
    if price is not None and sma50 is not None and price > sma50:
        setup += 4
        positive.append("50MA上")
    if sma50 is not None and sma200 is not None and sma50 > sma200:
        setup += 5
        positive.append("50MA > 200MA")
    if technical.get("sma200_rising"):
        setup += 4
        positive.append("200MA上向き")
    if price is not None and high52 is not None and high52 > 0 and price >= high52 * 0.85:
        setup += 4
        positive.append("52週高値圏")

    rs_proxy = _num(technical.get("rs_proxy"))
    if rs_proxy is not None:
        if rs_proxy >= 70:
            setup += 3
            positive.append(f"RS Proxy {rs_proxy:.0f}")
        elif rs_proxy >= 55:
            setup += 1.5
        elif rs_proxy < 40:
            risks.append(f"RS Proxy弱め {rs_proxy:.0f}")
    else:
        risks.append("RS Proxy未取得")

    vr = _num(technical.get("volume_ratio"))
    if vr is not None:
        if vr >= 1.5:
            setup += 3
            positive.append(f"出来高 {vr:.1f}x")
        elif vr >= 1.0:
            setup += 1
    else:
        risks.append("出来高比未取得")

    confirmed_patterns = [name for name, flag in pattern_flags.items() if bool(flag)]
    if confirmed_patterns:
        setup += min(3.0, float(len(confirmed_patterns)))
        positive.append("形状確認: " + " / ".join(confirmed_patterns))
    setup = min(40.0, setup)

    # ---------------- Entry Timing / 35 ----------------
    timing = 0.0
    dist = _num(technical.get("ema20_gap_pct"))
    if dist is not None:
        if 0 <= dist <= 3:
            timing += 8
            positive.append(f"20EMA乖離 {dist:+.1f}%")
        elif 3 < dist <= 5:
            timing += 6
        elif 5 < dist <= 7:
            timing += 3
            risks.append(f"20EMA乖離 {dist:+.1f}%")
        elif 7 < dist <= 10:
            timing += 1
            risks.append(f"やや伸び過ぎ {dist:+.1f}%")
        elif dist > 10:
            risks.append(f"高値追い注意 {dist:+.1f}%")
        elif -2 <= dist < 0:
            timing += 3
            risks.append("20EMA付近の下側")

    if technical.get("breakout20"):
        timing += 6
        positive.append("20日高値ブレイク")
    if technical.get("pullback20"):
        timing += 5
        positive.append("20EMA押し目反発")

    if vr is not None:
        if vr >= 2.0:
            timing += 5
        elif vr >= 1.5:
            timing += 4
        elif vr >= 1.0:
            timing += 2
        elif vr < 0.7:
            risks.append("出来高弱い")

    hunter_pts, hunter_positive, hunter_risks = _entry_hunter_points(entry_hunter)
    timing += hunter_pts
    positive.extend(hunter_positive)
    risks.extend(hunter_risks)

    gap = _num(technical.get("gap_pct"))
    if gap is not None:
        if -1 <= gap <= 3:
            timing += 3
        elif 3 < gap <= 5:
            timing += 2
        elif gap > 5:
            risks.append(f"GU大 {gap:+.1f}%")
        elif gap < -3:
            risks.append(f"GD {gap:+.1f}%")
    timing = min(35.0, timing)

    # ---------------- Risk Control / 25 ----------------
    risk_score = 0.0
    rr = _num(rr_info.get("rr"))
    if rr is not None:
        if rr >= 3:
            risk_score += 9
            positive.append(f"RR 1:{rr:.1f}")
        elif rr >= 2:
            risk_score += 8
            positive.append(f"RR 1:{rr:.1f}")
        elif rr >= 1.5:
            risk_score += 4
            risks.append(f"RR低め 1:{rr:.1f}")
        else:
            risks.append(f"RR不足 1:{rr:.1f}")
    else:
        risks.append("RR計算不可")

    risk_pct = _num(rr_info.get("risk_pct"))
    if risk_pct is not None:
        if 2 <= risk_pct <= 6:
            risk_score += 5
            positive.append(f"損切り幅 {risk_pct:.1f}%")
        elif 0 < risk_pct <= 8:
            risk_score += 4
        elif 0 < risk_pct <= 10:
            risk_score += 2
            risks.append(f"損切り幅大 {risk_pct:.1f}%")
        elif risk_pct > 10:
            risks.append(f"損切り幅過大 {risk_pct:.1f}%")

    if earnings_days is None:
        risk_score += 2
        risks.append("決算日未確認")
    elif earnings_days > 10:
        risk_score += 4
    elif 6 <= earnings_days <= 10:
        risk_score += 3
        risks.append(f"決算まで{earnings_days}日")
    elif 3 <= earnings_days <= 5:
        risk_score += 1
        risks.append(f"決算接近 {earnings_days}日")
    elif earnings_days >= 0:
        risks.append(f"決算直前 {earnings_days}日")

    atr = _num(technical.get("atr14"))
    rps = _num(rr_info.get("risk_per_share"))
    if atr is not None and atr > 0 and rps is not None and rps > 0:
        atr_multiple = rps / atr
        if 1.0 <= atr_multiple <= 2.5:
            risk_score += 3
            positive.append(f"Stop幅 {atr_multiple:.1f} ATR")
        elif 0.75 <= atr_multiple <= 3.0:
            risk_score += 2
        else:
            risk_score += 1
            risks.append(f"Stop幅 {atr_multiple:.1f} ATR")
    else:
        atr_multiple = None

    turnover = _num(technical.get("avg_turnover20"))
    if turnover is not None:
        if turnover >= 100_000_000:
            risk_score += 2
        elif turnover >= 30_000_000:
            risk_score += 1
        else:
            risks.append("売買代金小さめ")

    shares = int(pos.get("shares") or 0)
    if shares >= lot_size:
        risk_score += 2
        positive.append(f"許容損失内 {shares}株")
    else:
        risks.append("許容損失内で単元株を持てない")

    risk_score = min(25.0, risk_score)

    # ---------------- Hard Reject ----------------
    if entry is None or entry <= 0:
        blocks.append("Entry未設定")
    if stop is None or entry is None or stop <= 0 or stop >= entry:
        blocks.append("Stop未設定/不正")
    if target is None or entry is None or target <= entry:
        blocks.append("Target未設定/不正")
    if rr is None or rr < 1.5:
        blocks.append("RR 1:1.5未満")
    if technical.get("stage4"):
        blocks.append("Stage 4")
    if dist is not None and dist > 12:
        blocks.append("20EMA乖離 +12%超")
    change = _num(technical.get("change_pct"))
    if change is not None and change >= 15:
        blocks.append("当日+15%以上の急騰")
    if earnings_days == 0:
        blocks.append("決算当日")
    if entry_hunter and "CANCEL" in str(entry_hunter.get("status", "")):
        blocks.append("Entry Hunter CANCEL")

    total = round(setup + timing + risk_score, 1)
    blocked = bool(blocks)

    if blocked:
        verdict = "🔴 TRADE BLOCKED"
    elif total >= 85:
        verdict = "🟢 TRADE READY"
    elif total >= 75:
        verdict = "🟢 ENTRY CANDIDATE"
    elif total >= 65:
        verdict = "🟡 WAIT"
    elif total >= 50:
        verdict = "🟠 WATCH ONLY"
    else:
        verdict = "🔴 SKIP"

    return {
        "total_score": total,
        "setup_score": round(setup, 1),
        "entry_score": round(timing, 1),
        "risk_score": round(risk_score, 1),
        "verdict": verdict,
        "blocked": blocked,
        "block_reasons": list(dict.fromkeys(blocks)),
        "positive": list(dict.fromkeys(positive)),
        "risks": list(dict.fromkeys(risks)),
        "rr": rr,
        "risk_per_share": rr_info.get("risk_per_share"),
        "reward_per_share": rr_info.get("reward_per_share"),
        "risk_pct": rr_info.get("risk_pct"),
        "reward_pct": rr_info.get("reward_pct"),
        "atr_multiple": atr_multiple,
        "position": pos,
    }

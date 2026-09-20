from __future__ import annotations

from typing import Any

import pandas as pd

from trade_journal import closed_trades


BASE_WEIGHTS = {
    "Setup Quality": 40.0,
    "Entry Timing": 35.0,
    "Risk Control": 25.0,
}


def _num_series(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def _clean_execution_mask(df: pd.DataFrame) -> pd.Series:
    """Keep trades that are most useful for calibrating the score itself."""
    if df.empty:
        return pd.Series(index=df.index, dtype=bool)

    rule = df.get("rule_break", pd.Series("", index=df.index)).astype(str).str.strip()
    followed = df.get("followed_plan", pd.Series("", index=df.index)).astype(str).str.strip().str.lower()

    no_rule_break = rule.isin({"", "なし", "none", "no"})
    followed_plan = followed.isin({"はい", "true", "1", "yes", "y", "プラン通り"})

    # If plan-adherence was not recorded, do not automatically discard the trade.
    followed_known = followed.ne("")
    return no_rule_break & (~followed_known | followed_plan)


def calibration_sample(
    journal: pd.DataFrame | None,
    *,
    prefer_clean: bool = True,
    min_clean: int = 10,
) -> tuple[pd.DataFrame, str]:
    trades = closed_trades(journal)
    if trades.empty:
        return trades, "NO_DATA"

    trades = trades.copy()
    trades["r_multiple"] = pd.to_numeric(trades["r_multiple"], errors="coerce")
    trades = trades.dropna(subset=["r_multiple"])
    if trades.empty:
        return trades, "NO_R"

    if prefer_clean:
        clean = trades[_clean_execution_mask(trades)].copy()
        if len(clean) >= min_clean:
            return clean, "CLEAN"

    return trades, "ALL"


def _spearman(x: pd.Series, y: pd.Series) -> tuple[float | None, int]:
    pair = pd.concat(
        [pd.to_numeric(x, errors="coerce"), pd.to_numeric(y, errors="coerce")],
        axis=1,
    ).dropna()
    if len(pair) < 5:
        return None, len(pair)
    if pair.iloc[:, 0].nunique() < 2 or pair.iloc[:, 1].nunique() < 2:
        return None, len(pair)
    value = pair.iloc[:, 0].corr(pair.iloc[:, 1], method="spearman")
    if pd.isna(value):
        return None, len(pair)
    return float(value), len(pair)


def _corr_label(value: float | None) -> str:
    if value is None:
        return "データ不足"
    a = abs(value)
    if a >= 0.35:
        strength = "強め"
    elif a >= 0.20:
        strength = "中程度"
    elif a >= 0.10:
        strength = "弱め"
    else:
        strength = "ほぼなし"
    direction = "プラス" if value > 0 else ("マイナス" if value < 0 else "なし")
    return f"{direction} / {strength}"


def component_analysis(journal: pd.DataFrame | None) -> pd.DataFrame:
    sample, sample_mode = calibration_sample(journal)
    columns = [
        "component", "base_weight", "n", "spearman_r",
        "signal", "sample_mode",
    ]
    if sample.empty:
        return pd.DataFrame(columns=columns)

    target = _num_series(sample, "r_multiple")
    items = [
        ("Setup Quality", "setup_score", 40.0),
        ("Entry Timing", "entry_score", 35.0),
        ("Risk Control", "risk_score", 25.0),
    ]

    rows = []
    for label, col, weight in items:
        corr, n = _spearman(_num_series(sample, col), target)
        rows.append({
            "component": label,
            "base_weight": weight,
            "n": n,
            "spearman_r": corr,
            "signal": _corr_label(corr),
            "sample_mode": sample_mode,
        })
    return pd.DataFrame(rows, columns=columns)


def factor_analysis(journal: pd.DataFrame | None) -> pd.DataFrame:
    """Analyze observable Pre-Trade factors against realized R.

    Correlation is descriptive only. It is not treated as proof of causality.
    """
    sample, sample_mode = calibration_sample(journal)
    columns = [
        "factor", "n", "spearman_r", "signal", "sample_mode",
    ]
    if sample.empty:
        return pd.DataFrame(columns=columns)

    target = _num_series(sample, "r_multiple")
    factor_cols = [
        ("Total Score", "total_score"),
        ("RS Proxy", "rs_proxy"),
        ("Volume Ratio", "volume_ratio"),
        ("Hunter Score", "hunter_score"),
        ("Planned RR", "planned_rr"),
        ("EMA20乖離", "ema20_gap_pct"),
        ("Entry Slippage", "entry_slippage_pct"),
    ]

    rows = []
    for label, col in factor_cols:
        corr, n = _spearman(_num_series(sample, col), target)
        rows.append({
            "factor": label,
            "n": n,
            "spearman_r": corr,
            "signal": _corr_label(corr),
            "sample_mode": sample_mode,
        })
    return pd.DataFrame(rows, columns=columns)


def _bounded_reweight(
    base: dict[str, float],
    correlations: dict[str, float | None],
    n: int,
) -> dict[str, float]:
    """Produce conservative experimental weights.

    Changes are shrunk toward the current weights and capped so a small sample
    cannot radically rewrite the production scoring model.
    """
    reliability = min(1.0, max(0.0, n / 40.0))
    raw = {}
    for key, weight in base.items():
        corr = correlations.get(key)
        effect = 0.0 if corr is None else max(-0.50, min(0.50, float(corr)))
        multiplier = 1.0 + 0.40 * effect * reliability
        raw[key] = weight * multiplier

    total = sum(raw.values()) or 100.0
    normalized = {k: v / total * 100.0 for k, v in raw.items()}

    # Guard rails around the current architecture.
    bounds = {
        "Setup Quality": (34.0, 46.0),
        "Entry Timing": (29.0, 41.0),
        "Risk Control": (20.0, 30.0),
    }
    clipped = {
        k: min(bounds[k][1], max(bounds[k][0], normalized[k]))
        for k in normalized
    }

    # Re-normalize while preserving the guard rails closely.
    total2 = sum(clipped.values()) or 100.0
    scaled = {k: v / total2 * 100.0 for k, v in clipped.items()}
    return {k: round(v, 1) for k, v in scaled.items()}


def build_calibration_report(
    journal: pd.DataFrame | None,
    *,
    min_trades: int = 20,
    high_confidence_trades: int = 40,
) -> dict[str, Any]:
    sample, sample_mode = calibration_sample(journal)
    n = len(sample)

    components = component_analysis(journal)
    factors = factor_analysis(journal)

    if n == 0:
        status = "⚪ NO DATA"
        confidence = "なし"
        message = "CLOSEDトレードのR倍データがまだありません。"
    elif n < min_trades:
        status = "🟡 LEARNING"
        confidence = "低"
        message = (
            f"有効サンプル {n}件。{min_trades}件までは重み変更案を出さず、"
            "現行40/35/25を維持します。"
        )
    elif n < high_confidence_trades:
        status = "🟠 EXPERIMENTAL"
        confidence = "中"
        message = (
            f"有効サンプル {n}件。参考用の実験重みを表示しますが、"
            "自動適用はしません。"
        )
    else:
        status = "🟢 REVIEW READY"
        confidence = "高"
        message = (
            f"有効サンプル {n}件。重み見直しを検討できる件数ですが、"
            "適用は人の確認後に行います。"
        )

    correlations = {}
    if not components.empty:
        for _, row in components.iterrows():
            correlations[str(row["component"])] = (
                None if pd.isna(row["spearman_r"]) else float(row["spearman_r"])
            )

    if n >= min_trades:
        proposed = _bounded_reweight(BASE_WEIGHTS, correlations, n)
    else:
        proposed = dict(BASE_WEIGHTS)

    weight_rows = []
    for key, base_weight in BASE_WEIGHTS.items():
        proposed_weight = proposed[key]
        weight_rows.append({
            "component": key,
            "current_weight": base_weight,
            "experimental_weight": proposed_weight,
            "change": round(proposed_weight - base_weight, 1),
            "spearman_r": correlations.get(key),
            "eligible": n >= min_trades,
        })

    clean_count = 0
    all_trades = closed_trades(journal)
    if not all_trades.empty:
        all_trades = all_trades.copy()
        all_trades["r_multiple"] = pd.to_numeric(all_trades["r_multiple"], errors="coerce")
        all_trades = all_trades.dropna(subset=["r_multiple"])
        clean_count = int(_clean_execution_mask(all_trades).sum())

    return {
        "status": status,
        "confidence": confidence,
        "message": message,
        "sample_size": n,
        "sample_mode": sample_mode,
        "all_closed_with_r": int(len(all_trades)),
        "clean_closed_with_r": clean_count,
        "min_trades": min_trades,
        "high_confidence_trades": high_confidence_trades,
        "components": components,
        "factors": factors,
        "weights": pd.DataFrame(weight_rows),
        "proposed_weights": proposed,
    }

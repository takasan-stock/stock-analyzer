from __future__ import annotations

from typing import Any

import pandas as pd

from score_calibration import BASE_WEIGHTS, _bounded_reweight, _clean_execution_mask, _spearman
from trade_journal import closed_trades


COMPONENT_MAX = {
    "Setup Quality": 40.0,
    "Entry Timing": 35.0,
    "Risk Control": 25.0,
}

COMPONENT_COLS = {
    "Setup Quality": "setup_score",
    "Entry Timing": "entry_score",
    "Risk Control": "risk_score",
}


def _usable_trades(journal: pd.DataFrame | None, prefer_clean: bool = True) -> tuple[pd.DataFrame, str]:
    trades = closed_trades(journal)
    if trades.empty:
        return trades, "NO_DATA"

    trades = trades.copy()
    trades["r_multiple"] = pd.to_numeric(trades["r_multiple"], errors="coerce")
    trades = trades.dropna(subset=["r_multiple"])

    for col in COMPONENT_COLS.values():
        trades[col] = pd.to_numeric(trades[col], errors="coerce")
    trades = trades.dropna(subset=list(COMPONENT_COLS.values()))

    if trades.empty:
        return trades, "NO_SCORE"

    if prefer_clean:
        clean = trades[_clean_execution_mask(trades)].copy()
        if len(clean) >= 12:
            trades = clean
            mode = "CLEAN"
        else:
            mode = "ALL"
    else:
        mode = "ALL"

    # Sort chronologically using actual entry first, then confirmed/updated date.
    entry_dt = pd.to_datetime(trades.get("actual_entry_date"), errors="coerce")
    confirmed_dt = pd.to_datetime(trades.get("plan_confirmed_at"), errors="coerce")
    updated_dt = pd.to_datetime(trades.get("updated_at"), errors="coerce")
    trades["_sort_dt"] = entry_dt.fillna(confirmed_dt).fillna(updated_dt)
    trades["_sort_dt"] = trades["_sort_dt"].fillna(pd.Timestamp("1970-01-01"))
    trades = trades.sort_values(["_sort_dt", "plan_id"], kind="stable").reset_index(drop=True)
    return trades, mode


def rescore_components(df: pd.DataFrame, weights: dict[str, float]) -> pd.Series:
    if df.empty:
        return pd.Series(index=df.index, dtype=float)

    score = pd.Series(0.0, index=df.index, dtype=float)
    for name, col in COMPONENT_COLS.items():
        values = pd.to_numeric(df[col], errors="coerce")
        normalized = (values / COMPONENT_MAX[name]).clip(lower=0.0, upper=1.0)
        score = score + normalized * float(weights[name])
    return score


def _derive_weights_from_train(train: pd.DataFrame) -> dict[str, float]:
    target = pd.to_numeric(train["r_multiple"], errors="coerce")
    correlations: dict[str, float | None] = {}
    for name, col in COMPONENT_COLS.items():
        corr, _ = _spearman(pd.to_numeric(train[col], errors="coerce"), target)
        correlations[name] = corr
    return _bounded_reweight(BASE_WEIGHTS, correlations, len(train))


def _model_metrics(df: pd.DataFrame, score_col: str) -> dict[str, Any]:
    if df.empty:
        return {
            "n": 0,
            "spearman_r": None,
            "top_n": 0,
            "top_avg_r": None,
            "top_win_rate": None,
            "bottom_avg_r": None,
            "top_bottom_spread": None,
            "overall_avg_r": None,
        }

    work = df[[score_col, "r_multiple"]].copy()
    work[score_col] = pd.to_numeric(work[score_col], errors="coerce")
    work["r_multiple"] = pd.to_numeric(work["r_multiple"], errors="coerce")
    work = work.dropna()
    if work.empty:
        return {
            "n": 0,
            "spearman_r": None,
            "top_n": 0,
            "top_avg_r": None,
            "top_win_rate": None,
            "bottom_avg_r": None,
            "top_bottom_spread": None,
            "overall_avg_r": None,
        }

    corr, _ = _spearman(work[score_col], work["r_multiple"])
    n = len(work)
    q = max(1, int((n * 0.25) + 0.999999))
    ranked = work.sort_values(score_col, ascending=False, kind="stable")
    top = ranked.head(q)
    bottom = ranked.tail(q)

    top_avg = float(top["r_multiple"].mean())
    bottom_avg = float(bottom["r_multiple"].mean())
    return {
        "n": n,
        "spearman_r": corr,
        "top_n": len(top),
        "top_avg_r": top_avg,
        "top_win_rate": float((top["r_multiple"] > 0).mean() * 100.0),
        "bottom_avg_r": bottom_avg,
        "top_bottom_spread": top_avg - bottom_avg,
        "overall_avg_r": float(work["r_multiple"].mean()),
    }


def _comparison_rows(current: dict[str, Any], experimental: dict[str, Any]) -> pd.DataFrame:
    metrics = [
        ("ScoreとRの順位相関", "spearman_r"),
        ("上位25% 平均R", "top_avg_r"),
        ("上位25% 勝率%", "top_win_rate"),
        ("下位25% 平均R", "bottom_avg_r"),
        ("上位-下位 R差", "top_bottom_spread"),
    ]
    rows = []
    for label, key in metrics:
        cur = current.get(key)
        exp = experimental.get(key)
        delta = None if cur is None or exp is None else float(exp) - float(cur)
        rows.append({
            "metric": label,
            "current": cur,
            "experimental": exp,
            "delta": delta,
        })
    return pd.DataFrame(rows)


def build_calibration_backtest(
    journal: pd.DataFrame | None,
    *,
    train_fraction: float = 0.60,
    min_total: int = 20,
    min_validation: int = 6,
    prefer_clean: bool = True,
) -> dict[str, Any]:
    trades, sample_mode = _usable_trades(journal, prefer_clean=prefer_clean)
    total = len(trades)

    if total < min_total:
        return {
            "status": "🟡 DATA BUILDING",
            "message": f"有効トレード {total}件。バックテスト開始目安は{min_total}件です。",
            "sample_mode": sample_mode,
            "total": total,
            "train_n": 0,
            "validation_n": 0,
            "weights": dict(BASE_WEIGHTS),
            "current_metrics": {},
            "experimental_metrics": {},
            "comparison": pd.DataFrame(),
            "validation": pd.DataFrame(),
        }

    fraction = min(0.80, max(0.50, float(train_fraction)))
    split = int(total * fraction)
    split = max(10, min(split, total - min_validation))

    train = trades.iloc[:split].copy()
    validation = trades.iloc[split:].copy()

    if len(validation) < min_validation:
        return {
            "status": "🟡 DATA BUILDING",
            "message": f"検証期間が{len(validation)}件しかありません。最低{min_validation}件必要です。",
            "sample_mode": sample_mode,
            "total": total,
            "train_n": len(train),
            "validation_n": len(validation),
            "weights": dict(BASE_WEIGHTS),
            "current_metrics": {},
            "experimental_metrics": {},
            "comparison": pd.DataFrame(),
            "validation": pd.DataFrame(),
        }

    experimental_weights = _derive_weights_from_train(train)

    validation["current_score_bt"] = rescore_components(validation, BASE_WEIGHTS)
    validation["experimental_score_bt"] = rescore_components(validation, experimental_weights)

    current_metrics = _model_metrics(validation, "current_score_bt")
    experimental_metrics = _model_metrics(validation, "experimental_score_bt")
    comparison = _comparison_rows(current_metrics, experimental_metrics)

    # Descriptive validation status. This does not auto-approve production changes.
    comparable = comparison.dropna(subset=["delta"])
    positive_count = int((comparable["delta"] > 0).sum()) if not comparable.empty else 0
    negative_count = int((comparable["delta"] < 0).sum()) if not comparable.empty else 0

    if len(validation) < 10:
        status = "🟠 SMALL VALIDATION"
        message = (
            f"学習{len(train)}件 / 検証{len(validation)}件。"
            "検証件数が少ないため、結果は参考扱いです。"
        )
    elif positive_count >= 3 and positive_count > negative_count:
        status = "🟢 EXPERIMENT SHOWS PROMISE"
        message = (
            f"学習{len(train)}件で作った実験ウェイトを、その後の検証{len(validation)}件に適用。"
            "複数の選別指標で改善が見られましたが、本番適用はまだ行いません。"
        )
    elif negative_count >= 3 and negative_count > positive_count:
        status = "🔴 EXPERIMENT NOT CONFIRMED"
        message = (
            f"学習{len(train)}件で作った実験ウェイトは、検証{len(validation)}件では"
            "複数指標で改善を確認できませんでした。現行ウェイトを維持します。"
        )
    else:
        status = "🟡 MIXED RESULT"
        message = (
            f"学習{len(train)}件 / 検証{len(validation)}件。"
            "改善と悪化が混在しており、重み変更を支持するほど明確ではありません。"
        )

    visible_cols = [
        "actual_entry_date", "ticker", "name", "r_multiple",
        "setup_score", "entry_score", "risk_score",
        "current_score_bt", "experimental_score_bt",
    ]
    visible_cols = [c for c in visible_cols if c in validation.columns]

    return {
        "status": status,
        "message": message,
        "sample_mode": sample_mode,
        "total": total,
        "train_n": len(train),
        "validation_n": len(validation),
        "train_start": train["_sort_dt"].min(),
        "train_end": train["_sort_dt"].max(),
        "validation_start": validation["_sort_dt"].min(),
        "validation_end": validation["_sort_dt"].max(),
        "weights": experimental_weights,
        "current_metrics": current_metrics,
        "experimental_metrics": experimental_metrics,
        "comparison": comparison,
        "validation": validation[visible_cols].copy(),
    }

from __future__ import annotations

from typing import Any

import pandas as pd

from calibration_backtest import (
    _derive_weights_from_train,
    _model_metrics,
    _usable_trades,
    rescore_components,
)
from score_calibration import BASE_WEIGHTS


DECISION_METRICS = [
    "spearman_r",
    "top_avg_r",
    "top_win_rate",
    "top_bottom_spread",
]


def _metric_delta(current: dict[str, Any], experimental: dict[str, Any], key: str) -> float | None:
    cur = current.get(key)
    exp = experimental.get(key)
    if cur is None or exp is None:
        return None
    return float(exp) - float(cur)


def _fold_verdict(current: dict[str, Any], experimental: dict[str, Any]) -> tuple[str, int, int]:
    deltas = [_metric_delta(current, experimental, key) for key in DECISION_METRICS]
    deltas = [d for d in deltas if d is not None]
    positive = sum(1 for d in deltas if d > 0)
    negative = sum(1 for d in deltas if d < 0)

    if not deltas:
        return "NO SIGNAL", 0, 0
    if positive >= 3 and positive > negative:
        return "PROMISE", positive, negative
    if negative >= 3 and negative > positive:
        return "NOT CONFIRMED", positive, negative
    return "MIXED", positive, negative


def _aggregate_fold_metrics(folds: list[dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for fold in folds:
        cur = fold["current_metrics"]
        exp = fold["experimental_metrics"]
        row = {
            "fold": fold["fold"],
            "train_n": fold["train_n"],
            "validation_n": fold["validation_n"],
            "train_end": fold["train_end"],
            "validation_start": fold["validation_start"],
            "validation_end": fold["validation_end"],
            "verdict": fold["verdict"],
            "positive_metrics": fold["positive_metrics"],
            "negative_metrics": fold["negative_metrics"],
            "current_corr": cur.get("spearman_r"),
            "experimental_corr": exp.get("spearman_r"),
            "delta_corr": _metric_delta(cur, exp, "spearman_r"),
            "current_top_avg_r": cur.get("top_avg_r"),
            "experimental_top_avg_r": exp.get("top_avg_r"),
            "delta_top_avg_r": _metric_delta(cur, exp, "top_avg_r"),
            "current_top_win_rate": cur.get("top_win_rate"),
            "experimental_top_win_rate": exp.get("top_win_rate"),
            "delta_top_win_rate": _metric_delta(cur, exp, "top_win_rate"),
            "current_spread": cur.get("top_bottom_spread"),
            "experimental_spread": exp.get("top_bottom_spread"),
            "delta_spread": _metric_delta(cur, exp, "top_bottom_spread"),
            "setup_weight": fold["weights"]["Setup Quality"],
            "entry_weight": fold["weights"]["Entry Timing"],
            "risk_weight": fold["weights"]["Risk Control"],
        }
        rows.append(row)
    return pd.DataFrame(rows)


def _weight_stability(fold_table: pd.DataFrame) -> pd.DataFrame:
    columns = ["component", "mean_weight", "min_weight", "max_weight", "std_weight", "range"]
    if fold_table.empty:
        return pd.DataFrame(columns=columns)

    mapping = {
        "Setup Quality": "setup_weight",
        "Entry Timing": "entry_weight",
        "Risk Control": "risk_weight",
    }
    rows = []
    for component, col in mapping.items():
        values = pd.to_numeric(fold_table[col], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "component": component,
            "mean_weight": float(values.mean()),
            "min_weight": float(values.min()),
            "max_weight": float(values.max()),
            "std_weight": float(values.std(ddof=0)),
            "range": float(values.max() - values.min()),
        })
    return pd.DataFrame(rows, columns=columns)


def build_walk_forward_report(
    journal: pd.DataFrame | None,
    *,
    initial_train: int = 20,
    validation_window: int = 10,
    step: int | None = None,
    prefer_clean: bool = True,
) -> dict[str, Any]:
    trades, sample_mode = _usable_trades(journal, prefer_clean=prefer_clean)
    total = len(trades)

    initial_train = max(10, int(initial_train))
    validation_window = max(5, int(validation_window))
    step = validation_window if step is None else max(1, int(step))
    min_required = initial_train + validation_window

    empty = {
        "status": "🟡 DATA BUILDING",
        "message": (
            f"有効トレード {total}件。Walk-Forward開始には最低{min_required}件必要です。"
        ),
        "sample_mode": sample_mode,
        "total": total,
        "initial_train": initial_train,
        "validation_window": validation_window,
        "step": step,
        "folds": [],
        "fold_table": pd.DataFrame(),
        "weight_stability": pd.DataFrame(),
        "aggregate_current": {},
        "aggregate_experimental": {},
        "aggregate_deltas": {},
        "validation_total": 0,
        "promise_folds": 0,
        "mixed_folds": 0,
        "not_confirmed_folds": 0,
        "promise_rate": None,
        "oos_validation": pd.DataFrame(),
    }
    if total < min_required:
        return empty

    folds: list[dict[str, Any]] = []
    oos_parts: list[pd.DataFrame] = []

    train_end = initial_train
    fold_no = 1

    while train_end + validation_window <= total:
        train = trades.iloc[:train_end].copy()
        validation = trades.iloc[train_end:train_end + validation_window].copy()

        weights = _derive_weights_from_train(train)
        validation["current_score_wf"] = rescore_components(validation, BASE_WEIGHTS)
        validation["experimental_score_wf"] = rescore_components(validation, weights)
        validation["wf_fold"] = fold_no

        current_metrics = _model_metrics(validation, "current_score_wf")
        experimental_metrics = _model_metrics(validation, "experimental_score_wf")
        verdict, positive, negative = _fold_verdict(current_metrics, experimental_metrics)

        fold = {
            "fold": fold_no,
            "train_n": len(train),
            "validation_n": len(validation),
            "train_start": train["_sort_dt"].min(),
            "train_end": train["_sort_dt"].max(),
            "validation_start": validation["_sort_dt"].min(),
            "validation_end": validation["_sort_dt"].max(),
            "weights": weights,
            "current_metrics": current_metrics,
            "experimental_metrics": experimental_metrics,
            "verdict": verdict,
            "positive_metrics": positive,
            "negative_metrics": negative,
        }
        folds.append(fold)

        cols = [
            "wf_fold", "actual_entry_date", "ticker", "name", "r_multiple",
            "setup_score", "entry_score", "risk_score",
            "current_score_wf", "experimental_score_wf",
        ]
        cols = [c for c in cols if c in validation.columns]
        oos_parts.append(validation[cols].copy())

        fold_no += 1
        train_end += step

    if not folds:
        return empty

    oos = pd.concat(oos_parts, ignore_index=True) if oos_parts else pd.DataFrame()
    aggregate_current = _model_metrics(oos, "current_score_wf")
    aggregate_experimental = _model_metrics(oos, "experimental_score_wf")

    aggregate_deltas = {
        key: _metric_delta(aggregate_current, aggregate_experimental, key)
        for key in DECISION_METRICS
    }

    fold_table = _aggregate_fold_metrics(folds)
    stability = _weight_stability(fold_table)

    promise_folds = sum(1 for f in folds if f["verdict"] == "PROMISE")
    mixed_folds = sum(1 for f in folds if f["verdict"] == "MIXED")
    not_confirmed_folds = sum(1 for f in folds if f["verdict"] == "NOT CONFIRMED")
    evaluated = promise_folds + mixed_folds + not_confirmed_folds
    promise_rate = promise_folds / evaluated * 100.0 if evaluated else None

    agg_deltas = [d for d in aggregate_deltas.values() if d is not None]
    agg_positive = sum(1 for d in agg_deltas if d > 0)
    agg_negative = sum(1 for d in agg_deltas if d < 0)

    max_weight_std = None
    if not stability.empty:
        max_weight_std = float(pd.to_numeric(stability["std_weight"], errors="coerce").max())

    validation_total = len(oos)

    if len(folds) < 2 or validation_total < 15:
        status = "🟠 EARLY WALK-FORWARD"
        message = (
            f"{len(folds)}区間 / 検証延べ{validation_total}件。"
            "まだ期間横断の安定性を判断するには少なめです。"
        )
    elif (
        promise_rate is not None
        and promise_rate >= 60.0
        and agg_positive >= 3
        and agg_positive > agg_negative
        and (max_weight_std is None or max_weight_std <= 3.0)
    ):
        status = "🟢 STABLE PROMISE"
        message = (
            f"{len(folds)}区間のうち{promise_folds}区間で改善傾向。"
            "集約Out-of-Sampleでも複数指標が改善し、重みの変動も比較的安定しています。"
            "ただし本番ウェイトは自動変更しません。"
        )
    elif (
        not_confirmed_folds >= max(2, promise_folds + 1)
        and agg_negative >= 3
        and agg_negative > agg_positive
    ):
        status = "🔴 NOT ROBUST"
        message = (
            f"{len(folds)}区間で改善が安定せず、集約Out-of-Sampleでも弱い結果です。"
            "現行ウェイト維持が妥当な検証結果です。"
        )
    else:
        status = "🟡 REGIME DEPENDENT / MIXED"
        message = (
            f"{len(folds)}区間で結果が混在しています。"
            "相場局面によって効く重みが変わる可能性があり、固定ウェイト変更を支持するほど安定していません。"
        )

    return {
        "status": status,
        "message": message,
        "sample_mode": sample_mode,
        "total": total,
        "initial_train": initial_train,
        "validation_window": validation_window,
        "step": step,
        "folds": folds,
        "fold_table": fold_table,
        "weight_stability": stability,
        "aggregate_current": aggregate_current,
        "aggregate_experimental": aggregate_experimental,
        "aggregate_deltas": aggregate_deltas,
        "validation_total": validation_total,
        "promise_folds": promise_folds,
        "mixed_folds": mixed_folds,
        "not_confirmed_folds": not_confirmed_folds,
        "promise_rate": promise_rate,
        "oos_validation": oos,
    }

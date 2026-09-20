from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from short_cover import (
    apply_optimizer_condition,
    append_priority_alert_history,
    build_operational_health,
    build_price_feature_snapshots,
    build_priority_alerts,
    build_promotion_table,
    build_short_metrics,
    candidate_tickers,
    get_active_condition_version,
    load_jpx_events,
    normalize_alert_history,
    normalize_condition_versions,
    score_short_cover,
    summarize_alert_history,
    update_alert_history_outcomes,
)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
CONDITION_FILE = DATA_DIR / "short_cover_condition_versions.csv"
HISTORY_FILE = DATA_DIR / "short_cover_alert_history.csv"
STATUS_FILE = DATA_DIR / "short_cover_daily_status.json"


def load_conditions() -> pd.DataFrame:
    if not CONDITION_FILE.exists():
        return normalize_condition_versions(None)
    return normalize_condition_versions(
        pd.read_csv(CONDITION_FILE, encoding="utf-8-sig")
    )


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return normalize_alert_history(None)
    return normalize_alert_history(
        pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    )


def save_history(history: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    export = normalize_alert_history(history).copy()
    for col in ["alert_date", "entry_date", "last_updated"]:
        export[col] = pd.to_datetime(export[col], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    export.to_csv(HISTORY_FILE, index=False, encoding="utf-8-sig")


def save_status(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def main() -> int:
    versions = load_conditions()
    active = get_active_condition_version(versions)
    history = load_history()

    if active is None:
        save_status({
            "run_at": pd.Timestamp.now().isoformat(),
            "status": "NO_ACTIVE_CONDITION",
            "message": "ACTIVE条件がありません。",
        })
        print("Short Cover daily: ACTIVE条件なし")
        return 0

    # 既存アラートの追跡成績は、新規シグナル可否とは独立して更新する。
    history = update_alert_history_outcomes(history)

    jpx = load_jpx_events(archive_pages=2, max_files=70)
    events = jpx.events.copy()
    if events.empty:
        save_history(history)
        save_status({
            "run_at": pd.Timestamp.now().isoformat(),
            "status": "NO_JPX_DATA",
            "active_version": active.get("version_id", ""),
            "errors": jpx.errors[:10],
        })
        print("Short Cover daily: JPXデータなし。履歴追跡のみ更新。")
        return 0

    short_metrics = build_short_metrics(events)
    targets = candidate_tickers(short_metrics, limit=60)
    if not targets:
        save_history(history)
        save_status({
            "run_at": pd.Timestamp.now().isoformat(),
            "status": "NO_CANDIDATES",
            "active_version": active.get("version_id", ""),
        })
        print("Short Cover daily: 候補なし。")
        return 0

    prices, prev_prices = build_price_feature_snapshots(targets)
    target_metrics = short_metrics[short_metrics["ticker"].isin(targets)].copy()
    scored = score_short_cover(target_metrics, prices)
    prev_scored = score_short_cover(target_metrics, prev_prices)
    promotions = build_promotion_table(scored, prev_scored)

    if scored.empty:
        save_history(history)
        save_status({
            "run_at": pd.Timestamp.now().isoformat(),
            "status": "NO_SCORED_ROWS",
            "active_version": active.get("version_id", ""),
        })
        print("Short Cover daily: スコア対象なし。")
        return 0

    scored = apply_optimizer_condition(scored, active)
    if not promotions.empty:
        promotions = promotions.merge(
            scored[[
                "ticker", "optimizer_match", "optimizer_label",
                "match_strength", "condition_text",
            ]],
            on="ticker",
            how="left",
        )

    operational = build_operational_health(
        events,
        prices,
        active_condition=active,
        files_loaded=jpx.files_loaded,
        source_mode="JPX",
    )

    priority = build_priority_alerts(
        scored,
        promotions=promotions,
        limit=5,
        validation_ready=True,
    )

    added = 0
    if operational["status"] == "🟢 READY":
        history, added = append_priority_alert_history(
            history,
            priority,
            tracking_mode="ACTIVE",
            condition_version=str(active.get("version_id", "") or ""),
        )

    save_history(history)
    summary = summarize_alert_history(history)
    save_status({
        "run_at": pd.Timestamp.now().isoformat(),
        "status": operational["status"],
        "active_version": active.get("version_id", ""),
        "condition_text": operational.get("condition_text", ""),
        "jpx_date": operational.get("jpx_date"),
        "jpx_lag": operational.get("jpx_lag"),
        "market_date": operational.get("market_date"),
        "market_lag": operational.get("market_lag"),
        "warnings": operational.get("warnings", []),
        "candidates": int(len(scored)),
        "priority_count": int(len(priority)),
        "new_alerts": int(added),
        "official_alerts": int(summary.get("alerts", 0)),
        "tracked_alerts": int(summary.get("tracked", 0)),
    })

    print(
        "Short Cover daily:",
        operational["status"],
        f"candidates={len(scored)}",
        f"priority={len(priority)}",
        f"new_alerts={added}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

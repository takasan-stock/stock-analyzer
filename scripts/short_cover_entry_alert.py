from __future__ import annotations

import argparse
import json
import os
import smtplib
import sys
from email.message import EmailMessage
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import yfinance as yf

from short_cover import (
    build_entry_followup_snapshot,
    build_entry_hunter_snapshot,
    normalize_alert_history,
    select_entry_hunter_candidates,
)
from entry_hunter_sources import combine_entry_candidates, select_me_entry_candidates
from entry_opportunity import build_entry_opportunity
from daily_command_center import (
    build_daily_command_center,
    normalize_command_center_history,
    update_command_center_history,
)
from entry_source_performance import (
    build_ready_performance,
    normalize_candidate_history,
    normalize_performance,
    summarize_source_performance,
    summarize_signal_performance,
    summarize_trait_performance,
    update_candidate_history,
)

DATA_DIR = ROOT / "data"
HISTORY_FILE = DATA_DIR / "short_cover_alert_history.csv"
NOTIFICATION_FILE = DATA_DIR / "short_cover_entry_notifications.csv"
STATUS_FILE = DATA_DIR / "short_cover_entry_status.json"
ME_SCREENER_FILE = DATA_DIR / "multiple_expansion" / "me_screener_latest.csv"
ENTRY_CANDIDATE_HISTORY_FILE = DATA_DIR / "entry_hunter_candidate_history.csv"
ENTRY_SOURCE_PERFORMANCE_FILE = DATA_DIR / "entry_hunter_source_performance.csv"
ENTRY_SOURCE_SUMMARY_FILE = DATA_DIR / "entry_hunter_source_summary.csv"
ENTRY_SIGNAL_SUMMARY_FILE = DATA_DIR / "entry_hunter_signal_summary.csv"
ENTRY_TRAIT_SUMMARY_FILE = DATA_DIR / "entry_hunter_trait_summary.csv"
ME_UNIVERSE_FILE = DATA_DIR / "multiple_expansion" / "me_universe_snapshot.csv"
COMMAND_CENTER_LATEST_FILE = DATA_DIR / "daily_command_center_latest.csv"
COMMAND_CENTER_HISTORY_FILE = DATA_DIR / "daily_command_center_history.csv"
AFTER_CLOSE_FEEDBACK_FILE = DATA_DIR / "after_close_feedback_summary.csv"

NOTIFICATION_COLUMNS = [
    "market_date", "ticker", "name", "alert_date", "condition_version",
    "entry_status", "entry_score", "gap_pct", "relvol15",
    "current_price", "vwap", "reason", "risk",
    "first_detected_at", "email_sent", "email_sent_at", "email_error",
    "source", "source_detail", "signal_key",
    "trait_market", "trait_size", "trait_vol",
]


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def load_history() -> pd.DataFrame:
    if not HISTORY_FILE.exists():
        return normalize_alert_history(None)
    return normalize_alert_history(
        pd.read_csv(HISTORY_FILE, encoding="utf-8-sig")
    )


def load_me_universe() -> pd.DataFrame:
    if not ME_UNIVERSE_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(ME_UNIVERSE_FILE, dtype={"ticker": str})
    except Exception:
        return pd.DataFrame()


def load_me_screener() -> pd.DataFrame:
    if not ME_SCREENER_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(ME_SCREENER_FILE, dtype={"ticker": str})
    except Exception:
        return pd.DataFrame()


def load_command_center_history() -> pd.DataFrame:
    if not COMMAND_CENTER_HISTORY_FILE.exists():
        return normalize_command_center_history(None)
    try:
        return normalize_command_center_history(
            pd.read_csv(COMMAND_CENTER_HISTORY_FILE, dtype={"ticker": str})
        )
    except Exception:
        return normalize_command_center_history(None)


def save_command_center_snapshot(
    current: pd.DataFrame,
    history: pd.DataFrame,
) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    current.to_csv(COMMAND_CENTER_LATEST_FILE, index=False, encoding="utf-8-sig")
    history.to_csv(COMMAND_CENTER_HISTORY_FILE, index=False, encoding="utf-8-sig")



def load_after_close_feedback() -> pd.DataFrame:
    if not AFTER_CLOSE_FEEDBACK_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(AFTER_CLOSE_FEEDBACK_FILE)
    except Exception:
        return pd.DataFrame()


def load_notifications() -> pd.DataFrame:
    if not NOTIFICATION_FILE.exists():
        return pd.DataFrame(columns=NOTIFICATION_COLUMNS)

    try:
        out = pd.read_csv(NOTIFICATION_FILE, encoding="utf-8-sig")
    except Exception:
        return pd.DataFrame(columns=NOTIFICATION_COLUMNS)

    for col in NOTIFICATION_COLUMNS:
        if col not in out.columns:
            out[col] = None

    for col in ["market_date", "alert_date", "first_detected_at", "email_sent_at"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in [
        "entry_score", "gap_pct", "relvol15", "current_price", "vwap"
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ticker"] = out["ticker"].fillna("").astype(str)
    out["email_sent"] = out["email_sent"].map(_to_bool)
    return out[NOTIFICATION_COLUMNS].copy()


def save_notifications(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    for col in ["market_date", "alert_date", "first_detected_at", "email_sent_at"]:
        out[col] = pd.to_datetime(out[col], errors="coerce").dt.strftime(
            "%Y-%m-%d %H:%M:%S"
        )
    out.to_csv(NOTIFICATION_FILE, index=False, encoding="utf-8-sig")


def load_candidate_history() -> pd.DataFrame:
    if not ENTRY_CANDIDATE_HISTORY_FILE.exists():
        return normalize_candidate_history(None)
    try:
        return normalize_candidate_history(
            pd.read_csv(ENTRY_CANDIDATE_HISTORY_FILE, encoding="utf-8-sig")
        )
    except Exception:
        return normalize_candidate_history(None)


def load_source_performance() -> pd.DataFrame:
    if not ENTRY_SOURCE_PERFORMANCE_FILE.exists():
        return normalize_performance(None)
    try:
        return normalize_performance(
            pd.read_csv(ENTRY_SOURCE_PERFORMANCE_FILE, encoding="utf-8-sig")
        )
    except Exception:
        return normalize_performance(None)


def save_candidate_history(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = normalize_candidate_history(df)
    out.to_csv(ENTRY_CANDIDATE_HISTORY_FILE, index=False, encoding="utf-8-sig")


def save_source_performance(df: pd.DataFrame) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    out = normalize_performance(df)
    out.to_csv(ENTRY_SOURCE_PERFORMANCE_FILE, index=False, encoding="utf-8-sig")


def save_status(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def load_prices(ticker: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    symbol = ticker if "." in ticker else f"{ticker}.T"
    daily = yf.download(
        symbol,
        period="1mo",
        interval="1d",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    intraday = yf.download(
        symbol,
        period="10d",
        interval="5m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    for frame in (daily, intraday):
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = frame.columns.get_level_values(0)
    if not daily.empty and "Close" in daily.columns:
        daily = daily.dropna(subset=["Close"])
    if not intraday.empty and "Close" in intraday.columns:
        intraday = intraday.dropna(subset=["Close"])
    return daily, intraday


def email_config() -> dict:
    return {
        "to": os.getenv("SHORT_COVER_EMAIL_TO", "").strip(),
        "user": os.getenv("SHORT_COVER_EMAIL_USER", "").strip(),
        "password": os.getenv("SHORT_COVER_EMAIL_APP_PASSWORD", "").strip(),
        "host": (os.getenv("SHORT_COVER_SMTP_HOST", "").strip() or "smtp.gmail.com"),
        "port": int(os.getenv("SHORT_COVER_SMTP_PORT", "").strip() or "465"),
    }


def send_entry_email(row: dict, cfg: dict) -> tuple[bool, str]:
    if not (cfg["to"] and cfg["user"] and cfg["password"]):
        return False, "EMAIL_SECRETS_NOT_CONFIGURED"

    msg = EmailMessage()
    status = str(row.get("entry_status", "🟢 ENTRY READY"))
    subject_prefix = {
        "🟢 ENTRY READY": "🚨 ENTRY READY",
        "🟢 ENTRY CONFIRMED": "✅ ENTRY CONFIRMED",
        "🟡 WEAKENING": "⚠️ WEAKENING",
        "🔴 EXIT WATCH": "🛑 EXIT WATCH",
    }.get(status, "📌 ENTRY UPDATE")
    msg["Subject"] = f"{subject_prefix}: {row['ticker']} {row['name']}"
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]

    gap = "—" if pd.isna(row.get("gap_pct")) else f"{float(row['gap_pct']):+.1f}%"
    relvol = (
        "—" if pd.isna(row.get("relvol15"))
        else f"{float(row['relvol15']):.1f}x"
    )
    price = (
        "—" if pd.isna(row.get("current_price"))
        else f"{float(row['current_price']):,.1f}"
    )
    vwap = (
        "—" if pd.isna(row.get("vwap"))
        else f"{float(row['vwap']):,.1f}"
    )

    body = f"""Entry Hunter が監視条件成立を検知しました。

銘柄: {row['ticker']} {row['name']}
Entry Score: {float(row['entry_score']):.0f}
Gap: {gap}
現在値: {price}
VWAP: {vwap}
15分相対出来高: {relvol}
前日アラート日: {pd.Timestamp(row['alert_date']).strftime('%Y-%m-%d')}
条件Version: {row.get('condition_version', '')}
監視ソース: {row.get('source', '')}
ソース理由: {row.get('source_detail', '')}

成立条件:
{row.get('reason', '')}

注意:
{row.get('risk', '') or '特記事項なし'}

※これは売買推奨ではなく、ME Hunter / Short Coverの候補を寄り付き後に確認するための監視通知です。
"""

    msg.set_content(body)

    try:
        with smtplib.SMTP_SSL(cfg["host"], cfg["port"], timeout=20) as smtp:
            smtp.login(cfg["user"], cfg["password"])
            smtp.send_message(msg)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:500]


def send_test_email(cfg: dict) -> tuple[bool, str]:
    row = {
        "ticker": "TEST",
        "name": "Short Cover Entry Hunter",
        "entry_score": 88,
        "gap_pct": 1.8,
        "relvol15": 1.7,
        "current_price": 1234.5,
        "vwap": 1228.0,
        "alert_date": pd.Timestamp.now().normalize(),
        "condition_version": "TEST",
        "reason": "VWAP上 / 15分高値突破 / 出来高継続",
        "risk": "テストメールです",
    }
    row["entry_status"] = "🟢 ENTRY READY"
    return send_entry_email(row, cfg)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--test-email",
        action="store_true",
        help="Send one test email and exit without touching Entry Hunter history.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    now = pd.Timestamp.now(tz="Asia/Tokyo")

    if args.test_email:
        cfg = email_config()
        ok, error = send_test_email(cfg)

        # Test mode must not touch alert/notification history, but it should
        # update the UI-facing status so a successful test is reflected there.
        status = {}
        if STATUS_FILE.exists():
            try:
                status = json.loads(STATUS_FILE.read_text(encoding="utf-8"))
            except Exception:
                status = {}

        status.update({
            "run_at": now.isoformat(),
            "email_configured": bool(cfg["to"] and cfg["user"] and cfg["password"]),
            "last_test_email_at": now.isoformat() if ok else status.get("last_test_email_at"),
            "last_test_email_ok": bool(ok),
            "last_test_email_error": "" if ok else error,
        })
        save_status(status)

        if ok:
            print("Short Cover Email Test: SUCCESS")
            return 0
        print(f"Short Cover Email Test: FAILED - {error}")
        return 1

    # Scheduled workflow runs a wider UTC window. Keep the actual monitoring
    # window strictly between 09:15 and 11:00 JST on weekdays. This covers
    # READY detection plus roughly 30-90 minutes of post-entry follow-up.
    if now.weekday() >= 5:
        print("Short Cover Entry Alert: weekend skip")
        return 0
    hhmm = now.hour * 60 + now.minute
    if hhmm < 9 * 60 + 15 or hhmm > 11 * 60:
        print(f"Short Cover Entry Alert: outside monitoring window ({now:%H:%M} JST)")
        return 0

    history = load_history()
    notifications = load_notifications()
    cfg = email_config()

    short_candidates = select_entry_hunter_candidates(
        history,
        as_of=now.tz_localize(None),
        max_calendar_days=4,
        limit=5,
    )
    me_candidates = select_me_entry_candidates(
        load_me_screener(),
        as_of=now.tz_localize(None),
        max_calendar_days=4,
        limit=5,
    )
    source_summary_for_rank = pd.DataFrame()
    if ENTRY_SOURCE_SUMMARY_FILE.exists():
        try:
            source_summary_for_rank = pd.read_csv(ENTRY_SOURCE_SUMMARY_FILE)
        except Exception:
            source_summary_for_rank = pd.DataFrame()

    signal_summary_for_rank = pd.DataFrame()
    if ENTRY_SIGNAL_SUMMARY_FILE.exists():
        try:
            signal_summary_for_rank = pd.read_csv(ENTRY_SIGNAL_SUMMARY_FILE)
        except Exception:
            signal_summary_for_rank = pd.DataFrame()

    trait_summary_for_rank = pd.DataFrame()
    if ENTRY_TRAIT_SUMMARY_FILE.exists():
        try:
            trait_summary_for_rank = pd.read_csv(ENTRY_TRAIT_SUMMARY_FILE)
        except Exception:
            trait_summary_for_rank = pd.DataFrame()

    candidates = combine_entry_candidates(
        short_candidates,
        me_candidates,
        limit=8,
        source_summary=source_summary_for_rank,
        signal_summary=signal_summary_for_rank,
        trait_summary=trait_summary_for_rank,
        universe_meta=load_me_universe(),
        fast_feedback_summary=load_after_close_feedback(),
    )

    status_rows = []
    new_ready = 0
    emails_sent = 0

    for _, candidate in candidates.iterrows():
        ticker = str(candidate["ticker"])
        try:
            daily, intraday = load_prices(ticker)
            entry = build_entry_hunter_snapshot(daily, intraday)
        except Exception as exc:
            entry = {
                "status": "⚪ NO DATA",
                "score": 0.0,
                "reason": "",
                "risk": "",
            }
            opportunity = build_entry_opportunity(
                candidate.to_dict(),
                entry,
            )
            status_rows.append({
                "ticker": ticker,
                "name": candidate.get("name", ""),
                "status": "⚪ NO DATA",
                "score": 0,
                "error": f"{type(exc).__name__}: {exc}"[:300],
                "source": candidate.get("source", ""),
                "source_detail": candidate.get("source_detail", ""),
                "signal_key": candidate.get("signal_key", ""),
                "trait_market": candidate.get("trait_market", ""),
                "trait_size": candidate.get("trait_size", ""),
                "trait_vol": candidate.get("trait_vol", ""),
                "opportunity_score": opportunity.get("opportunity_score"),
                "opportunity_rating": opportunity.get("opportunity_rating"),
                "opportunity_action": opportunity.get("opportunity_action"),
                "opportunity_reason": opportunity.get("opportunity_reason"),
                "opportunity_coverage": opportunity.get("opportunity_coverage"),
                "learning_confidence": opportunity.get("learning_confidence"),
            })
            continue

        market_date = pd.to_datetime(entry.get("market_date"), errors="coerce")
        alert_date = pd.to_datetime(candidate.get("alert_date"), errors="coerce")

        if (
            pd.notna(market_date)
            and pd.notna(alert_date)
            and pd.Timestamp(market_date).normalize()
            <= pd.Timestamp(alert_date).normalize()
        ):
            entry["status"] = "🟡 WAIT"
            entry["score"] = 0.0
            entry["reason"] = "翌営業日の取引データ待ち"
            entry["risk"] = ""

        opportunity = build_entry_opportunity(
            candidate.to_dict(),
            entry,
        )

        status_rows.append({
            "ticker": ticker,
            "name": candidate.get("name", ""),
            "status": entry.get("status", "⚪ NO DATA"),
            "score": entry.get("score", 0),
            "market_date": market_date,
            "gap_pct": entry.get("gap_pct"),
            "relvol15": entry.get("relvol15"),
            "reason": entry.get("reason", ""),
            "risk": entry.get("risk", ""),
            "source": candidate.get("source", ""),
            "source_detail": candidate.get("source_detail", ""),
            "signal_key": candidate.get("signal_key", ""),
            "trait_market": candidate.get("trait_market", ""),
            "trait_size": candidate.get("trait_size", ""),
            "trait_vol": candidate.get("trait_vol", ""),
            "opportunity_score": opportunity.get("opportunity_score"),
            "opportunity_rating": opportunity.get("opportunity_rating"),
            "opportunity_action": opportunity.get("opportunity_action"),
            "opportunity_reason": opportunity.get("opportunity_reason"),
            "opportunity_coverage": opportunity.get("opportunity_coverage"),
            "learning_confidence": opportunity.get("learning_confidence"),
        })

        if entry.get("status") != "🟢 ENTRY READY" or pd.isna(market_date):
            continue

        market_date_n = pd.Timestamp(market_date).normalize()
        mask = (
            (pd.to_datetime(notifications["market_date"], errors="coerce").dt.normalize() == market_date_n)
            & (notifications["ticker"].astype(str) == ticker)
            & (notifications["entry_status"].astype(str) == "🟢 ENTRY READY")
        )

        row = {
            "market_date": market_date_n,
            "ticker": ticker,
            "name": candidate.get("name", ""),
            "alert_date": alert_date,
            "condition_version": candidate.get("condition_version", ""),
            "entry_status": "🟢 ENTRY READY",
            "entry_score": entry.get("score"),
            "gap_pct": entry.get("gap_pct"),
            "relvol15": entry.get("relvol15"),
            "current_price": entry.get("current_price"),
            "vwap": entry.get("vwap"),
            "reason": entry.get("reason", ""),
            "risk": entry.get("risk", ""),
            "first_detected_at": now.tz_localize(None),
            "email_sent": False,
            "email_sent_at": pd.NaT,
            "email_error": "",
            "source": candidate.get("source", ""),
            "source_detail": candidate.get("source_detail", ""),
            "signal_key": candidate.get("signal_key", ""),
            "trait_market": candidate.get("trait_market", ""),
            "trait_size": candidate.get("trait_size", ""),
            "trait_vol": candidate.get("trait_vol", ""),
        }

        if not mask.any():
            notifications = pd.concat(
                [notifications, pd.DataFrame([row])],
                ignore_index=True,
            )
            idx = notifications.index[-1]
            new_ready += 1
        else:
            idx = notifications.index[mask][0]
            for key in [
                "entry_score", "gap_pct", "relvol15", "current_price",
                "vwap", "reason", "risk", "source", "source_detail",
                "signal_key", "trait_market", "trait_size", "trait_vol",
            ]:
                notifications.at[idx, key] = row[key]

        already_sent = _to_bool(notifications.at[idx, "email_sent"])
        if not already_sent:
            email_row = notifications.loc[idx].to_dict()
            ok, error = send_entry_email(email_row, cfg)
            notifications.at[idx, "email_sent"] = bool(ok)
            notifications.at[idx, "email_error"] = error
            if ok:
                notifications.at[idx, "email_sent_at"] = now.tz_localize(None)
                emails_sent += 1

    # Follow up every READY detected today. Each follow-up state is stored and
    # emailed at most once per ticker/day/status.
    ready_rows = notifications[
        (notifications["entry_status"].astype(str) == "🟢 ENTRY READY")
        & (
            pd.to_datetime(notifications["market_date"], errors="coerce").dt.normalize()
            == now.tz_localize(None).normalize()
        )
    ].copy()

    for _, ready_row in ready_rows.iterrows():
        ticker = str(ready_row["ticker"])
        try:
            daily, intraday = load_prices(ticker)
            followup = build_entry_followup_snapshot(
                daily,
                intraday,
                entry_price=ready_row.get("current_price"),
                entry_time=ready_row.get("first_detected_at"),
            )
        except Exception:
            continue

        follow_status = str(followup.get("status", "🟦 MONITORING"))
        if follow_status not in {
            "🟢 ENTRY CONFIRMED",
            "🟡 WEAKENING",
            "🔴 EXIT WATCH",
        }:
            continue

        market_date_n = pd.to_datetime(
            ready_row.get("market_date"),
            errors="coerce",
        )
        if pd.isna(market_date_n):
            continue
        market_date_n = pd.Timestamp(market_date_n).normalize()

        follow_mask = (
            (
                pd.to_datetime(
                    notifications["market_date"],
                    errors="coerce",
                ).dt.normalize()
                == market_date_n
            )
            & (notifications["ticker"].astype(str) == ticker)
            & (notifications["entry_status"].astype(str) == follow_status)
        )
        if follow_mask.any():
            continue

        follow_row = {
            "market_date": market_date_n,
            "ticker": ticker,
            "name": ready_row.get("name", ""),
            "alert_date": ready_row.get("alert_date"),
            "condition_version": ready_row.get("condition_version", ""),
            "entry_status": follow_status,
            "entry_score": ready_row.get("entry_score"),
            "gap_pct": ready_row.get("gap_pct"),
            "relvol15": ready_row.get("relvol15"),
            "current_price": followup.get("current_price"),
            "vwap": followup.get("vwap"),
            "reason": (
                f"{followup.get('reason', '')} | "
                f"MFE {followup.get('mfe_pct', '—')}% | "
                f"MAE {followup.get('mae_pct', '—')}% | "
                f"Entry比 {followup.get('return_pct', '—')}%"
            ),
            "risk": followup.get("risk", ""),
            "first_detected_at": now.tz_localize(None),
            "email_sent": False,
            "email_sent_at": pd.NaT,
            "email_error": "",
            "source": ready_row.get("source", ""),
            "source_detail": ready_row.get("source_detail", ""),
            "signal_key": ready_row.get("signal_key", ""),
            "trait_market": ready_row.get("trait_market", ""),
            "trait_size": ready_row.get("trait_size", ""),
            "trait_vol": ready_row.get("trait_vol", ""),
        }

        notifications = pd.concat(
            [notifications, pd.DataFrame([follow_row])],
            ignore_index=True,
        )
        idx = notifications.index[-1]

        ok, error = send_entry_email(
            notifications.loc[idx].to_dict(),
            cfg,
        )
        notifications.at[idx, "email_sent"] = bool(ok)
        notifications.at[idx, "email_error"] = error
        if ok:
            notifications.at[idx, "email_sent_at"] = now.tz_localize(None)
            emails_sent += 1

    save_notifications(notifications)

    candidate_history = update_candidate_history(
        load_candidate_history(),
        candidates,
        status_rows,
        observed_at=now.tz_localize(None),
    )
    save_candidate_history(candidate_history)

    perf_frames = {}
    ready_tickers = (
        notifications.loc[
            notifications["entry_status"].astype(str) == "🟢 ENTRY READY",
            "ticker",
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )
    for perf_ticker in ready_tickers:
        try:
            daily_frame, _ = load_prices(perf_ticker)
            perf_frames[str(perf_ticker)] = daily_frame
        except Exception:
            continue

    source_performance = build_ready_performance(
        notifications,
        perf_frames,
        prior=load_source_performance(),
        updated_at=now.tz_localize(None),
    )
    save_source_performance(source_performance)

    source_summary = summarize_source_performance(
        candidate_history,
        source_performance,
        notifications,
    )
    source_summary.to_csv(
        ENTRY_SOURCE_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    signal_summary = summarize_signal_performance(
        candidate_history,
        source_performance,
    )
    signal_summary.to_csv(
        ENTRY_SIGNAL_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    trait_summary = summarize_trait_performance(
        candidate_history,
        source_performance,
    )
    trait_summary.to_csv(
        ENTRY_TRAIT_SUMMARY_FILE,
        index=False,
        encoding="utf-8-sig",
    )

    status_payload = {
        "run_at": now.isoformat(),
        "email_configured": bool(cfg["to"] and cfg["user"] and cfg["password"]),
        "candidate_count": int(len(candidates)),
        "short_cover_candidates": int(len(short_candidates)),
        "me_candidates": int(len(me_candidates)),
        "confluence_candidates": int(sum(1 for x in candidates.get("source", pd.Series(dtype=str)).astype(str) if x == "SHORT+ME")),
        "ready_count": int(sum(
            1 for row in status_rows if row.get("status") == "🟢 ENTRY READY"
        )),
        "wait_count": int(sum(
            1 for row in status_rows if row.get("status") == "🟡 WAIT"
        )),
        "cancel_count": int(sum(
            1 for row in status_rows if row.get("status") == "🔴 CANCEL"
        )),
        "new_ready": int(new_ready),
        "confirmed_count": int(sum(
            1 for x in notifications["entry_status"].astype(str)
            if x == "🟢 ENTRY CONFIRMED"
        )),
        "weakening_count": int(sum(
            1 for x in notifications["entry_status"].astype(str)
            if x == "🟡 WEAKENING"
        )),
        "exit_watch_count": int(sum(
            1 for x in notifications["entry_status"].astype(str)
            if x == "🔴 EXIT WATCH"
        )),
        "emails_sent": int(emails_sent),
        "source_summary_rows": int(len(source_summary)),
        "signal_summary_rows": int(len(signal_summary)),
        "trait_summary_rows": int(len(trait_summary)),
        "top_opportunity": (
            max([float(x.get("opportunity_score", 0) or 0) for x in status_rows], default=0.0)
        ),
        "rows": status_rows,
    }
    save_status(status_payload)

    command_history = load_command_center_history()
    command_center = build_daily_command_center(
        status_payload,
        load_me_screener(),
        as_of=now.tz_localize(None),
        limit=3,
        history=command_history,
    )
    command_history = update_command_center_history(
        command_history,
        command_center,
        snapshot_at=now.tz_localize(None),
    )
    save_command_center_snapshot(
        command_center,
        command_history,
    )

    print(
        "Short Cover Entry Alert:",
        f"candidates={len(candidates)}",
        f"new_ready={new_ready}",
        f"emails_sent={emails_sent}",
        f"email_configured={bool(cfg['to'] and cfg['user'] and cfg['password'])}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

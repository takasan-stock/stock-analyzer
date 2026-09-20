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
    build_entry_hunter_snapshot,
    normalize_alert_history,
    select_entry_hunter_candidates,
)

DATA_DIR = ROOT / "data"
HISTORY_FILE = DATA_DIR / "short_cover_alert_history.csv"
NOTIFICATION_FILE = DATA_DIR / "short_cover_entry_notifications.csv"
STATUS_FILE = DATA_DIR / "short_cover_entry_status.json"

NOTIFICATION_COLUMNS = [
    "market_date", "ticker", "name", "alert_date", "condition_version",
    "entry_status", "entry_score", "gap_pct", "relvol15",
    "current_price", "vwap", "reason", "risk",
    "first_detected_at", "email_sent", "email_sent_at", "email_error",
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


def send_ready_email(row: dict, cfg: dict) -> tuple[bool, str]:
    if not (cfg["to"] and cfg["user"] and cfg["password"]):
        return False, "EMAIL_SECRETS_NOT_CONFIGURED"

    msg = EmailMessage()
    msg["Subject"] = (
        f"🚨 Short Cover ENTRY READY: {row['ticker']} {row['name']}"
    )
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

    body = f"""Short Cover Entry Hunter が ENTRY READY を検知しました。

銘柄: {row['ticker']} {row['name']}
Entry Score: {float(row['entry_score']):.0f}
Gap: {gap}
現在値: {price}
VWAP: {vwap}
15分相対出来高: {relvol}
前日アラート日: {pd.Timestamp(row['alert_date']).strftime('%Y-%m-%d')}
条件Version: {row.get('condition_version', '')}

成立条件:
{row.get('reason', '')}

注意:
{row.get('risk', '') or '特記事項なし'}

※これは売買推奨ではなく、TradingView/証券会社の現在値を確認するための監視通知です。
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
    return send_ready_email(row, cfg)


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
        if ok:
            print("Short Cover Email Test: SUCCESS")
            return 0
        print(f"Short Cover Email Test: FAILED - {error}")
        return 1

    # Scheduled workflow runs a wider UTC window. Keep the actual monitoring
    # window strictly between 09:15 and 10:00 JST on weekdays.
    if now.weekday() >= 5:
        print("Short Cover Entry Alert: weekend skip")
        return 0
    hhmm = now.hour * 60 + now.minute
    if hhmm < 9 * 60 + 15 or hhmm > 10 * 60:
        print(f"Short Cover Entry Alert: outside monitoring window ({now:%H:%M} JST)")
        return 0

    history = load_history()
    notifications = load_notifications()
    cfg = email_config()

    candidates = select_entry_hunter_candidates(
        history,
        as_of=now.tz_localize(None),
        max_calendar_days=4,
        limit=5,
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
            status_rows.append({
                "ticker": ticker,
                "name": candidate.get("name", ""),
                "status": "⚪ NO DATA",
                "score": 0,
                "error": f"{type(exc).__name__}: {exc}"[:300],
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
            # Refresh market fields while keeping the original first-detected time.
            for key in [
                "entry_score", "gap_pct", "relvol15", "current_price",
                "vwap", "reason", "risk",
            ]:
                notifications.at[idx, key] = row[key]

        already_sent = _to_bool(notifications.at[idx, "email_sent"])
        if not already_sent:
            email_row = notifications.loc[idx].to_dict()
            ok, error = send_ready_email(email_row, cfg)
            notifications.at[idx, "email_sent"] = bool(ok)
            notifications.at[idx, "email_error"] = error
            if ok:
                notifications.at[idx, "email_sent_at"] = now.tz_localize(None)
                emails_sent += 1

    save_notifications(notifications)

    save_status({
        "run_at": now.isoformat(),
        "email_configured": bool(cfg["to"] and cfg["user"] and cfg["password"]),
        "candidate_count": int(len(candidates)),
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
        "emails_sent": int(emails_sent),
        "rows": status_rows,
    })

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

from __future__ import annotations

import json
import os
import smtplib
from email.message import EmailMessage
from pathlib import Path

import pandas as pd

from me_screener_alerts import compact_change_summary


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "multiple_expansion"
CHANGE_FILE = DATA_DIR / "me_screener_changes.csv"
STATUS_FILE = DATA_DIR / "me_screener_alert_status.json"


def _env(primary: str, fallback: str = "") -> str:
    value = os.getenv(primary, "").strip()
    if value:
        return value
    if fallback:
        return os.getenv(fallback, "").strip()
    return ""


def email_config() -> dict:
    return {
        "to": _env("ME_EMAIL_TO", "SHORT_COVER_EMAIL_TO"),
        "user": _env("ME_EMAIL_USER", "SHORT_COVER_EMAIL_USER"),
        "password": _env(
            "ME_EMAIL_APP_PASSWORD",
            "SHORT_COVER_EMAIL_APP_PASSWORD",
        ),
        "host": (
            _env("ME_SMTP_HOST", "SHORT_COVER_SMTP_HOST")
            or "smtp.gmail.com"
        ),
        "port": int(
            _env("ME_SMTP_PORT", "SHORT_COVER_SMTP_PORT")
            or "465"
        ),
    }


def _fmt_num(value, digits=1, suffix=""):
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}{suffix}"
    except Exception:
        return "—"


def _load_changes() -> pd.DataFrame:
    if not CHANGE_FILE.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(CHANGE_FILE, dtype={"ticker": str})
    except Exception:
        return pd.DataFrame()


def _save_status(payload: dict) -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    STATUS_FILE.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )


def _subject(summary: pd.DataFrame) -> str:
    if summary.empty:
        return "ME Hunter: no actionable changes"

    priority = summary["events"].astype(str)
    if priority.str.contains("RE-EXP CONFIRMED").any():
        prefix = "🔥 RE-EXP"
    elif priority.str.contains("PRIORITY WATCH").any():
        prefix = "🔥 PRIORITY"
    elif priority.str.contains("R-READY").any():
        prefix = "🟢 R-READY"
    elif priority.str.contains("NEW R-EARLY").any():
        prefix = "🟡 R-EARLY"
    else:
        prefix = "📌 ME UPDATE"

    return f"{prefix}: {len(summary)}銘柄に重要変化"


def _build_body(summary: pd.DataFrame) -> str:
    lines = [
        "Multiple Expansion Daily Screener が重要な状態変化を検知しました。",
        "",
        "※順位そのものではなく、状態変化・優先度上昇だけを通知しています。",
        "",
    ]

    for _, row in summary.head(12).iterrows():
        ticker = str(row.get("ticker", ""))
        name = str(row.get("company_name", "") or "")
        events = str(row.get("events", ""))
        state = str(row.get("current_state", ""))
        decision = str(row.get("current_decision", ""))
        rank = _fmt_num(row.get("current_rank"), 0)
        sw = _fmt_num(row.get("sw_score"), 0, "/100")
        hist = _fmt_num(row.get("hist_edge_score"), 0, "/100")
        fcf = _fmt_num(row.get("fcf_engine_score"), 0, "/100")
        mlp = _fmt_num(row.get("mlp_c_core"), 3)
        vel = _fmt_num(row.get("me_velocity_pct"), 2, "%")
        acc = _fmt_num(row.get("me_acceleration_pct"), 2, "%")
        reason = str(row.get("screen_reason", "") or "")

        lines.extend(
            [
                f"【{ticker} {name}】",
                f"変化: {events}",
                f"State: {state}",
                f"Decision: {decision}",
                f"Rank: {rank} / SW Score: {sw}",
                f"MLP-C: {mlp} / Velocity: {vel} / Accel: {acc}",
                f"Hist Edge: {hist} / FCF Score: {fcf}",
                f"Reason: {reason}",
                "",
            ]
        )

    lines.extend(
        [
            "推奨確認順:",
            "ME Screener → TradingView → Entry Hunter → Pre-Trade",
            "",
            "※これは売買推奨ではなく、確認優先順位を絞るための監視通知です。",
        ]
    )
    return "\n".join(lines)


def send_digest(changes: pd.DataFrame, cfg: dict) -> tuple[bool, str]:
    if changes.empty:
        return True, "NO_CHANGES"

    if not (cfg["to"] and cfg["user"] and cfg["password"]):
        return False, "EMAIL_SECRETS_NOT_CONFIGURED"

    summary = compact_change_summary(changes)
    if summary.empty:
        return True, "NO_SUMMARY"

    msg = EmailMessage()
    msg["Subject"] = _subject(summary)
    msg["From"] = cfg["user"]
    msg["To"] = cfg["to"]
    msg.set_content(_build_body(summary))

    try:
        with smtplib.SMTP_SSL(
            cfg["host"],
            cfg["port"],
            timeout=20,
        ) as smtp:
            smtp.login(cfg["user"], cfg["password"])
            smtp.send_message(msg)
        return True, ""
    except Exception as exc:
        return False, f"{type(exc).__name__}: {exc}"[:500]


def main() -> int:
    changes = _load_changes()
    cfg = email_config()

    now = pd.Timestamp.now(tz="Asia/Tokyo")
    status = {
        "run_at": now.isoformat(),
        "email_configured": bool(
            cfg["to"] and cfg["user"] and cfg["password"]
        ),
        "change_count": int(len(changes)),
        "email_sent": False,
        "email_error": "",
    }

    if changes.empty:
        status["result"] = "NO_CHANGES"
        _save_status(status)
        print("[ME-ALERT] no actionable changes")
        return 0

    ok, error = send_digest(changes, cfg)
    status["email_sent"] = bool(ok and error == "")
    status["email_error"] = "" if ok else error
    status["result"] = "SENT" if ok and error == "" else error
    _save_status(status)

    if ok:
        print(f"[ME-ALERT] sent digest for {len(changes)} changes")
        return 0

    # Do not fail the whole market scan just because email is not configured.
    # The change CSV remains available in Streamlit and GitHub.
    print(f"[ME-ALERT] skipped/failed: {error}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

from __future__ import annotations

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st

from trade_journal import (
    JOURNAL_COLUMNS,
    build_group_summary,
    build_summary,
    calc_trade_outcome,
    normalize_journal,
    score_bucket,
    upsert_journal_record,
)


st.set_page_config(page_title="Trade Journal", page_icon="📓", layout="wide")

PLAN_FILE = "data/pretrade_trade_plans.csv"
JOURNAL_FILE = "data/trade_journal.csv"


def _github_config():
    try:
        return {
            "token": st.secrets["GITHUB_TOKEN"],
            "repo": st.secrets["GITHUB_REPO"],
            "branch": st.secrets.get("GITHUB_BRANCH", "main"),
        }
    except Exception:
        return None


def _headers(config):
    return {
        "Authorization": f"Bearer {config['token']}",
        "Accept": "application/vnd.github+json",
    }


def _load_csv(path: str, columns: list[str] | None = None) -> pd.DataFrame:
    config = _github_config()
    if config:
        url = f"https://api.github.com/repos/{config['repo']}/contents/{path}"
        try:
            resp = requests.get(
                url,
                headers=_headers(config),
                params={"ref": config["branch"]},
                timeout=10,
            )
            if resp.status_code == 200:
                raw = resp.json().get("content", "").replace("\n", "")
                text = base64.b64decode(raw).decode("utf-8-sig")
                df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
                if columns:
                    for col in columns:
                        if col not in df.columns:
                            df[col] = ""
                    return df[columns]
                return df
        except Exception:
            pass

    if os.path.exists(path):
        try:
            df = pd.read_csv(path, dtype=str).fillna("")
            if columns:
                for col in columns:
                    if col not in df.columns:
                        df[col] = ""
                return df[columns]
            return df
        except Exception:
            pass

    return pd.DataFrame(columns=columns or [])


def _save_csv(path: str, df: pd.DataFrame) -> tuple[bool, str]:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8-sig")

    config = _github_config()
    if not config:
        return True, "ローカル保存"

    url = f"https://api.github.com/repos/{config['repo']}/contents/{path}"
    current_sha = None
    try:
        get_resp = requests.get(
            url,
            headers=_headers(config),
            params={"ref": config["branch"]},
            timeout=10,
        )
        if get_resp.status_code == 200:
            current_sha = get_resp.json().get("sha")

        payload = {
            "message": "Update Trade Journal",
            "content": base64.b64encode(
                df.to_csv(index=False).encode("utf-8-sig")
            ).decode("ascii"),
            "branch": config["branch"],
        }
        if current_sha:
            payload["sha"] = current_sha

        resp = requests.put(
            url,
            headers=_headers(config),
            json=payload,
            timeout=15,
        )
        if resp.status_code in (200, 201):
            return True, "GitHubへ保存"
        return True, f"ローカル保存（GitHub保存失敗: {resp.status_code}）"
    except Exception as exc:
        return True, f"ローカル保存（GitHub保存失敗: {exc}）"


def _num(value, fallback=0.0) -> float:
    try:
        text = str(value).replace(",", "").replace("円", "").strip()
        return float(text) if text else float(fallback)
    except Exception:
        return float(fallback)


def _yen(value) -> str:
    try:
        return f"¥{float(value):,.0f}"
    except Exception:
        return "—"


plans = _load_csv(PLAN_FILE)
journal = normalize_journal(_load_csv(JOURNAL_FILE, JOURNAL_COLUMNS))

incoming_plan_id = str(st.session_state.pop("journal_plan_id", "") or "").strip()

st.title("📓 Trade Journal v1")
st.caption(
    "Pre-Trade Checkで確認した計画と、実際の約定・決済結果を紐づけます。"
    "ここでは『計画どおりに実行できたか』と『どんな条件で成績が良かったか』を検証します。"
)

# ---------------- Dashboard ----------------
summary = build_summary(journal)

st.markdown("## 📊 実績ダッシュボード")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("完了トレード", f"{summary['trades']}件")
m2.metric(
    "勝率",
    "—" if summary["win_rate"] is None else f"{summary['win_rate']:.1f}%",
)
m3.metric("累計損益", _yen(summary["total_pnl"]))
m4.metric(
    "平均R",
    "—" if summary["avg_r"] is None else f"{summary['avg_r']:+.2f}R",
)
_pf = summary["profit_factor"]
m5.metric(
    "Profit Factor",
    "—" if _pf is None else ("∞" if _pf == float("inf") else f"{_pf:.2f}"),
)

if summary["trades"] > 0:
    s1, s2, s3, s4 = st.columns(4)
    s1.metric(
        "平均騰落率",
        "—" if summary["avg_return_pct"] is None else f"{summary['avg_return_pct']:+.2f}%",
    )
    s2.metric(
        "R中央値",
        "—" if summary["median_r"] is None else f"{summary['median_r']:+.2f}R",
    )
    s3.metric(
        "平均保有日数",
        "—" if summary["avg_holding_days"] is None else f"{summary['avg_holding_days']:.1f}日",
    )
    s4.metric(
        "プラン遵守率",
        "—" if summary["plan_follow_rate"] is None else f"{summary['plan_follow_rate']:.1f}%",
    )

st.divider()

if plans.empty:
    st.info("確認済みのPre-Tradeプランがまだありません。")
    st.stop()

# Already journaled plan IDs
existing_by_plan = {
    str(r["plan_id"]): r.to_dict()
    for _, r in journal.iterrows()
    if str(r.get("plan_id", "")).strip()
}

plan_options = []
plan_map = {}
for _, p in plans.iloc[::-1].iterrows():
    plan_id = str(p.get("plan_id", ""))
    if not plan_id:
        continue
    ticker = str(p.get("ticker", ""))
    name = str(p.get("name", ""))
    status = str(existing_by_plan.get(plan_id, {}).get("status", "PLAN"))
    label = f"{status}｜{ticker} {name}｜{str(p.get('confirmed_at',''))[:16]}｜{plan_id}"
    plan_options.append(label)
    plan_map[label] = p.to_dict()

if not plan_options:
    st.info("記録可能な確認済みプランがありません。")
    st.stop()

default_index = 0
if incoming_plan_id:
    for i, label in enumerate(plan_options):
        if incoming_plan_id in label:
            default_index = i
            break

selected_label = st.selectbox(
    "記録する確認済みプラン",
    plan_options,
    index=default_index,
)
plan = plan_map[selected_label]
plan_id = str(plan.get("plan_id", ""))
existing = existing_by_plan.get(plan_id, {})

ticker = str(plan.get("ticker", ""))
name = str(plan.get("name", ""))

st.markdown(f"## {name or ticker}（{ticker}）")
p1, p2, p3, p4, p5 = st.columns(5)
p1.metric("Pre-Trade判定", str(plan.get("verdict", "—")))
p2.metric("Score", str(plan.get("total_score", "—")))
p3.metric("計画Entry", _yen(plan.get("entry")))
p4.metric("計画Stop", _yen(plan.get("stop")))
p5.metric("計画Target", _yen(plan.get("target")))

st.caption(
    f"Stage {plan.get('stage','—')}｜RS Proxy {plan.get('rs_proxy','—')}｜"
    f"Entry Hunter {plan.get('hunter_status','—')}｜"
    f"流入元 {plan.get('source','—')}"
)

st.markdown("### ① 実際のエントリー")

planned_entry = _num(plan.get("entry"))
planned_stop = _num(plan.get("stop"))
planned_target = _num(plan.get("target"))
planned_shares = int(_num(plan.get("shares"), 0))

today = pd.Timestamp.now(tz="Asia/Tokyo").date()

e1, e2, e3, e4 = st.columns(4)
with e1:
    entry_date = st.date_input(
        "約定日",
        value=(
            pd.to_datetime(existing.get("actual_entry_date")).date()
            if str(existing.get("actual_entry_date", "")).strip()
            else today
        ),
        key=f"journal_entry_date_{plan_id}",
    )
with e2:
    actual_entry = st.number_input(
        "実際の約定価格",
        min_value=0.0,
        value=_num(existing.get("actual_entry"), planned_entry),
        step=max(1.0, planned_entry * 0.001 if planned_entry else 1.0),
        key=f"journal_entry_{plan_id}",
    )
with e3:
    actual_stop = st.number_input(
        "実際に使うStop",
        min_value=0.0,
        value=_num(existing.get("actual_stop"), planned_stop),
        step=max(1.0, planned_entry * 0.001 if planned_entry else 1.0),
        key=f"journal_stop_{plan_id}",
    )
with e4:
    actual_shares = st.number_input(
        "実際の株数",
        min_value=0,
        value=int(_num(existing.get("actual_shares"), planned_shares)),
        step=100,
        key=f"journal_shares_{plan_id}",
    )

actual_target = st.number_input(
    "実際のTarget",
    min_value=0.0,
    value=_num(existing.get("actual_target"), planned_target),
    step=max(1.0, planned_entry * 0.001 if planned_entry else 1.0),
    key=f"journal_target_{plan_id}",
)

pre_outcome = calc_trade_outcome(
    planned_entry=planned_entry,
    actual_entry=actual_entry,
    actual_stop=actual_stop,
    shares=actual_shares,
    entry_date=entry_date,
)

x1, x2, x3 = st.columns(3)
x1.metric(
    "Entry Slippage",
    "—" if pre_outcome["entry_slippage_pct"] is None
    else f"{pre_outcome['entry_slippage_pct']:+.2f}%",
)
x2.metric(
    "1株リスク",
    _yen(pre_outcome["risk_per_share"]),
)
x3.metric(
    "実際の想定損失",
    _yen(pre_outcome["risk_amount"]),
)

st.markdown("### ② 決済")

status_options = ["OPEN", "CLOSED", "SKIPPED"]
existing_status = str(existing.get("status", "") or "OPEN")
status_index = status_options.index(existing_status) if existing_status in status_options else 0

d1, d2, d3 = st.columns(3)
with d1:
    status = st.selectbox(
        "状態",
        status_options,
        index=status_index,
        key=f"journal_status_{plan_id}",
    )
with d2:
    exit_date = st.date_input(
        "決済日",
        value=(
            pd.to_datetime(existing.get("actual_exit_date")).date()
            if str(existing.get("actual_exit_date", "")).strip()
            else today
        ),
        key=f"journal_exit_date_{plan_id}",
        disabled=(status != "CLOSED"),
    )
with d3:
    actual_exit = st.number_input(
        "決済価格",
        min_value=0.0,
        value=_num(existing.get("actual_exit"), 0.0),
        step=max(1.0, planned_entry * 0.001 if planned_entry else 1.0),
        key=f"journal_exit_{plan_id}",
        disabled=(status != "CLOSED"),
    )

fees = st.number_input(
    "手数料・諸経費",
    min_value=0.0,
    value=_num(existing.get("fees"), 0.0),
    step=100.0,
    key=f"journal_fees_{plan_id}",
    disabled=(status != "CLOSED"),
)

outcome = calc_trade_outcome(
    planned_entry=planned_entry,
    actual_entry=actual_entry,
    actual_stop=actual_stop,
    shares=actual_shares,
    actual_exit=actual_exit if status == "CLOSED" else None,
    fees=fees,
    entry_date=entry_date,
    exit_date=exit_date if status == "CLOSED" else None,
)

if status == "CLOSED":
    o1, o2, o3, o4 = st.columns(4)
    o1.metric("損益", _yen(outcome["pnl"]))
    o2.metric(
        "騰落率",
        "—" if outcome["return_pct"] is None else f"{outcome['return_pct']:+.2f}%",
    )
    o3.metric(
        "R倍",
        "—" if outcome["r_multiple"] is None else f"{outcome['r_multiple']:+.2f}R",
    )
    o4.metric(
        "保有日数",
        "—" if outcome["holding_days"] is None else f"{outcome['holding_days']}日",
    )

st.markdown("### ③ 振り返り")

r1, r2, r3 = st.columns(3)
with r1:
    exit_reason = st.selectbox(
        "決済理由",
        ["", "Target到達", "Stop到達", "トレーリング", "ルール売却", "裁量売却", "その他"],
        index=0,
        key=f"journal_exit_reason_{plan_id}",
        disabled=(status != "CLOSED"),
    )
with r2:
    followed_plan = st.selectbox(
        "プラン通りだった？",
        ["", "はい", "いいえ"],
        index=0,
        key=f"journal_followed_{plan_id}",
    )
with r3:
    rule_break = st.selectbox(
        "ルール違反",
        ["なし", "追いかけ買い", "Stop変更", "株数超過", "決算持越し", "その他"],
        index=0,
        key=f"journal_rule_break_{plan_id}",
    )

skip_reason = ""
if status == "SKIPPED":
    skip_reason = st.text_input(
        "見送った理由",
        value=str(existing.get("skip_reason", "")),
        key=f"journal_skip_reason_{plan_id}",
    )

memo = st.text_area(
    "振り返りメモ",
    value=str(existing.get("memo", "")),
    placeholder="例：ENTRY READY後に飛びついた。次回はVWAPへの押しを待つ。",
    key=f"journal_memo_{plan_id}",
)

save_disabled = False
validation = []
if status in {"OPEN", "CLOSED"}:
    if actual_entry <= 0:
        validation.append("約定価格")
    if actual_stop <= 0 or actual_stop >= actual_entry:
        validation.append("Stop")
    if actual_shares <= 0:
        validation.append("株数")
if status == "CLOSED" and actual_exit <= 0:
    validation.append("決済価格")

if validation:
    save_disabled = True
    st.warning("入力確認: " + " / ".join(validation))

if st.button(
    "💾 Trade Journalを保存",
    type="primary",
    use_container_width=True,
    disabled=save_disabled,
    key=f"journal_save_{plan_id}",
):
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    record = {
        "journal_id": str(existing.get("journal_id", "") or f"J-{plan_id}"),
        "plan_id": plan_id,
        "updated_at": now.isoformat(),
        "status": status,
        "ticker": ticker,
        "name": name,
        "source": plan.get("source", ""),
        "plan_confirmed_at": plan.get("confirmed_at", ""),
        "verdict": plan.get("verdict", ""),
        "total_score": plan.get("total_score", ""),
        "setup_score": plan.get("setup_score", ""),
        "entry_score": plan.get("entry_score", ""),
        "risk_score": plan.get("risk_score", ""),
        "stage": plan.get("stage", ""),
        "rs_proxy": plan.get("rs_proxy", ""),
        "volume_ratio": plan.get("volume_ratio", ""),
        "ema20_gap_pct": plan.get("ema20_gap_pct", ""),
        "hunter_status": plan.get("hunter_status", ""),
        "hunter_score": plan.get("hunter_score", ""),
        "planned_entry": planned_entry,
        "planned_stop": planned_stop,
        "planned_target": planned_target,
        "planned_rr": plan.get("rr", ""),
        "planned_shares": planned_shares,
        "planned_max_loss": plan.get("actual_max_loss", ""),
        "actual_entry_date": str(entry_date),
        "actual_entry": actual_entry,
        "actual_stop": actual_stop,
        "actual_target": actual_target,
        "actual_shares": actual_shares,
        "entry_slippage_pct": outcome.get("entry_slippage_pct", ""),
        "risk_per_share": outcome.get("risk_per_share", ""),
        "risk_amount": outcome.get("risk_amount", ""),
        "actual_exit_date": str(exit_date) if status == "CLOSED" else "",
        "actual_exit": actual_exit if status == "CLOSED" else "",
        "fees": fees if status == "CLOSED" else 0,
        "pnl": outcome.get("pnl", "") if status == "CLOSED" else "",
        "return_pct": outcome.get("return_pct", "") if status == "CLOSED" else "",
        "r_multiple": outcome.get("r_multiple", "") if status == "CLOSED" else "",
        "holding_days": outcome.get("holding_days", "") if status == "CLOSED" else "",
        "exit_reason": exit_reason if status == "CLOSED" else "",
        "skip_reason": skip_reason if status == "SKIPPED" else "",
        "followed_plan": followed_plan,
        "rule_break": rule_break,
        "memo": memo.strip(),
    }
    journal = upsert_journal_record(journal, record)
    ok, message = _save_csv(JOURNAL_FILE, journal)
    if ok:
        st.success(f"✅ Trade Journalを保存しました（{message}）")
        st.rerun()
    else:
        st.error(message)

st.divider()
st.markdown("## 🔬 条件別パフォーマンス")

journal = normalize_journal(_load_csv(JOURNAL_FILE, JOURNAL_COLUMNS))
closed = journal[journal["status"].astype(str) == "CLOSED"].copy()

if closed.empty:
    st.info("CLOSEDトレードが増えると、Score帯・Stage・Entry Hunter・流入元別の成績を表示します。")
else:
    tabs = st.tabs(["Score帯", "Stage", "Entry Hunter", "流入元", "全履歴"])

    with tabs[0]:
        score_summary = build_group_summary(journal, "score_bucket")
        st.dataframe(score_summary, hide_index=True, use_container_width=True)

    with tabs[1]:
        stage_summary = build_group_summary(journal, "stage")
        st.dataframe(stage_summary, hide_index=True, use_container_width=True)

    with tabs[2]:
        hunter_summary = build_group_summary(journal, "hunter_status")
        st.dataframe(hunter_summary, hide_index=True, use_container_width=True)

    with tabs[3]:
        source_summary = build_group_summary(journal, "source")
        st.dataframe(source_summary, hide_index=True, use_container_width=True)

    with tabs[4]:
        show = closed.copy().iloc[::-1]
        display_cols = [
            "actual_exit_date", "ticker", "name", "total_score", "status",
            "actual_entry", "actual_exit", "pnl", "return_pct", "r_multiple",
            "holding_days", "followed_plan", "rule_break", "exit_reason", "memo",
        ]
        st.dataframe(
            show[display_cols].rename(columns={
                "actual_exit_date": "決済日",
                "ticker": "コード",
                "name": "銘柄",
                "total_score": "Score",
                "status": "状態",
                "actual_entry": "Entry",
                "actual_exit": "Exit",
                "pnl": "損益",
                "return_pct": "騰落率%",
                "r_multiple": "R",
                "holding_days": "保有日数",
                "followed_plan": "プラン遵守",
                "rule_break": "ルール違反",
                "exit_reason": "決済理由",
                "memo": "メモ",
            }),
            hide_index=True,
            use_container_width=True,
        )

st.info(
    "Pre-Trade Scoreは今後固定ではなく、Trade Journalの実績から『どの条件が本当に効いたか』を検証し、"
    "必要なら重みを再調整するための土台として使います。",
    icon="🧪",
)

if st.button(
    "🧪 Score Calibrationを開く",
    use_container_width=True,
    key="open_score_calibration",
):
    st.switch_page("pages/11_Score_Calibration.py")

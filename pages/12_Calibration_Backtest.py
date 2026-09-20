from __future__ import annotations

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st

from calibration_backtest import build_calibration_backtest
from trade_journal import JOURNAL_COLUMNS, normalize_journal


st.set_page_config(page_title="Calibration Backtest", page_icon="🧭", layout="wide")

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


def load_journal() -> pd.DataFrame:
    config = _github_config()
    if config:
        url = f"https://api.github.com/repos/{config['repo']}/contents/{JOURNAL_FILE}"
        try:
            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {config['token']}",
                    "Accept": "application/vnd.github+json",
                },
                params={"ref": config["branch"]},
                timeout=10,
            )
            if resp.status_code == 200:
                raw = resp.json().get("content", "").replace("\n", "")
                text = base64.b64decode(raw).decode("utf-8-sig")
                return normalize_journal(
                    pd.read_csv(io.StringIO(text), dtype=str).fillna("")
                )
        except Exception:
            pass

    if os.path.exists(JOURNAL_FILE):
        try:
            return normalize_journal(
                pd.read_csv(JOURNAL_FILE, dtype=str).fillna("")
            )
        except Exception:
            pass

    return pd.DataFrame(columns=JOURNAL_COLUMNS)


def _fmt(value, digits=3, suffix=""):
    if value is None or pd.isna(value):
        return "—"
    return f"{float(value):+.{digits}f}{suffix}"


st.title("🧭 Calibration Backtest v1")
st.caption(
    "Score Calibrationで作った実験ウェイトを、後の期間だけで検証します。"
    "前半データで重みを作り、後半データでは重みを触らず評価するため、"
    "同じデータへの過剰適合を抑えます。"
)

journal = load_journal()

st.markdown("## ① 検証設定")
c1, c2 = st.columns(2)
with c1:
    train_fraction = st.slider(
        "学習期間の割合",
        min_value=0.50,
        max_value=0.80,
        value=0.60,
        step=0.05,
        help="時系列の前半を学習、後半を検証に使います。",
    )
with c2:
    prefer_clean = st.checkbox(
        "クリーン実績を優先",
        value=True,
        help="十分な件数があれば、ルール違反なし・プラン遵守の実績を優先します。",
    )

report = build_calibration_backtest(
    journal,
    train_fraction=train_fraction,
    prefer_clean=prefer_clean,
)

st.markdown("## ② Backtest Status")
m1, m2, m3, m4 = st.columns(4)
m1.metric("状態", report["status"])
m2.metric("全体", f"{report['total']}件")
m3.metric("学習", f"{report['train_n']}件")
m4.metric("検証", f"{report['validation_n']}件")

st.info(report["message"], icon="ℹ️")

if report["train_n"] == 0 or report["validation_n"] == 0:
    st.warning(
        "Trade JournalのCLOSEDトレードが増えると、時系列分割バックテストを実行できます。"
    )
    st.stop()

st.caption(
    f"サンプル: {report['sample_mode']}｜"
    f"学習 {pd.Timestamp(report['train_start']).date()}〜{pd.Timestamp(report['train_end']).date()}｜"
    f"検証 {pd.Timestamp(report['validation_start']).date()}〜{pd.Timestamp(report['validation_end']).date()}"
)

st.divider()
st.markdown("## ③ 学習期間で作った実験ウェイト")
weights = report["weights"]

w1, w2, w3 = st.columns(3)
w1.metric("Setup Quality", f"{weights['Setup Quality']:.1f}")
w2.metric("Entry Timing", f"{weights['Entry Timing']:.1f}")
w3.metric("Risk Control", f"{weights['Risk Control']:.1f}")

st.caption("現行ウェイトは Setup 40 / Entry 35 / Risk 25。実験ウェイトは検証期間では固定です。")

st.divider()
st.markdown("## ④ 現行 vs 実験 — 検証期間だけで比較")

comparison = report["comparison"].copy()
if comparison.empty:
    st.caption("比較可能な検証データがありません。")
else:
    formatted = comparison.copy()
    formatted["current"] = formatted["current"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}"
    )
    formatted["experimental"] = formatted["experimental"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}"
    )
    formatted["delta"] = formatted["delta"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}"
    )
    st.dataframe(
        formatted.rename(columns={
            "metric": "指標",
            "current": "現行",
            "experimental": "実験",
            "delta": "実験-現行",
        }),
        hide_index=True,
        use_container_width=True,
    )

cur = report["current_metrics"]
exp = report["experimental_metrics"]

a1, a2, a3 = st.columns(3)
a1.metric(
    "Score↔R相関",
    _fmt(exp.get("spearman_r")),
    delta=(
        None if exp.get("spearman_r") is None or cur.get("spearman_r") is None
        else f"{exp['spearman_r'] - cur['spearman_r']:+.3f}"
    ),
)
a2.metric(
    "上位25% 平均R",
    _fmt(exp.get("top_avg_r"), 2, "R"),
    delta=(
        None if exp.get("top_avg_r") is None or cur.get("top_avg_r") is None
        else f"{exp['top_avg_r'] - cur['top_avg_r']:+.2f}R"
    ),
)
a3.metric(
    "上位-下位 R差",
    _fmt(exp.get("top_bottom_spread"), 2, "R"),
    delta=(
        None if exp.get("top_bottom_spread") is None or cur.get("top_bottom_spread") is None
        else f"{exp['top_bottom_spread'] - cur['top_bottom_spread']:+.2f}R"
    ),
)

st.caption(
    "ここでの『上位25%』は各スコア方式で高Scoreになった検証トレードです。"
    "同じ検証期間に対して、現行ウェイトと実験ウェイトを別々に並べ替えて比較します。"
)

st.divider()
st.markdown("## ⑤ 検証トレード明細")

validation = report["validation"].copy()
if validation.empty:
    st.caption("検証明細はありません。")
else:
    for col in ["r_multiple", "setup_score", "entry_score", "risk_score", "current_score_bt", "experimental_score_bt"]:
        if col in validation.columns:
            validation[col] = pd.to_numeric(validation[col], errors="coerce")

    st.dataframe(
        validation.rename(columns={
            "actual_entry_date": "Entry日",
            "ticker": "コード",
            "name": "銘柄",
            "r_multiple": "実現R",
            "setup_score": "Setup",
            "entry_score": "Entry",
            "risk_score": "Risk",
            "current_score_bt": "現行Score",
            "experimental_score_bt": "実験Score",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.markdown("## ⑥ 判定の使い方")

st.markdown(
    """
- **DATA BUILDING:** まだ件数不足。判断しない。
- **SMALL VALIDATION:** 検証件数が少ない。参考のみ。
- **EXPERIMENT SHOWS PROMISE:** 複数の選別指標で改善が見られた。
- **MIXED RESULT:** 改善と悪化が混在。
- **EXPERIMENT NOT CONFIRMED:** 検証側で改善を確認できなかった。

**重要:** どの状態でも本番ウェイトは自動変更しません。
"""
)

st.success(
    "このBacktestは『重みを作るデータ』と『重みを評価するデータ』を分けるため、"
    "Calibration単体より一段厳しい検証です。",
    icon="🧭",
)

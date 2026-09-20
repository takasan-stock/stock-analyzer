from __future__ import annotations

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st

from trade_journal import JOURNAL_COLUMNS, normalize_journal
from walk_forward_calibration import build_walk_forward_report


st.set_page_config(page_title="Walk-Forward Calibration", page_icon="🚶", layout="wide")

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


st.title("🚶 Walk-Forward Calibration v1")
st.caption(
    "学習→検証を1回だけではなく、時間を前へ進めながら何度も繰り返します。"
    "実験ウェイトが複数期間で安定して効くかを確認する、より厳しい検証です。"
)

journal = load_journal()

st.markdown("## ① Walk-Forward設定")
c1, c2, c3, c4 = st.columns(4)
with c1:
    initial_train = st.number_input(
        "初期学習件数",
        min_value=10,
        max_value=100,
        value=20,
        step=5,
    )
with c2:
    validation_window = st.number_input(
        "1回の検証件数",
        min_value=5,
        max_value=50,
        value=10,
        step=5,
    )
with c3:
    step = st.number_input(
        "前進ステップ",
        min_value=1,
        max_value=50,
        value=10,
        step=1,
        help="通常は検証件数と同じにすると、検証区間が重複しません。",
    )
with c4:
    prefer_clean = st.checkbox(
        "クリーン実績を優先",
        value=True,
        help="十分な件数があれば、ルール違反なし・プラン遵守の実績を優先します。",
    )

report = build_walk_forward_report(
    journal,
    initial_train=int(initial_train),
    validation_window=int(validation_window),
    step=int(step),
    prefer_clean=prefer_clean,
)

st.markdown("## ② Walk-Forward Status")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("状態", report["status"])
m2.metric("有効実績", f"{report['total']}件")
m3.metric("検証区間", f"{len(report['folds'])}区間")
m4.metric("OOS検証延べ", f"{report['validation_total']}件")
m5.metric(
    "PROMISE率",
    "—" if report["promise_rate"] is None else f"{report['promise_rate']:.1f}%",
)

st.info(report["message"], icon="ℹ️")

if not report["folds"]:
    st.warning(
        f"Walk-Forward開始には、初期学習{report['initial_train']}件 + "
        f"検証{report['validation_window']}件以上の有効実績が必要です。"
    )
    st.stop()

st.caption(
    f"サンプル: {report['sample_mode']}｜"
    f"初期学習 {report['initial_train']}件｜"
    f"検証窓 {report['validation_window']}件｜"
    f"前進 {report['step']}件"
)

st.divider()
st.markdown("## ③ 区間ごとの結果")

fold_table = report["fold_table"].copy()
if fold_table.empty:
    st.caption("区間結果はありません。")
else:
    display = fold_table.copy()
    date_cols = ["train_end", "validation_start", "validation_end"]
    for col in date_cols:
        display[col] = pd.to_datetime(display[col], errors="coerce").dt.date.astype(str)

    numeric_cols = [
        "current_corr", "experimental_corr", "delta_corr",
        "current_top_avg_r", "experimental_top_avg_r", "delta_top_avg_r",
        "current_top_win_rate", "experimental_top_win_rate", "delta_top_win_rate",
        "current_spread", "experimental_spread", "delta_spread",
        "setup_weight", "entry_weight", "risk_weight",
    ]
    for col in numeric_cols:
        if col in display.columns:
            display[col] = pd.to_numeric(display[col], errors="coerce")

    st.dataframe(
        display.rename(columns={
            "fold": "区間",
            "train_n": "学習件数",
            "validation_n": "検証件数",
            "train_end": "学習終了",
            "validation_start": "検証開始",
            "validation_end": "検証終了",
            "verdict": "判定",
            "positive_metrics": "改善指標数",
            "negative_metrics": "悪化指標数",
            "current_corr": "現行 相関",
            "experimental_corr": "実験 相関",
            "delta_corr": "相関差",
            "current_top_avg_r": "現行 上位R",
            "experimental_top_avg_r": "実験 上位R",
            "delta_top_avg_r": "上位R差",
            "current_top_win_rate": "現行 上位勝率",
            "experimental_top_win_rate": "実験 上位勝率",
            "delta_top_win_rate": "上位勝率差",
            "current_spread": "現行 R差",
            "experimental_spread": "実験 R差",
            "delta_spread": "R差改善",
            "setup_weight": "Setup重み",
            "entry_weight": "Entry重み",
            "risk_weight": "Risk重み",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.markdown("## ④ 重みの安定性")

stability = report["weight_stability"].copy()
if stability.empty:
    st.caption("重み安定性を計算できません。")
else:
    st.dataframe(
        stability.rename(columns={
            "component": "項目",
            "mean_weight": "平均",
            "min_weight": "最小",
            "max_weight": "最大",
            "std_weight": "標準偏差",
            "range": "レンジ",
        }),
        hide_index=True,
        use_container_width=True,
    )

    s1, s2, s3 = st.columns(3)
    for col, row in zip([s1, s2, s3], stability.to_dict("records")):
        col.metric(
            row["component"],
            f"{row['mean_weight']:.1f}",
            delta=f"σ {row['std_weight']:.2f}",
            delta_color="off",
        )

st.caption(
    "重みの標準偏差が大きいほど、期間によって最適重みが揺れている可能性があります。"
    "固定ウェイト変更より、相場局面別ルールを検討した方がよいケースもあります。"
)

st.divider()
st.markdown("## ⑤ 全Out-of-Sample集約")

cur = report["aggregate_current"]
exp = report["aggregate_experimental"]
deltas = report["aggregate_deltas"]

a1, a2, a3, a4 = st.columns(4)
a1.metric(
    "Score↔R相関",
    _fmt(exp.get("spearman_r")),
    delta=(
        None if deltas.get("spearman_r") is None
        else f"{deltas['spearman_r']:+.3f}"
    ),
)
a2.metric(
    "上位25% 平均R",
    _fmt(exp.get("top_avg_r"), 2, "R"),
    delta=(
        None if deltas.get("top_avg_r") is None
        else f"{deltas['top_avg_r']:+.2f}R"
    ),
)
a3.metric(
    "上位25% 勝率",
    "—" if exp.get("top_win_rate") is None else f"{exp['top_win_rate']:.1f}%",
    delta=(
        None if deltas.get("top_win_rate") is None
        else f"{deltas['top_win_rate']:+.1f}pt"
    ),
)
a4.metric(
    "上位-下位 R差",
    _fmt(exp.get("top_bottom_spread"), 2, "R"),
    delta=(
        None if deltas.get("top_bottom_spread") is None
        else f"{deltas['top_bottom_spread']:+.2f}R"
    ),
)

st.caption(
    "deltaは『実験ウェイト − 現行ウェイト』です。"
    "各区間で未来データを見ずに作ったスコアを、Out-of-Sampleだけで合算しています。"
)

st.divider()
st.markdown("## ⑥ OOS検証明細")

oos = report["oos_validation"].copy()
if oos.empty:
    st.caption("OOS明細はありません。")
else:
    for col in [
        "r_multiple", "setup_score", "entry_score", "risk_score",
        "current_score_wf", "experimental_score_wf",
    ]:
        if col in oos.columns:
            oos[col] = pd.to_numeric(oos[col], errors="coerce")

    st.dataframe(
        oos.rename(columns={
            "wf_fold": "区間",
            "actual_entry_date": "Entry日",
            "ticker": "コード",
            "name": "銘柄",
            "r_multiple": "実現R",
            "setup_score": "Setup",
            "entry_score": "Entry",
            "risk_score": "Risk",
            "current_score_wf": "現行Score",
            "experimental_score_wf": "実験Score",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.markdown("## ⑦ 判定の使い方")

st.markdown(
    """
- **DATA BUILDING:** 件数不足。判断しない。
- **EARLY WALK-FORWARD:** 区間数・検証件数がまだ少ない。
- **STABLE PROMISE:** 複数区間で改善し、集約OOSでも改善、重みのブレも比較的小さい。
- **REGIME DEPENDENT / MIXED:** 時期によって効き方が違う可能性。
- **NOT ROBUST:** 複数区間で改善が安定しない。

**重要:** STABLE PROMISEでも本番ウェイトは自動変更しません。
"""
)

st.success(
    "Walk-Forwardは、1回の偶然ではなく『複数の未来区間で再現したか』を見るための検証です。",
    icon="🚶",
)

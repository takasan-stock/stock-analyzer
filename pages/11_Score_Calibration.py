from __future__ import annotations

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st

from score_calibration import build_calibration_report
from trade_journal import JOURNAL_COLUMNS, build_group_summary, normalize_journal


st.set_page_config(page_title="Score Calibration", page_icon="🧪", layout="wide")

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
                df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
                return normalize_journal(df)
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


st.title("🧪 Score Calibration v1")
st.caption(
    "Trade Journalの実績から、Pre-Trade Scoreの各要素が実現Rとどう関係したかを検証します。"
    "ここで表示する実験重みは参考値で、Pre-Trade本番ロジックには自動反映しません。"
)

journal = load_journal()
report = build_calibration_report(journal)

st.markdown("## ① 学習ステータス")
c1, c2, c3, c4 = st.columns(4)
c1.metric("状態", report["status"])
c2.metric("信頼度", report["confidence"])
c3.metric("有効サンプル", f"{report['sample_size']}件")
c4.metric("クリーン実績", f"{report['clean_closed_with_r']}件")

st.info(report["message"], icon="ℹ️")

if report["sample_mode"] == "CLEAN":
    st.caption(
        "Calibrationには、原則として「ルール違反なし」かつ、記録がある場合は"
        "「プラン通り」のCLOSEDトレードを優先使用しています。"
    )
elif report["sample_mode"] == "ALL":
    st.caption(
        "クリーン実績が10件未満のため、現時点では全CLOSEDトレードを参考サンプルに使っています。"
    )

st.divider()
st.markdown("## ② 現行ウェイト vs 実験ウェイト")

weights = report["weights"].copy()
if weights.empty:
    st.caption("重み分析に使える実績がありません。")
else:
    display = weights.copy()
    display["current_weight"] = display["current_weight"].map(lambda x: f"{float(x):.1f}")
    display["experimental_weight"] = display["experimental_weight"].map(lambda x: f"{float(x):.1f}")
    display["change"] = display["change"].map(lambda x: f"{float(x):+.1f}")
    display["spearman_r"] = display["spearman_r"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}"
    )
    display["eligible"] = display["eligible"].map(lambda x: "✅" if bool(x) else "学習中")

    st.dataframe(
        display.rename(columns={
            "component": "項目",
            "current_weight": "現行",
            "experimental_weight": "実験",
            "change": "差",
            "spearman_r": "Rとの順位相関",
            "eligible": "状態",
        }),
        hide_index=True,
        use_container_width=True,
    )

    p = report["proposed_weights"]
    w1, w2, w3 = st.columns(3)
    w1.metric("Setup Quality", f"{p['Setup Quality']:.1f}")
    w2.metric("Entry Timing", f"{p['Entry Timing']:.1f}")
    w3.metric("Risk Control", f"{p['Risk Control']:.1f}")

    if report["sample_size"] < report["min_trades"]:
        st.warning(
            f"{report['min_trades']}件に届くまでは現行40/35/25を維持します。"
            "少数サンプルで重みを動かすと過学習しやすいためです。"
        )
    else:
        st.warning(
            "実験重みは『検討候補』です。相関は因果関係を証明しないため、"
            "本番反映は別の検証ステップを通してから行います。"
        )

st.divider()
st.markdown("## ③ 3大スコアと実現R")

components = report["components"].copy()
if components.empty:
    st.caption("分析可能なデータがありません。")
else:
    components["spearman_r"] = components["spearman_r"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}"
    )
    st.dataframe(
        components.rename(columns={
            "component": "項目",
            "base_weight": "現行ウェイト",
            "n": "件数",
            "spearman_r": "順位相関",
            "signal": "読み方",
            "sample_mode": "サンプル",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.markdown("## ④ 個別ファクター診断")
factors = report["factors"].copy()
if factors.empty:
    st.caption("個別ファクターを評価できるデータがありません。")
else:
    factors["spearman_r"] = factors["spearman_r"].map(
        lambda x: "—" if pd.isna(x) else f"{float(x):+.3f}"
    )
    st.dataframe(
        factors.rename(columns={
            "factor": "ファクター",
            "n": "件数",
            "spearman_r": "実現Rとの順位相関",
            "signal": "読み方",
            "sample_mode": "サンプル",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.caption(
    "Spearman順位相関は『数値が高いほどRも高くなる傾向があったか』を見る指標です。"
    " +1に近いほど同方向、-1に近いほど逆方向ですが、件数が少ない段階では大きく変動します。"
)

st.divider()
st.markdown("## ⑤ Score帯の実績")

score_summary = build_group_summary(journal, "score_bucket")
if score_summary.empty:
    st.caption("CLOSEDトレードが増えるとScore帯ごとの実績を表示します。")
else:
    show = score_summary.copy()
    for col in ["win_rate", "avg_r", "median_r", "avg_return_pct", "total_pnl"]:
        show[col] = pd.to_numeric(show[col], errors="coerce")
    st.dataframe(
        show.rename(columns={
            "score_bucket": "Score帯",
            "trades": "件数",
            "wins": "勝ち",
            "win_rate": "勝率%",
            "avg_r": "平均R",
            "median_r": "R中央値",
            "avg_return_pct": "平均騰落率%",
            "total_pnl": "累計損益",
        }),
        hide_index=True,
        use_container_width=True,
    )

st.divider()
st.markdown("## ⑥ 運用ルール")

st.markdown(
    """
- **0〜19件:** 学習中。40/35/25を固定。
- **20〜39件:** 実験重みを表示するが、本番には反映しない。
- **40件以上:** 重み見直し候補としてレビュー可能。
- **ルール違反トレード:** スコア自体の評価を歪めるため、十分な件数があればCalibrationから除外。
- **本番反映:** 今後、旧ウェイトと新ウェイトを過去実績で比較してから手動承認する。
"""
)

st.success(
    "Score Calibration v1は『自動で賢くなる』のではなく、"
    "まず実績から仮説を作り、過学習を避けながら検証する仕組みです。",
    icon="🧪",
)

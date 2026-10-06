from __future__ import annotations

import os

import pandas as pd
import streamlit as st

from multiple_expansion_screener import build_daily_screener, candidate_only


st.set_page_config(
    page_title="ME Daily Screener",
    page_icon="🔥",
    layout="wide",
)

HISTORY_FILE = "data/multiple_expansion/mex_history.csv"
SCREENER_FILE = "data/multiple_expansion/me_screener_latest.csv"
CHANGE_FILE = "data/multiple_expansion/me_screener_changes.csv"


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _pct(value, digits=1):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value) * 100:.{digits}f}%"
    except Exception:
        return "—"


def _pct_raw(value, digits=1):
    try:
        if pd.isna(value):
            return "—"
        return f"{float(value):.{digits}f}%"
    except Exception:
        return "—"


st.title("🔥 Multiple Expansion Daily Screener v0.9")
st.caption(
    "マルチプル・エクスパンションの第1波/第2波が始まりそうな銘柄を、"
    "FCF・過去エッジ・MLP再加速形状を合わせて毎日順位付けします。"
)

cached = _load_csv(SCREENER_FILE)
history = _load_csv(HISTORY_FILE)
changes = _load_csv(CHANGE_FILE)

if cached.empty:
    if history.empty:
        st.info(
            "MEX履歴がまだありません。先にMEXビルドを実行すると、"
            "このページに候補ランキングが表示されます。"
        )
        st.stop()
    with st.spinner("MEX履歴からDaily Screenerを計算しています..."):
        screener = build_daily_screener(history)
else:
    screener = cached.copy()

if screener.empty:
    st.warning("スクリーナー結果がありません。")
    st.stop()

if "screen_rank" not in screener.columns and not history.empty:
    screener = build_daily_screener(history)

candidates = candidate_only(screener)

if not changes.empty:
    st.markdown("## 🚨 今日の重要変化")
    change_cols = [
        col
        for col in [
            "event_type",
            "ticker",
            "company_name",
            "previous_state",
            "current_state",
            "current_decision",
            "current_rank",
            "sw_score",
            "hist_edge_score",
            "fcf_engine_score",
            "change_reason",
        ]
        if col in changes.columns
    ]
    change_view = changes[change_cols].copy()
    change_view = change_view.rename(
        columns={
            "event_type": "変化",
            "ticker": "Code",
            "company_name": "銘柄",
            "previous_state": "前回",
            "current_state": "現在",
            "current_decision": "Decision",
            "current_rank": "Rank",
            "sw_score": "SW Score",
            "hist_edge_score": "Hist Edge",
            "fcf_engine_score": "FCF Score",
            "change_reason": "理由",
        }
    )
    st.dataframe(
        change_view,
        hide_index=True,
        use_container_width=True,
        height=min(300, 72 + 34 * len(change_view)),
    )
    st.caption("同じ状態のままなら通知しません。R-EARLY / R-READY / RE-EXP / PRIORITY WATCH / TOP10新規だけを抽出します。")
else:
    st.info("今日は新しい重要状態変化はありません。ランキングは下で確認できます。")

c1, c2, c3, c4 = st.columns(4)
c1.metric("候補数", len(candidates))
c2.metric(
    "PRIORITY WATCH",
    int((candidates.get("sw_decision") == "PRIORITY WATCH").sum())
    if "sw_decision" in candidates.columns else 0,
)
c3.metric(
    "R-READY",
    int((candidates.get("second_wave_state") == "RE-WATCH READY").sum())
    if "second_wave_state" in candidates.columns else 0,
)
c4.metric(
    "RE-EXP",
    int((candidates.get("second_wave_state") == "RE-EXP").sum())
    if "second_wave_state" in candidates.columns else 0,
)

st.markdown("## 今日見るべき銘柄")

decision_options = [
    x
    for x in ["PRIORITY WATCH", "READY", "ACTIVE", "WATCH", "FIRST WAVE"]
    if x in set(candidates.get("sw_decision", pd.Series(dtype=str)).astype(str))
]
selected_decisions = st.multiselect(
    "判定",
    options=decision_options,
    default=decision_options,
)

f1, f2, f3 = st.columns(3)
min_score = f1.slider("SW Score 最低", 0, 100, 0, 5)
min_hist = f2.slider("Hist Edge 最低", 0, 100, 0, 5)
candidate_type = f3.selectbox(
    "候補タイプ",
    ["ALL", "SECOND WAVE", "FIRST WAVE"],
)

view = candidates.copy()
if selected_decisions:
    view = view[view["sw_decision"].isin(selected_decisions)]
if "sw_score" in view.columns:
    view = view[pd.to_numeric(view["sw_score"], errors="coerce").fillna(-1) >= min_score]
if "hist_edge_score" in view.columns:
    view = view[
        pd.to_numeric(view["hist_edge_score"], errors="coerce").fillna(-1) >= min_hist
    ]
if candidate_type != "ALL" and "candidate_type" in view.columns:
    view = view[view["candidate_type"] == candidate_type]

cols = [
    c
    for c in [
        "screen_rank",
        "ticker",
        "company_name",
        "market_name",
        "candidate_type",
        "second_wave_state",
        "sw_decision",
        "sw_score",
        "fundamental_label",
        "fcf_engine_score",
        "mlp_c_core",
        "me_velocity_pct",
        "me_acceleration_pct",
        "me_curvature_pct",
        "hist_edge_score",
        "hist_edge_n",
        "hist_60d_win",
        "hist_60d_excess",
        "re_engine",
        "re_route",
        "screen_reason",
    ]
    if c in view.columns
]
display = view[cols].copy()

rename = {
    "screen_rank": "Rank",
    "ticker": "Code",
    "company_name": "銘柄",
    "market_name": "市場",
    "candidate_type": "Wave",
    "second_wave_state": "State",
    "sw_decision": "Decision",
    "sw_score": "SW Score",
    "fundamental_label": "FCF",
    "fcf_engine_score": "FCF Score",
    "mlp_c_core": "MLP-C",
    "me_velocity_pct": "Velocity",
    "me_acceleration_pct": "Accel",
    "me_curvature_pct": "Curvature",
    "hist_edge_score": "Hist Edge",
    "hist_edge_n": "Hist N",
    "hist_60d_win": "60D Win",
    "hist_60d_excess": "60D Excess",
    "re_engine": "RE Engine",
    "re_route": "Route",
    "screen_reason": "Reason",
}
display = display.rename(columns=rename)

for col in ["SW Score", "FCF Score", "MLP-C", "Hist Edge"]:
    if col in display.columns:
        display[col] = pd.to_numeric(display[col], errors="coerce").round(1)
for col in ["Velocity", "Accel", "Curvature"]:
    if col in display.columns:
        display[col] = display[col].map(lambda x: _pct_raw(x, 2))
if "60D Win" in display.columns:
    display["60D Win"] = display["60D Win"].map(lambda x: _pct(x, 1))
if "60D Excess" in display.columns:
    display["60D Excess"] = display["60D Excess"].map(lambda x: _pct(x, 1))

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
    height=520,
)

if view.empty:
    st.warning("現在のフィルター条件に該当する銘柄はありません。")
    st.stop()

st.divider()
st.markdown("## 銘柄詳細")

options = view["ticker"].astype(str).tolist()
selected = st.selectbox("銘柄", options=options)
row = view[view["ticker"].astype(str) == selected].iloc[0]

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Decision", str(row.get("sw_decision", "—")))
m2.metric("State", str(row.get("second_wave_state", "—")))
m3.metric("SW Score", f"{float(row.get('sw_score', 0)):.0f}/100")
m4.metric("MLP-C", f"{float(row.get('mlp_c_core', 0)):.3f}")
m5.metric("Velocity", _pct_raw(row.get("me_velocity_pct"), 2))
m6.metric("Accel", _pct_raw(row.get("me_acceleration_pct"), 2))

left, right = st.columns(2)
with left:
    st.markdown("### 再加速エンジン")
    st.write({
        "RE Engine": row.get("re_engine", "—"),
        "Route": row.get("re_route", "—"),
        "MLP Pullback": _pct(row.get("mlp_pullback"), 1),
        "Advance Gate": row.get("advance_gate_progress", "—"),
        "Curvature": _pct_raw(row.get("me_curvature_pct"), 2),
    })

with right:
    st.markdown("### 過去エッジ / FCF")
    st.write({
        "Hist Edge": row.get("hist_edge_score", "—"),
        "Hist N": row.get("hist_edge_n", "—"),
        "60D Win": _pct(row.get("hist_60d_win"), 1),
        "60D Excess": _pct(row.get("hist_60d_excess"), 1),
        "FCF Engine": row.get("fcf_engine_score", "—"),
        "FCF": row.get("fundamental_label", "—"),
    })

st.markdown("### 判定理由")
st.info(str(row.get("screen_reason", "—")))

st.caption(
    "この画面は『買い推奨』ではなく、今日確認すべき銘柄の優先順位です。"
    "上位候補はTradingView → Entry Hunter → Pre-Tradeの順で確認します。"
)

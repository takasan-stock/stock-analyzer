from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from sidebar_nav import render_sidebar_navigation
from site_context import get_active_stock, set_active_stock

from multiple_expansion_screener import build_daily_screener, candidate_only


st.set_page_config(
    page_title="ME Daily Screener",
    page_icon="🔥",
    layout="wide",
)
render_sidebar_navigation()

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


def _safe_num(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _safe_int(value):
    num = _safe_num(value)
    return None if num is None else int(num)


def _state_label(value: object) -> str:
    labels = {
        "EXP. DECELERATING": "減速中（再加速待ち）",
        "RE-WATCH EARLY": "再加速の初期監視",
        "RE-WATCH READY": "再加速準備ほぼ完了",
        "RE-EXP": "再拡大を確認",
        "OVERHEATED": "過熱気味",
        "UNAVAILABLE": "判定材料不足",
    }
    text = str(value) if value is not None else "—"
    return labels.get(text, text)


def _decision_label(value: object) -> str:
    labels = {
        "PRIORITY WATCH": "最優先で監視",
        "READY": "準備完了",
        "ACTIVE": "進行中",
        "WATCH": "監視",
        "FIRST WAVE": "第1波候補",
        "PASS": "見送り",
    }
    text = str(value) if value is not None else "—"
    return labels.get(text, text)


def _engine_label(value: object) -> str:
    labels = {
        "ARMED": "再加速待機中",
        "LOCKED": "再判定ロック中",
        "IDLE": "待機",
    }
    text = str(value) if value is not None else "—"
    return labels.get(text, text)


def _route_label(value: object) -> str:
    labels = {
        "OPEN": "判定可能",
        "RE-ARMED": "再判定可能",
        "LOCKED": "再判定ロック中",
    }
    text = str(value) if value is not None else "—"
    return labels.get(text, text)


def _fcf_label(value: object) -> str:
    text = str(value) if value is not None and not pd.isna(value) else ""
    if not text or text in {"nan", "UNKNOWN", "—"}:
        return "未判定"
    labels = {
        "A": "良好",
        "B": "やや良好",
        "C": "中立",
        "D": "弱い",
        "INSUFFICIENT": "データ不足",
    }
    return labels.get(text, text)


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
_active_ticker, _, _ = get_active_stock()
_active_index = options.index(_active_ticker) if _active_ticker in options else 0
selected = st.selectbox(
    "銘柄",
    options=options,
    index=_active_index,
    key="me_daily_detail_ticker",
)
row = view[view["ticker"].astype(str) == selected].iloc[0]

company_name = str(row.get("company_name", "") or "").strip()
set_active_stock(selected, company_name, source="ME Daily Screener")
if company_name:
    st.markdown(f"### {selected}｜{company_name}")

decision = _decision_label(row.get("sw_decision", "—"))
state = _state_label(row.get("second_wave_state", "—"))
st.markdown("#### 現在の判断")
j1, j2 = st.columns(2)
j1.info(f"**優先度：{decision}**")
j2.info(f"**状態：{state}**")

m1, m2, m3, m4 = st.columns(4)
sw_score = _safe_num(row.get("sw_score"))
mlp_core = _safe_num(row.get("mlp_c_core"))
m1.metric("SWスコア", "—" if sw_score is None else f"{sw_score:.0f}/100")
m2.metric("MLP-C", "—" if mlp_core is None else f"{mlp_core:.3f}")
m3.metric("速度", _pct_raw(row.get("me_velocity_pct"), 2))
m4.metric("加速度", _pct_raw(row.get("me_acceleration_pct"), 2))

st.caption(
    "SWスコア＝第2波候補としての総合点。MLP-Cは市場評価の水準、"
    "速度/加速度はマルチプルが再び上向き始めているかを見る指標です。"
)

left, right = st.columns(2)

with left:
    st.markdown("### 🚀 再加速の準備状況")
    engine = _engine_label(row.get("re_engine", "—"))
    route = _route_label(row.get("re_route", "—"))
    pullback = _pct(row.get("mlp_pullback"), 1)
    curvature = _pct_raw(row.get("me_curvature_pct"), 2)
    gate = _safe_int(row.get("advance_gate_progress"))

    r1, r2 = st.columns(2)
    r1.metric("再加速エンジン", engine)
    r2.metric("判定ルート", route)

    r3, r4 = st.columns(2)
    r3.metric("MLP押し戻し", pullback)
    r4.metric("曲率", curvature)

    if gate is None:
        st.caption("アドバンスゲート：未適用")
    else:
        remain = max(0, 120 - gate)
        if remain > 0:
            st.caption(f"アドバンスゲート：前回RE-EXPから {gate} 日経過／解除まで約 {remain} 日")
        else:
            st.caption(f"アドバンスゲート：{gate} 日経過・再判定可能")

    engine_raw = str(row.get("re_engine", ""))
    vel = _safe_num(row.get("me_velocity_pct"))
    acc = _safe_num(row.get("me_acceleration_pct"))
    if engine_raw == "ARMED":
        st.success("再加速の準備条件は整っています。速度と加速度の反転確認が次の焦点です。")
    elif engine_raw == "LOCKED":
        st.warning("再加速シグナルの連発を防ぐため、現在は再判定ロック中です。")
    elif vel is not None and vel < 0:
        st.info("現在は減速局面です。速度が0%方向へ戻り、加速度がプラスへ転じるかを確認します。")
    else:
        st.info("再加速の準備条件を監視中です。")

with right:
    st.markdown("### 📊 過去実績とFCF")
    hist_edge = _safe_num(row.get("hist_edge_score"))
    hist_n = _safe_int(row.get("hist_edge_n"))
    hist_win = _pct(row.get("hist_60d_win"), 1)
    hist_excess = _pct(row.get("hist_60d_excess"), 1)
    fcf_score = _safe_num(row.get("fcf_engine_score"))
    fcf_label = _fcf_label(row.get("fundamental_label", "—"))

    h1, h2 = st.columns(2)
    h1.metric("過去エッジ", "—" if hist_edge is None else f"{hist_edge:.0f}/100")
    h2.metric("検証サンプル数", "—" if hist_n is None else f"{hist_n}件")

    h3, h4 = st.columns(2)
    h3.metric("60日勝率", hist_win)
    h4.metric("60日超過収益", hist_excess)

    f1, f2 = st.columns(2)
    f1.metric("FCFエンジン", "未算出" if fcf_score is None else f"{fcf_score:.0f}/100")
    f2.metric("FCF評価", fcf_label)

    if hist_n is None or hist_n < 5:
        st.warning("過去実績のサンプル数が少ないため、勝率や超過収益は参考値として扱ってください。")
    elif hist_edge is not None and hist_edge >= 70:
        st.success("過去の類似局面では比較的良い成績が確認されています。")
    elif hist_edge is not None and hist_edge < 50:
        st.info("過去エッジは強くありません。テクニカル確認を優先してください。")

st.markdown("### ✅ 次に確認すること")
checks = []

vel = _safe_num(row.get("me_velocity_pct"))
acc = _safe_num(row.get("me_acceleration_pct"))
curv = _safe_num(row.get("me_curvature_pct"))
mlp = _safe_num(row.get("mlp_c_core"))

if str(row.get("second_wave_state", "")) == "EXP. DECELERATING":
    checks.append("速度が0%方向へ戻るか（減速終了の確認）")
if acc is not None and acc <= 0:
    checks.append("加速度がプラスへ転じるか")
if curv is not None and curv <= 0:
    checks.append("曲率がプラスへ転じ、上向きの形が出るか")
if mlp is not None and mlp > 1.20:
    checks.append("MLPが高すぎないか（過熱・追いかけ買いに注意）")
if _safe_num(row.get("fcf_engine_score")) is None:
    checks.append("FCFデータ未算出の理由を確認")
if _safe_int(row.get("hist_edge_n")) in (None, 0, 1, 2, 3, 4):
    checks.append("過去エッジはサンプル不足なので過信しない")

if not checks:
    checks.append("TradingViewで価格形状を確認し、Entry Hunter → Pre-Tradeへ進む")

for item in checks:
    st.write(f"・{item}")

reason = str(row.get("screen_reason", "—"))
st.markdown("### 判定の要約")
st.info(
    reason.replace("EXP. DECELERATING", "減速中")
          .replace("RE-WATCH READY", "再加速準備")
          .replace("RE-WATCH EARLY", "再加速初期")
          .replace("RE-EXP", "再拡大確認")
          .replace("LOW SAMPLE", "過去サンプル少")
          .replace("ACCEL+", "加速度プラス")
          .replace("OVERHEATED", "過熱")
          .replace("GATE LOCKED", "再判定ロック")
)

st.caption(
    "この画面は『買い推奨』ではなく、今日確認すべき銘柄の優先順位です。"
    "上位候補はTradingView → Entry Hunter → Pre-Tradeの順で確認します。"
)

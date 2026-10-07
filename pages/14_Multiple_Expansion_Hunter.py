from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Multiple Expansion Hunter",
    page_icon="🚀",
    layout="wide",
)

LATEST_FILE = "data/multiple_expansion/mex_latest.csv"
HISTORY_FILE = "data/multiple_expansion/mex_history.csv"

STATE_LABELS = {
    "IGNITION": "🚀 IGNITION",
    "SPRING": "🔵 SPRING",
    "PREP": "🟨 PREP",
    "EXPANSION": "🟢 EXPANSION",
    "MATURE": "🟩 MATURE",
    "EXHAUSTION": "⚠️ EXHAUSTION",
    "SPECULATIVE": "🟠 SPECULATIVE",
    "WATCH": "👀 WATCH",
    "DISCOUNT": "💎 DISCOUNT",
    "DE_RATING": "🔻 DE-RATING",
    "UNAVAILABLE": "⚪ UNAVAILABLE",
}


def _load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def _num(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt(value, digits=1, suffix=""):
    num = _num(value)
    if num is None:
        return "—"
    return f"{num:.{digits}f}{suffix}"


def _pct(value, digits=1):
    num = _num(value)
    if num is None:
        return "—"
    return f"{num * 100:.{digits}f}%"


st.title("🚀 Multiple Expansion Hunter v0.1")
st.caption(
    "FCFという企業価値のエンジンが強い銘柄の中から、"
    "市場評価（マルチプル）が低位から上向き始める局面を検出します。"
)

latest = _load_csv(LATEST_FILE)
history = _load_csv(HISTORY_FILE)

if latest.empty:
    st.info(
        "MEXの計算結果はまだありません。"
        "v0.1では point-in-time 財務データを必須とし、"
        "未来の決算情報を過去日に混ぜる近道は使いません。"
    )
    st.markdown(
        """
### 実装済みUIの入力ファイル
- `data/multiple_expansion/mex_latest.csv`
- `data/multiple_expansion/mex_history.csv`

計算パイプラインが上記CSVを生成すると、このページが自動的にランキングと詳細チャートを表示します。
"""
    )
    st.stop()

latest = latest.copy()
for col in [
    "mex_score",
    "fcf_engine_score",
    "mlp_c_core",
    "mlp_z",
    "mlp_velocity_pct_20",
    "mlp_acceleration",
    "normalization_gap",
    "mex_coverage",
]:
    if col in latest.columns:
        latest[col] = pd.to_numeric(
            latest[col],
            errors="coerce",
        )

if "state_confirmed" not in latest.columns:
    latest["state_confirmed"] = latest.get(
        "state_raw",
        "UNAVAILABLE",
    )

latest["state_label"] = (
    latest["state_confirmed"]
    .map(STATE_LABELS)
    .fillna(latest["state_confirmed"])
)

st.markdown("## 今日の再評価候補")

all_states = sorted(
    latest["state_confirmed"]
    .dropna()
    .astype(str)
    .unique()
)
selected_states = st.multiselect(
    "State",
    options=all_states,
    default=all_states,
)

f1, f2, f3, f4 = st.columns(4)
min_mex = f1.slider(
    "MEX 最低点",
    0,
    100,
    0,
    5,
)
min_fcf = f2.slider(
    "FCF Engine 最低点",
    0,
    100,
    0,
    5,
)
max_mlp = f3.number_input(
    "MLP 上限",
    min_value=0.0,
    value=2.0,
    step=0.05,
)
min_gap = f4.number_input(
    "Normalization Gap 最低",
    value=0.0,
    step=0.05,
)

view = latest.copy()
if selected_states:
    view = view[
        view["state_confirmed"]
        .astype(str)
        .isin(selected_states)
    ]
if "mex_score" in view.columns:
    view = view[
        view["mex_score"].fillna(-1) >= min_mex
    ]
if "fcf_engine_score" in view.columns:
    view = view[
        view["fcf_engine_score"].fillna(-1)
        >= min_fcf
    ]
if "mlp_c_core" in view.columns:
    view = view[
        view["mlp_c_core"].fillna(999) <= max_mlp
    ]
if "normalization_gap" in view.columns:
    view = view[
        view["normalization_gap"].fillna(-999)
        >= min_gap
    ]

sort_cols = [
    c
    for c in [
        "mex_score",
        "mex_coverage",
        "ticker",
    ]
    if c in view.columns
]
ascending = [
    False if c != "ticker" else True
    for c in sort_cols
]
if sort_cols:
    view = view.sort_values(
        sort_cols,
        ascending=ascending,
    )

display_cols = [
    c
    for c in [
        "ticker",
        "company_name",
        "mex_score",
        "fcf_engine_score",
        "mlp_c_core",
        "mlp_z",
        "mlp_velocity_pct_20",
        "mlp_acceleration",
        "normalization_gap",
        "state_label",
        "data_quality",
    ]
    if c in view.columns
]

display = view[display_cols].copy()
rename = {
    "ticker": "コード",
    "company_name": "銘柄",
    "mex_score": "MEX",
    "fcf_engine_score": "FCF Engine",
    "mlp_c_core": "MLP",
    "mlp_z": "Z",
    "mlp_velocity_pct_20": "Velocity",
    "mlp_acceleration": "Acceleration",
    "normalization_gap": "Gap",
    "state_label": "State",
    "data_quality": "Data",
}
display = display.rename(columns=rename)

for col in [
    "MEX",
    "FCF Engine",
    "MLP",
    "Z",
    "Acceleration",
]:
    if col in display.columns:
        display[col] = pd.to_numeric(
            display[col],
            errors="coerce",
        ).round(2)

for col in ["Velocity", "Gap"]:
    if col in display.columns:
        display[col] = display[col].map(
            lambda x: (
                "—"
                if pd.isna(x)
                else f"{float(x) * 100:+.1f}%"
            )
        )

st.dataframe(
    display,
    hide_index=True,
    use_container_width=True,
)

if view.empty:
    st.warning(
        "現在のフィルター条件に該当する銘柄はありません。"
    )
    st.stop()

st.divider()
st.markdown("## 銘柄詳細")

options = view["ticker"].astype(str).tolist()
selected_ticker = st.selectbox(
    "銘柄",
    options=options,
)
row = view[
    view["ticker"].astype(str) == selected_ticker
].iloc[0]

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric(
    "MEX",
    _fmt(row.get("mex_score"), 1),
)
c2.metric(
    "State",
    STATE_LABELS.get(
        str(row.get("state_confirmed")),
        str(row.get("state_confirmed")),
    ),
)
c3.metric(
    "FCF Engine",
    _fmt(row.get("fcf_engine_score"), 1),
)
c4.metric(
    "MLP",
    _fmt(row.get("mlp_c_core"), 3),
)
c5.metric(
    "MLP Z",
    _fmt(row.get("mlp_z"), 2),
)
c6.metric(
    "Gap",
    _pct(row.get("normalization_gap"), 1),
)

st.markdown("### 着火条件")
checks = {
    "FCF Engine ≥ 70": (
        _num(row.get("fcf_engine_score")) or -999
    ) >= 70,
    "0.80 ≤ MLP ≤ 1.05": (
        0.80
        <= (_num(row.get("mlp_c_core")) or -999)
        <= 1.05
    ),
    "Velocity > 0": (
        _num(row.get("mlp_velocity_20")) or -999
    ) > 0,
    "Acceleration > 0": (
        _num(row.get("mlp_acceleration")) or -999
    ) > 0,
    "市場確認 2個以上": (
        _num(row.get("market_confirm_count")) or 0
    ) >= 2,
}
st.write(
    "　".join(
        ("✅ " if ok else "⬜ ") + label
        for label, ok in checks.items()
    )
)

if not history.empty and "ticker" in history.columns:
    hist = history[
        history["ticker"].astype(str) == selected_ticker
    ].copy()
    if not hist.empty and "trade_date" in hist.columns:
        hist["trade_date"] = pd.to_datetime(
            hist["trade_date"],
            errors="coerce",
        )
        hist = hist.sort_values("trade_date")

        st.markdown("### 株価とMLP")
        fig = go.Figure()

        if "adj_close" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist["trade_date"],
                    y=pd.to_numeric(
                        hist["adj_close"],
                        errors="coerce",
                    ),
                    name="株価",
                    yaxis="y",
                )
            )

        if "mlp_c_core" in hist.columns:
            fig.add_trace(
                go.Scatter(
                    x=hist["trade_date"],
                    y=pd.to_numeric(
                        hist["mlp_c_core"],
                        errors="coerce",
                    ),
                    name="MLP Core",
                    yaxis="y2",
                )
            )

        fig.add_shape(
            type="line",
            x0=hist["trade_date"].min(),
            x1=hist["trade_date"].max(),
            y0=1.0,
            y1=1.0,
            xref="x",
            yref="y2",
            line=dict(dash="dot"),
        )

        fig.update_layout(
            height=480,
            yaxis=dict(title="株価"),
            yaxis2=dict(
                title="MLP",
                overlaying="y",
                side="right",
            ),
            legend=dict(orientation="h"),
            margin=dict(
                l=20,
                r=20,
                t=20,
                b=20,
            ),
        )
        st.plotly_chart(
            fig,
            use_container_width=True,
        )

st.markdown("### FCF / Multiple スナップショット")
left, right = st.columns(2)

with left:
    st.write({
        "FCF Quality": row.get(
            "fcf_quality_grade",
            "—",
        ),
        "FCF Engine Coverage": _pct(
            row.get("fcf_engine_coverage"),
            0,
        ),
        "MEX Coverage": _pct(
            row.get("mex_coverage"),
            0,
        ),
        "Data Quality": row.get(
            "data_quality",
            "—",
        ),
    })

with right:
    st.write({
        "Velocity": _pct(
            row.get("mlp_velocity_pct_20"),
            1,
        ),
        "Acceleration": _fmt(
            row.get("mlp_acceleration"),
            3,
        ),
        "Normalization Gap": _pct(
            row.get("normalization_gap"),
            1,
        ),
        "Market Confirm": (
            f"{int(_num(row.get('market_confirm_count')) or 0)}/4"
        ),
    })

st.caption(
    "MEXは買い推奨ではなく『再評価開始を調査する優先順位』です。"
    "実売買はEntry Hunter / Pre-Tradeと組み合わせて確認します。"
)

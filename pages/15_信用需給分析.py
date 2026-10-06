from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from credit_supply import analyze_credit_supply, load_jpx_margin_history, normalize_ticker

st.set_page_config(page_title="信用需給分析", page_icon="⚖️", layout="wide")

st.title("⚖️ 信用需給分析")
st.caption(
    "JPXの銘柄別信用取引残高と株価・出来高を組み合わせ、信用買いの重さ・信用倍率・需給の方向を確認します。"
    " 機械判定は売買推奨ではなく、需給確認の補助です。"
)


@st.cache_data(ttl=1800, show_spinner=False)
def load_margin_cached(max_files: int):
    return load_jpx_margin_history(max_files=max_files)


@st.cache_data(ttl=900, show_spinner=False)
def load_price_cached(ticker: str) -> pd.DataFrame:
    symbol = ticker if ticker.endswith(".T") else f"{ticker}.T"
    try:
        df = yf.download(
            symbol,
            period="3mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.sort_index()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def load_shares_cached(ticker: str) -> float | None:
    symbol = ticker if ticker.endswith(".T") else f"{ticker}.T"
    try:
        tk = yf.Ticker(symbol)
        try:
            shares = tk.get_shares_full()
            if shares is not None and len(shares):
                value = pd.to_numeric(shares, errors="coerce").dropna()
                if not value.empty:
                    return float(value.iloc[-1])
        except Exception:
            pass
        try:
            info = tk.info or {}
            value = info.get("sharesOutstanding")
            if value:
                return float(value)
        except Exception:
            pass
    except Exception:
        pass
    return None


def fmt_int(value) -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.0f}"
    except Exception:
        return "—"


def fmt_float(value, digits=2, suffix="") -> str:
    try:
        if value is None or pd.isna(value):
            return "—"
        return f"{float(value):,.{digits}f}{suffix}"
    except Exception:
        return "—"


def fmt_delta(value, suffix="株") -> str | None:
    try:
        if value is None or pd.isna(value):
            return None
        return f"{float(value):+,.0f}{suffix}"
    except Exception:
        return None


def status_label(status: str) -> str:
    return {
        "🟢 改善優位": "信用買残の整理や倍率低下が進み、需給は改善方向。",
        "🟢 やや改善": "改善材料が優勢。ただし一部の確認項目はまだ中立。",
        "🟡 中立": "強い改善・悪化のどちらにも傾いていない状態。",
        "🟠 悪化注意": "信用買残増加や倍率上昇など、需給悪化の材料が優勢。",
        "NO DATA": "判定に必要なデータが不足。",
    }.get(status, "")


portfolio_choices: list[str] = []
if os.path.exists("portfolio_data.csv"):
    try:
        portfolio = pd.read_csv("portfolio_data.csv")
        for col in ("ティッカー", "ticker", "Ticker"):
            if col in portfolio.columns:
                portfolio_choices = [
                    normalize_ticker(x)
                    for x in portfolio[col].dropna().tolist()
                    if normalize_ticker(x)
                ]
                break
    except Exception:
        portfolio_choices = []

left, right = st.columns([2, 1])
with left:
    ticker_input = st.text_input(
        "証券コード",
        value=portfolio_choices[0] if portfolio_choices else "4063",
        help="4桁コードまたは英数字コード。例: 4063 / 472A",
    )
with right:
    max_files = st.selectbox(
        "取得するJPXファイル数",
        [8, 16, 24, 40],
        index=1,
        help="日次化直後のため、当面は旧週次ファイルが混在する場合があります。",
    )

ticker = normalize_ticker(ticker_input)
if not ticker:
    st.warning("証券コードを入力してください。")
    st.stop()

with st.spinner("JPX信用残と株価データを取得中..."):
    margin_result = load_margin_cached(int(max_files))
    prices = load_price_cached(ticker)
    shares_outstanding = load_shares_cached(ticker)

if margin_result.balances.empty:
    st.error("JPX信用残データを取得できませんでした。")
    if margin_result.errors:
        with st.expander("取得エラー"):
            for err in margin_result.errors[:10]:
                st.write(f"- {err}")
    st.stop()

latest_close = previous_close = avg_volume_5 = avg_volume_25 = None
if prices is not None and not prices.empty:
    close = pd.to_numeric(prices.get("Close"), errors="coerce").dropna()
    volume = pd.to_numeric(prices.get("Volume"), errors="coerce").dropna()
    if len(close) >= 1:
        latest_close = float(close.iloc[-1])
    if len(close) >= 2:
        previous_close = float(close.iloc[-2])
    if len(volume) >= 1:
        avg_volume_5 = float(volume.tail(5).mean())
        avg_volume_25 = float(volume.tail(25).mean())

result = analyze_credit_supply(
    margin_result.balances,
    ticker,
    latest_close=latest_close,
    previous_close=previous_close,
    avg_volume_5=avg_volume_5,
    avg_volume_25=avg_volume_25,
    shares_outstanding=shares_outstanding,
)

if result.get("status") == "NO DATA":
    st.warning(
        f"{ticker} の信用残データが見つかりません。"
        " JPXファイルの掲載対象・コード形式・ページ更新時刻をご確認ください。"
    )
    with st.expander("JPX取得状況"):
        st.write(
            f"発見ファイル: {margin_result.files_found} / 読込成功: {margin_result.files_loaded}"
        )
        for err in margin_result.errors[:10]:
            st.write(f"- {err}")
    st.stop()

name = result.get("name") or ""
as_of = result.get("date")
as_of_text = pd.Timestamp(as_of).strftime("%Y/%m/%d") if as_of is not None else "—"

st.subheader(f"{ticker} {name}")
st.caption(
    f"JPX信用残基準日: {as_of_text} ｜ "
    f"JPXファイル読込 {margin_result.files_loaded}/{margin_result.files_found}"
)

score = result.get("score")
status = result.get("status", "NO DATA")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("信用買残", f"{fmt_int(result.get('long_balance'))} 株", fmt_delta(result.get("long_delta")), delta_color="inverse")
m2.metric("信用売残", f"{fmt_int(result.get('short_balance'))} 株", fmt_delta(result.get("short_delta")))
m3.metric(
    "信用倍率",
    fmt_float(result.get("credit_ratio"), 2, "倍"),
    fmt_float(result.get("credit_ratio_delta"), 2, "倍") if result.get("credit_ratio_delta") is not None else None,
    delta_color="inverse",
)
m4.metric("買残 / 25日平均出来高", fmt_float(result.get("volume_days_25"), 1, "日分"))
m5.metric("信用需給スコア", f"{fmt_float(score, 1)} / 100" if score is not None else "—", status)

st.info(
    f"**機械判定: {status}** — {status_label(status)}"
    f"　判定カバレッジ {fmt_float(result.get('coverage'), 0, '%')}"
)

history = result.get("history", pd.DataFrame()).copy()

c1, c2 = st.columns([1.5, 1])
with c1:
    st.markdown("### 信用残推移")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=history["date"], y=history["long_balance"], mode="lines+markers", name="信用買残"))
    fig.add_trace(go.Scatter(x=history["date"], y=history["short_balance"], mode="lines+markers", name="信用売残"))
    fig.update_layout(height=370, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="株", legend_title="", hovermode="x unified")
    st.plotly_chart(fig, use_container_width=True)

with c2:
    st.markdown("### 信用倍率推移")
    ratio_history = history.dropna(subset=["credit_ratio"])
    fig_ratio = go.Figure()
    fig_ratio.add_trace(go.Scatter(x=ratio_history["date"], y=ratio_history["credit_ratio"], mode="lines+markers", name="信用倍率"))
    fig_ratio.update_layout(height=370, margin=dict(l=20, r=20, t=20, b=20), yaxis_title="倍", showlegend=False, hovermode="x unified")
    st.plotly_chart(fig_ratio, use_container_width=True)

st.markdown("### 今回の読み")
good, risk = st.columns(2)
with good:
    st.markdown("**🟢 改善材料**")
    reasons = result.get("reasons") or []
    if reasons:
        for item in reasons:
            st.write(f"- {item}")
    else:
        st.write("- 明確な改善材料はまだありません")

with risk:
    st.markdown("**🟠 注意材料**")
    risks = result.get("risks") or []
    if risks:
        for item in risks:
            st.write(f"- {item}")
    else:
        st.write("- 大きな注意材料は検出していません")

st.markdown("### 株価 × 信用買残の読み方")
price_change = result.get("price_change_pct")
long_delta = result.get("long_delta")
if price_change is not None and long_delta is not None:
    if price_change > 0 and long_delta < 0:
        matrix_state = "🟢 株価↑ ＋ 信用買残↓：需給改善型"
        matrix_note = "株価が上がりながら信用買いが整理されています。"
    elif price_change > 0 and long_delta >= 0:
        matrix_state = "🟡 株価↑ ＋ 信用買残↑：信用買い主導型"
        matrix_note = "上昇は強い一方、将来の戻り売り圧力が増えていないか確認します。"
    elif price_change <= 0 and long_delta > 0:
        matrix_state = "🟠 株価↓ ＋ 信用買残↑：捕まり玉増加型"
        matrix_note = "含み損の信用買いが増えやすく、戻り売り圧力に注意します。"
    else:
        matrix_state = "🟡 株価↓ ＋ 信用買残↓：整理・投げ型"
        matrix_note = "需給整理が進行中。下げ止まりと買残減少の継続を確認します。"
    st.write(f"**{matrix_state}**")
    st.caption(f"直近株価変化 {price_change:+.2f}% ｜ 信用買残変化 {long_delta:+,.0f}株。{matrix_note}")
else:
    st.caption("株価または前回信用残が不足しているため、マトリクス判定は保留です。")

d1, d2, d3, d4 = st.columns(4)
d1.metric("5日平均出来高換算", fmt_float(result.get("volume_days_5"), 1, "日分"))
d2.metric("25日平均出来高換算", fmt_float(result.get("volume_days_25"), 1, "日分"))
d3.metric("買残 / 発行済株式", fmt_float(result.get("long_pct_shares"), 3, "%"))
d4.metric("売残 / 発行済株式", fmt_float(result.get("short_pct_shares"), 3, "%"))

st.markdown("### 信用残データ")
show_cols = [c for c in ["date", "long_balance", "short_balance", "credit_ratio", "long_general", "long_standard", "short_general", "short_standard"] if c in history.columns]
display = history[show_cols].sort_values("date", ascending=False).copy()
display = display.rename(columns={
    "date": "基準日",
    "long_balance": "買残",
    "short_balance": "売残",
    "credit_ratio": "信用倍率",
    "long_general": "一般買残",
    "long_standard": "制度買残",
    "short_general": "一般売残",
    "short_standard": "制度売残",
})
st.dataframe(display, use_container_width=True, hide_index=True)

with st.expander("📘 銘柄が変わっても使える『信用需給ひな型』"):
    st.markdown(
        """
**① 信用買残**
- 減少 → 将来の売り圧力が整理される方向
- 増加 → 買い長化に注意

**② 信用売残**
- 増加 → 将来の買い戻し余地は増えるが、株価下落中なら売り方優勢の可能性も確認
- 減少 → 買い戻しが進んだ可能性

**③ 信用倍率**
- 数字の絶対値だけでなく、前回より上がったか下がったかを見る
- 低下 → 需給改善方向
- 上昇 → 買い長化方向

**④ 株価と信用買残の組み合わせ**
- 株価↑ ＋ 買残↓ → 🟢 需給改善型
- 株価↑ ＋ 買残↑ → 🟡 信用買い主導型
- 株価↓ ＋ 買残↑ → 🟠 捕まり玉増加型
- 株価↓ ＋ 買残↓ → 🟡 整理・投げ型

**⑤ 買残 / 平均出来高**
- 0.5日未満 → 非常に軽い
- 0.5〜1日 → 軽い
- 1〜2日 → 普通
- 2〜3日 → やや重い
- 3日超 → 重い

**⑥ 最後に見ること**
1日だけで決めず、数営業日の方向が続くかを確認します。
"""
    )

with st.expander("データについて"):
    st.write(
        "・信用買残/売残: JPX『銘柄別信用取引残高』を自動取得。"
        " 2026年9月28日以降は日次公表が始まったため、今後データ点が増えるほど日次トレンドが見やすくなります。"
    )
    st.write(
        "・株価/出来高/発行済株式: yfinance。取得不能時は該当項目を欠損扱いにし、"
        "利用可能な項目だけでスコアを再計算します。"
    )
    st.write(
        "・画像にあった『当日の現物買越し』『信用新規/返済の売買内訳』は、"
        "JPXの銘柄別残高だけでは同じ粒度で再現できないため、v1では推測値を表示しません。"
    )

if margin_result.errors:
    with st.expander("取得ログ / 注意"):
        for err in margin_result.errors[:15]:
            st.write(f"- {err}")

st.caption(
    "次段階では Short Cover Hunter の Cover Score / Phase / Entry Hunter と接続し、"
    "『空売り買い戻し + 信用需給改善 + エントリー条件』の一致を1画面で確認できるようにします。"
)

from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from short_cover import (
    build_price_feature_table,
    build_short_metrics,
    candidate_tickers,
    load_jpx_events,
    load_uploaded_workbooks,
    normalize_ticker,
    score_short_cover,
)

st.set_page_config(page_title="Short Cover Hunter", page_icon="🔥", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_jpx_cached(archive_pages: int, max_files: int):
    return load_jpx_events(archive_pages=archive_pages, max_files=max_files)


@st.cache_data(ttl=900, show_spinner=False)
def price_features_cached(tickers: tuple[str, ...]) -> pd.DataFrame:
    return build_price_feature_table(list(tickers))


@st.cache_data(ttl=900, show_spinner=False)
def load_detail_price(ticker: str) -> pd.DataFrame:
    symbol = ticker if "." in ticker else f"{ticker}.T"
    try:
        data = yf.download(symbol, period="6mo", interval="1d", auto_adjust=True, progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        return data.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def load_watchlist_codes() -> list[str]:
    path = "portfolio_data.csv"
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, dtype=str)
        if "ティッカー" not in df.columns:
            return []
        return [normalize_ticker(x) for x in df["ティッカー"].dropna().tolist() if normalize_ticker(x)]
    except Exception:
        return []



def _truthy(value) -> bool:
    try:
        return bool(value) if pd.notna(value) else False
    except Exception:
        return False

def fmt_num(v, digits=1, suffix=""):
    if v is None or pd.isna(v):
        return "—"
    return f"{float(v):,.{digits}f}{suffix}"


st.title("🔥 Short Cover Hunter v1")
st.caption(
    "大口空売りの『買い戻し初動』を、JPX公表残高イベント＋価格・出来高・AVWAP代理値・RSで推定します。"
)

with st.expander("このツールの読み方", expanded=False):
    st.markdown(
        """
**狙う順番は「燃料 → 吸収 → 点火 → 確認」です。**

- **Short Pressure**：0.5%以上で公表された大口空売りの蓄積度
- **Absorption**：空売り圧力があるのに株価が崩れにくくなった兆候
- **Cover Score**：出来高、AVWAP回復、5日高値突破、RSなどを統合した初動スコア
- **Long Demand**：単なる買い戻しではなく、新規買い資金も入っていそうかを見る補助スコア

`🔥 COVER EARLY` は「買い戻しの可能性が高まった候補」であり、空売り主体の実注文を直接識別したものではありません。
        """
    )

with st.sidebar:
    st.markdown("### ⚙️ Short Cover設定")
    archive_pages = st.slider("JPX履歴（月ページ）", 1, 4, 2, help="現在ページ＋過去月ページ。2でおおむね2〜3か月分を確認します。")
    max_files = st.slider("読み込むJPXファイル上限", 25, 100, 70, step=5)
    candidate_limit = st.slider("JPX候補の価格分析数", 20, 100, 60, step=10)
    universe = st.radio("分析対象", ["JPX候補＋登録銘柄", "登録銘柄のみ", "JPX候補のみ"], index=0)
    min_score = st.slider("ランキング最低Cover Score", 0, 90, 45, step=5)

    st.divider()
    if st.button("🔄 データを再取得", use_container_width=True):
        load_jpx_cached.clear()
        price_features_cached.clear()
        load_detail_price.clear()
        st.rerun()

watchlist = load_watchlist_codes()

with st.spinner("JPX空売り残高データを取得・解析中..."):
    jpx = load_jpx_cached(archive_pages, max_files)

events = jpx.events.copy()

if events.empty:
    st.warning("JPXデータを自動取得できませんでした。下からJPXのExcelファイルを複数アップロードして続行できます。")
    uploads = st.file_uploader(
        "JPX『空売り残高に関する情報』Excelをアップロード",
        type=["xlsx", "xls"],
        accept_multiple_files=True,
    )
    if uploads:
        events = load_uploaded_workbooks(uploads)

if events.empty:
    st.error("解析できるJPX空売り残高データがありません。")
    if jpx.errors:
        with st.expander("取得エラー詳細"):
            st.code("\n".join(jpx.errors[:20]))
    st.stop()

short_metrics = build_short_metrics(events)

jpx_candidates = candidate_tickers(short_metrics, limit=candidate_limit)
if universe == "登録銘柄のみ":
    targets = watchlist
elif universe == "JPX候補のみ":
    targets = jpx_candidates
else:
    targets = list(dict.fromkeys(jpx_candidates + watchlist))

targets = [t for t in targets if t]
if not targets:
    st.info("分析対象の銘柄がありません。登録銘柄を追加するか、JPX候補を含む設定にしてください。")
    st.stop()

with st.spinner(f"価格・出来高を分析中... {len(targets)}銘柄"):
    prices = price_features_cached(tuple(sorted(set(targets))))

scored = score_short_cover(short_metrics[short_metrics["ticker"].isin(targets)].copy(), prices)
if scored.empty:
    st.info("対象銘柄に、今回読み込んだJPX空売り報告イベントがありませんでした。")
    st.stop()

latest_date = events["calc_date"].max()
col1, col2, col3, col4 = st.columns(4)
col1.metric("JPXファイル", f"{jpx.files_loaded}/{jpx.files_found}")
col2.metric("空売り報告イベント", f"{len(events):,}")
col3.metric("分析候補", f"{len(scored)}銘柄")
col4.metric("最新計算日", latest_date.strftime("%Y/%m/%d") if pd.notna(latest_date) else "—")

st.info(
    "JPXの日次ファイルは『その日に報告されたイベント』で、全銘柄・全機関の完全な日次スナップショットではありません。"
    "v1は読み込んだ期間内の最新報告をつないで状態を再構成するため、長期間更新のないポジションは過小評価される場合があります。",
    icon="ℹ️",
)

filtered = scored[scored["cover_score"] >= min_score].copy()
filtered.insert(0, "順位", range(1, len(filtered) + 1))

st.markdown("## 🏹 買い戻し初動ランキング")
if filtered.empty:
    st.warning("現在の閾値を超える候補はありません。最低Cover Scoreを下げると候補を広げられます。")
else:
    display = filtered.copy()
    display["空売り%"] = display["short_ratio"].map(lambda x: fmt_num(x, 2, "%"))
    display["Δ空売り"] = display["delta_short"].map(lambda x: fmt_num(x, 2, "pt"))
    display["買戻Breadth"] = display["cover_breadth"].map(lambda x: fmt_num(x, 0, "%"))
    display["DTC"] = display["dtc"].map(lambda x: fmt_num(x, 1, "日"))
    display["出来高"] = display["vol_ratio"].map(lambda x: fmt_num(x, 2, "x"))
    display["AVWAP"] = display["above_avwap"].map(lambda x: "✅上" if _truthy(x) else "—")
    display["5日高値"] = display["breakout5"].map(lambda x: "✅突破" if _truthy(x) else "—")
    display["RS"] = display["rs_watch"].map(lambda x: fmt_num(x, 0))
    display["信頼度"] = display["confidence"].map(lambda x: f"{int(x)}")

    cols = [
        "順位", "ticker", "name", "phase", "cover_score", "long_demand_score", "short_pressure",
        "空売り%", "Δ空売り", "institution_count", "買戻Breadth", "DTC", "出来高", "AVWAP", "5日高値", "RS", "信頼度",
    ]
    labels = {
        "ticker": "コード", "name": "銘柄", "phase": "Phase", "cover_score": "Cover",
        "long_demand_score": "Long", "short_pressure": "Pressure", "institution_count": "機関数",
    }
    table = display[cols].rename(columns=labels)
    st.dataframe(table, hide_index=True, use_container_width=True, height=min(760, 80 + 35 * len(table)))

    strong = filtered[filtered["phase"].isin(["🔥 COVER EARLY", "✅ COVER CONFIRMED", "🚀 SQUEEZE"])]
    if not strong.empty:
        st.markdown("### 🔥 今見る候補")
        for _, r in strong.head(5).iterrows():
            st.markdown(
                f"**{r['phase']}｜{r['name']}（{r['ticker']}）**　"
                f"Cover **{r['cover_score']:.0f}** / Long **{r['long_demand_score']:.0f}** / "
                f"Pressure **{r['short_pressure']:.0f}**　"
                f"出来高 {fmt_num(r.get('vol_ratio'), 2, 'x')}　RS {fmt_num(r.get('rs_watch'), 0)}"
            )

st.divider()
st.markdown("## 🔎 個別銘柄ドリルダウン")
choices_df = scored.copy()
choices_df["label"] = choices_df.apply(
    lambda r: f"{r['ticker']} {r['name']}｜{r['phase']}｜Cover {r['cover_score']:.0f}", axis=1
)
selected_label = st.selectbox("銘柄を選択", choices_df["label"].tolist())
selected = choices_df[choices_df["label"] == selected_label].iloc[0]
ticker = selected["ticker"]

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Cover Score", f"{selected['cover_score']:.0f}")
m2.metric("Long Demand", f"{selected['long_demand_score']:.0f}")
m3.metric("Short Pressure", f"{selected['short_pressure']:.0f}")
m4.metric("公表空売り", fmt_num(selected.get("short_ratio"), 2, "%"))
m5.metric("DTC", fmt_num(selected.get("dtc"), 1, "日"))

st.markdown(f"### {selected['phase']}　{selected['name']}（{ticker}）")

price_df = load_detail_price(ticker)
left, right = st.columns([1.45, 1])
with left:
    if not price_df.empty:
        p = price_df.tail(100).copy()
        tail20 = p.tail(20)
        anchor = tail20["Low"].idxmin() if "Low" in tail20 else tail20["Close"].idxmin()
        anchored = p.loc[anchor:].copy()
        if "Volume" in anchored and anchored["Volume"].sum() > 0:
            typical = (anchored["High"] + anchored["Low"] + anchored["Close"]) / 3
            anchored["AVWAP_proxy"] = (typical * anchored["Volume"]).cumsum() / anchored["Volume"].cumsum()
        else:
            anchored["AVWAP_proxy"] = pd.NA

        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=p.index, open=p["Open"], high=p["High"], low=p["Low"], close=p["Close"], name="Price"
        ))
        if anchored["AVWAP_proxy"].notna().any():
            fig.add_trace(go.Scatter(x=anchored.index, y=anchored["AVWAP_proxy"], mode="lines", name="AVWAP proxy"))
        fig.update_layout(height=440, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("AVWAPは日足のTypical Price×出来高から計算した代理値。約定ベースの真のVWAPではありません。")
    else:
        st.warning("価格チャートを取得できませんでした。")

with right:
    st.markdown("#### 判定内訳")
    detail = pd.DataFrame([
        ["Cover Score", selected.get("cover_score")],
        ["Long Demand", selected.get("long_demand_score")],
        ["Short Pressure", selected.get("short_pressure")],
        ["Absorption", selected.get("absorption_score")],
        ["買戻Breadth", selected.get("cover_breadth")],
        ["Δ空売り(pt)", selected.get("delta_short")],
        ["機関数", selected.get("institution_count")],
        ["出来高倍率", selected.get("vol_ratio")],
        ["RS（分析対象内）", selected.get("rs_watch")],
        ["5日高値突破", "YES" if _truthy(selected.get("breakout5")) else "NO"],
        ["AVWAP上", "YES" if _truthy(selected.get("above_avwap")) else "NO"],
        ["20日安値ダマシ", "YES" if _truthy(selected.get("failed_breakdown")) else "NO"],
        ["データ信頼度", selected.get("confidence")],
    ], columns=["項目", "値"])
    st.dataframe(detail, hide_index=True, use_container_width=True)

hist = events[events["ticker"] == ticker].copy().sort_values("calc_date")
if not hist.empty:
    st.markdown("#### 🏦 機関別・空売り報告履歴")
    chart_hist = hist.dropna(subset=["calc_date", "short_ratio"]).copy()
    if not chart_hist.empty:
        fig2 = px.line(chart_hist, x="calc_date", y="short_ratio", color="seller", markers=True,
                       labels={"calc_date": "計算日", "short_ratio": "空売り残高割合(%)", "seller": "空売り主体"})
        fig2.update_layout(height=390, margin=dict(l=10, r=10, t=35, b=10))
        st.plotly_chart(fig2, use_container_width=True)
    hist_table = hist[["calc_date", "seller", "short_ratio", "short_shares"]].copy()
    hist_table["calc_date"] = hist_table["calc_date"].dt.strftime("%Y-%m-%d")
    hist_table = hist_table.rename(columns={"calc_date": "計算日", "seller": "空売り主体", "short_ratio": "残高割合%", "short_shares": "残高株数"})
    st.dataframe(hist_table.sort_values("計算日", ascending=False), hide_index=True, use_container_width=True)

with st.expander("🧮 v1 スコア設計"):
    st.markdown(
        """
### Cover Score（0〜100）

- Short Pressure **25%**
- Absorption（下がらなくなる兆候） **20%**
- 出来高モメンタム **15%**
- AVWAP回復 **15%**
- 5日高値・短期反転 **10%**
- RS（今回の分析対象内の相対順位） **10%**
- 空売り減少・買戻Breadth確認 **5%**

初動を取りにいくため、**空売り残高の減少確認は5%の確認ボーナスに抑えています。**
残高減少を重くしすぎると「確認できた頃には株価がかなり上がっている」ためです。

### Phase

- `🧱 SHORT BUILDUP`：燃料はあるが点火前
- `👀 COVER WATCH`：下げ止まり・AVWAP回復などが出始めた
- `🔥 COVER EARLY`：買い戻し初動候補
- `✅ COVER CONFIRMED`：機関別残高減少も確認
- `🚀 SQUEEZE`：出来高＋高値突破まで伴う踏み上げ候補
        """
    )

st.caption(
    "注意：JPX公表対象は原則0.5%以上の空売り残高等です。0.5%未満のポジションや未報告分は含まれず、"
    "DTCも『公表された大口残高株数÷20日平均出来高』の下限寄り代理値です。売買推奨ではなく需給分析用です。"
)

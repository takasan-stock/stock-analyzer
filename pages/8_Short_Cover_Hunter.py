from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yfinance as yf

from short_cover import (
    backtest_short_cover,
    build_price_feature_snapshots,
    build_priority_alerts,
    build_promotion_table,
    build_short_metrics,
    candidate_tickers,
    load_jpx_events,
    load_uploaded_workbooks,
    normalize_ticker,
    apply_optimizer_condition,
    optimize_short_cover_thresholds,
    score_short_cover,
    summarize_backtest,
)

st.set_page_config(page_title="Short Cover Hunter", page_icon="🔥", layout="wide")


@st.cache_data(ttl=3600, show_spinner=False)
def load_jpx_cached(archive_pages: int, max_files: int):
    return load_jpx_events(archive_pages=archive_pages, max_files=max_files)


@st.cache_data(ttl=900, show_spinner=False)
def price_features_cached(tickers: tuple[str, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    return build_price_feature_snapshots(list(tickers))


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
- **Ignition**：出来高急増・AVWAP回復・短期高値突破・RS加速から「点火」を見る補助スコア
- **Long Demand**：単なる買い戻しではなく、新規買い資金も入っていそうかを見る補助スコア
- **資金フロー**：COVER + NEW MONEY / PURE SHORT COVER / NEW MONEY / NEUTRAL の4分類

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

    st.markdown("### 🧪 バックテスト")
    bt_sessions = st.slider("検証する直近営業日数", 20, 80, 40, step=10)
    bt_tickers = st.slider("バックテスト銘柄上限", 10, 60, 30, step=10)
    bt_min_score = st.slider("検証最低Cover Score", 45, 80, 55, step=5)
    opt_horizon = st.selectbox("最適化の評価期間", [3, 5, 10], index=1, format_func=lambda x: f"{x}営業日")
    opt_train_fraction = st.slider("学習期間の割合", 0.55, 0.80, 0.65, step=0.05)

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
    prices, prev_prices = price_features_cached(tuple(sorted(set(targets))))

_target_metrics = short_metrics[short_metrics["ticker"].isin(targets)].copy()
scored = score_short_cover(_target_metrics, prices)
prev_scored = score_short_cover(_target_metrics, prev_prices)
promotions = build_promotion_table(scored, prev_scored)
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

optimizer_live = st.session_state.get("short_cover_optimizer")
active_condition = None
if optimizer_live is not None and not optimizer_live.empty:
    preferred = optimizer_live[
        optimizer_live["robustness"].isin(["🟢 ROBUST", "🟡 PROMISING"])
    ]
    if not preferred.empty:
        active_condition = preferred.iloc[0]

scored = apply_optimizer_condition(scored, active_condition)
promotions = promotions.merge(
    scored[["ticker", "optimizer_match", "optimizer_label", "match_strength", "condition_text"]],
    on="ticker",
    how="left",
) if not promotions.empty else promotions

priority_alerts = build_priority_alerts(
    scored,
    promotions=promotions,
    limit=5,
)

st.markdown("## 🚨 今日の最優先チェック")
st.caption(
    "過去検証との一致・今日の昇格・Phase・資金フロー・信頼度・出来高を統合した確認優先度です。"
)
if priority_alerts.empty:
    st.info("現在、優先表示できる候補はありません。")
else:
    _priority = priority_alerts.copy()
    _pcols = st.columns(min(5, len(_priority)))
    for _idx, (_, _r) in enumerate(_priority.iterrows()):
        with _pcols[_idx]:
            st.metric(
                label=f"{_r['alert_tier']}",
                value=f"{_r['alert_score']:.0f}",
                delta=f"{_r['ticker']} {_r['name']}",
            )
            st.caption(
                f"{_r['alert_reason']}  \\n"
                f"Cover {_r['cover_score']:.0f}｜Ignition {_r['ignition_score']:.0f}｜"
                f"Long {_r['long_demand_score']:.0f}"
            )

    _priority_show = _priority.copy()
    _priority_show["優先度"] = _priority_show["alert_score"].map(lambda x: f"{float(x):.0f}")
    _priority_show["出来高"] = _priority_show["vol_ratio"].map(lambda x: fmt_num(x, 2, "x"))
    _priority_show["検証条件"] = _priority_show["optimizer_label"].replace("", "—")
    _priority_show["今日昇格"] = _priority_show["is_promotion"].map(lambda x: "⚡" if bool(x) else "—")
    _priority_cols = [
        "alert_tier", "ticker", "name", "優先度", "検証条件", "今日昇格",
        "phase", "regime", "cover_score", "ignition_score",
        "long_demand_score", "short_pressure", "出来高", "confidence",
        "alert_reason",
    ]
    _priority_labels = {
        "alert_tier": "Tier", "ticker": "コード", "name": "銘柄",
        "phase": "Phase", "regime": "資金フロー", "cover_score": "Cover",
        "ignition_score": "Ignition", "long_demand_score": "Long",
        "short_pressure": "Pressure", "confidence": "信頼度",
        "alert_reason": "確認理由",
    }
    st.dataframe(
        _priority_show[_priority_cols].rename(columns=_priority_labels),
        hide_index=True,
        use_container_width=True,
    )

st.divider()

filtered = scored[scored["cover_score"] >= min_score].copy()
filtered.insert(0, "順位", range(1, len(filtered) + 1))

if active_condition is not None:
    _label = str(active_condition.get("robustness", ""))
    _cond = (
        f"C{int(active_condition.get('cover_min', 0))}/"
        f"I{int(active_condition.get('ignition_min', 0))}/"
        f"L{int(active_condition.get('long_min', 0))}/"
        f"P{int(active_condition.get('pressure_min', 0))}/"
        f"Q{int(active_condition.get('confidence_min', 0))}"
    )
    _matches = scored[scored["optimizer_match"] == True].copy()
    st.markdown("## ⭐ 検証済み条件マッチ")
    st.caption(f"{_label} 条件 {_cond} を現在ランキングへ自動適用しています。")
    if _matches.empty:
        st.info("現在、この検証済み条件をすべて満たす銘柄はありません。")
    else:
        _matches = _matches.sort_values(
            ["match_strength", "cover_score", "ignition_score"],
            ascending=False,
        )
        for _, _r in _matches.head(8).iterrows():
            st.markdown(
                f"**{_r['optimizer_label']}｜{_r['name']}（{_r['ticker']}）**　"
                f"Match **{_r['match_strength']:.0f}** / Cover **{_r['cover_score']:.0f}** / "
                f"Ignition **{_r['ignition_score']:.0f}** / Long **{_r['long_demand_score']:.0f}** / "
                f"Pressure **{_r['short_pressure']:.0f}** / Confidence **{_r['confidence']:.0f}**"
            )
    st.divider()

st.markdown("## ⚡ 初動昇格ランキング")
st.caption(
    "直近営業日と前営業日を同じ公表空売りデータで比較し、価格・出来高側の状態が一段強くなった銘柄だけを抽出します。"
)
if promotions.empty:
    st.info("今回は新しい昇格シグナルがありません。")
else:
    promo = promotions.copy().head(20)
    promo.insert(0, "順位", range(1, len(promo) + 1))
    promo["前Cover"] = promo["prev_cover_score"].map(lambda x: fmt_num(x, 0))
    promo["現Cover"] = promo["cover_score"].map(lambda x: fmt_num(x, 0))
    promo["ΔCover"] = promo["cover_delta"].map(lambda x: f"{float(x):+.0f}")
    promo["前Ignition"] = promo["prev_ignition_score"].map(lambda x: fmt_num(x, 0))
    promo["現Ignition"] = promo["ignition_score"].map(lambda x: fmt_num(x, 0))
    promo["ΔIgnition"] = promo["ignition_delta"].map(lambda x: f"{float(x):+.0f}")
    promo["出来高"] = promo["vol_ratio"].map(lambda x: fmt_num(x, 2, "x"))
    promo["RS"] = promo["rs_watch"].map(lambda x: fmt_num(x, 0))
    promo_cols = [
        "順位", "ticker", "name", "prev_phase", "phase", "promotion_reason",
        "前Cover", "現Cover", "ΔCover", "前Ignition", "現Ignition", "ΔIgnition",
        "regime", "optimizer_label", "match_strength", "出来高", "RS", "confidence",
    ]
    promo_labels = {
        "ticker": "コード", "name": "銘柄", "prev_phase": "前回Phase",
        "phase": "今回Phase", "promotion_reason": "昇格理由",
        "regime": "資金フロー", "optimizer_label": "検証条件",
        "match_strength": "Match", "confidence": "信頼度",
    }
    st.dataframe(
        promo[promo_cols].rename(columns=promo_labels),
        hide_index=True,
        use_container_width=True,
        height=min(680, 80 + 35 * len(promo)),
    )

    top_promos = promo.head(5)
    st.markdown("### 🚨 今日変化した上位候補")
    for _, r in top_promos.iterrows():
        st.markdown(
            f"**{r['name']}（{r['ticker']}）**　{r['prev_phase']} → **{r['phase']}**  \\n"
            f"{r['promotion_reason']}  \\n"
            f"Cover {r['prev_cover_score']:.0f}→**{r['cover_score']:.0f}** "
            f"({r['cover_delta']:+.0f})｜Ignition {r['prev_ignition_score']:.0f}→"
            f"**{r['ignition_score']:.0f}** ({r['ignition_delta']:+.0f})｜{r['regime']}"
        )

st.divider()
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
    display["Ignition"] = display["ignition_score"].map(lambda x: fmt_num(x, 0))
    display["信頼度"] = display["confidence"].map(lambda x: f"{int(x)}")
    display["検証条件"] = display["optimizer_label"].replace("", "—")
    display["Match"] = display["match_strength"].map(
        lambda x: "—" if pd.isna(x) or float(x) <= 0 else f"{float(x):.0f}"
    )

    cols = [
        "順位", "ticker", "name", "phase", "検証条件", "Match", "regime",
        "cover_score", "short_pressure", "Ignition", "long_demand_score", "空売り%",
        "Δ空売り", "institution_count", "買戻Breadth", "DTC", "出来高",
        "AVWAP", "5日高値", "RS", "信頼度",
    ]
    labels = {
        "ticker": "コード", "name": "銘柄", "phase": "Phase", "regime": "資金フロー",
        "cover_score": "Cover", "long_demand_score": "Long",
        "short_pressure": "Pressure", "institution_count": "機関数",
    }
    table = display[cols].rename(columns=labels)
    st.dataframe(table, hide_index=True, use_container_width=True, height=min(760, 80 + 35 * len(table)))

    strong = filtered[filtered["phase"].isin(["🔥 COVER EARLY", "✅ COVER CONFIRMED", "🚀 SQUEEZE"])]
    if not strong.empty:
        st.markdown("### 🔥 今見る候補")
        for _, r in strong.head(5).iterrows():
            st.markdown(
                f"**{r['phase']}｜{r['name']}（{r['ticker']}）**　"
                f"Cover **{r['cover_score']:.0f}** / Ignite **{r['ignition_score']:.0f}** / "
                f"Long **{r['long_demand_score']:.0f}** / Pressure **{r['short_pressure']:.0f}**　"
                f"{r.get('regime', '')}　出来高 {fmt_num(r.get('vol_ratio'), 2, 'x')}　"
                f"RS {fmt_num(r.get('rs_watch'), 0)}"
            )

st.divider()
st.markdown("## 🧪 シグナル実績バックテスト")
st.caption(
    "当日終値でシグナル確定 → 翌営業日始値でエントリーした想定です。"
    "JPXは掲載日が取得できる場合は掲載日を利用可能日として扱い、未来情報の混入を抑えます。"
)

_bt_targets = list(dict.fromkeys(jpx_candidates + watchlist))[:bt_tickers]
if not _bt_targets:
    st.info("バックテスト対象銘柄がありません。")
else:
    if st.button("▶️ バックテストを実行", key="short_cover_backtest_btn", use_container_width=False):
        with st.spinner("過去シグナルを再構成して検証中..."):
            _new_bt = backtest_short_cover(
                events=events,
                tickers=_bt_targets,
                sessions=bt_sessions,
                max_tickers=bt_tickers,
                min_cover_score=bt_min_score,
            )
            st.session_state.short_cover_backtest = _new_bt
            st.session_state.short_cover_optimizer = optimize_short_cover_thresholds(
                _new_bt,
                horizon=opt_horizon,
                train_fraction=opt_train_fraction,
                min_train_signals=8,
                min_test_signals=4,
                top_train_candidates=30,
            )
        st.rerun()

    bt = st.session_state.get("short_cover_backtest")
    if bt is not None:
        if bt.empty:
            st.warning("指定条件では検証可能なシグナルがありませんでした。")
        else:
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("検証シグナル", f"{len(bt)}件")
            _r5 = pd.to_numeric(bt["ret_5d"], errors="coerce").dropna()
            _r10 = pd.to_numeric(bt["ret_10d"], errors="coerce").dropna()
            b2.metric("5日勝率", f"{((_r5 > 0).mean() * 100):.1f}%" if not _r5.empty else "—")
            b3.metric("5日平均", f"{_r5.mean():+.2f}%" if not _r5.empty else "—")
            b4.metric("10日平均", f"{_r10.mean():+.2f}%" if not _r10.empty else "—")

            phase_summary = summarize_backtest(bt, "phase")
            regime_summary = summarize_backtest(bt, "regime")

            left_bt, right_bt = st.columns(2)
            with left_bt:
                st.markdown("### Phase別")
                if not phase_summary.empty:
                    ps = phase_summary.copy()
                    for col in [
                        "win_1d", "win_3d", "win_5d", "win_10d",
                        "avg_1d", "avg_3d", "avg_5d", "avg_10d",
                        "median_5d", "avg_mfe_10d", "avg_mae_10d",
                    ]:
                        ps[col] = ps[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%")
                    ps = ps.rename(columns={
                        "phase": "Phase", "signals": "件数",
                        "win_1d": "1日勝率", "win_3d": "3日勝率",
                        "win_5d": "5日勝率", "win_10d": "10日勝率",
                        "avg_1d": "1日平均", "avg_3d": "3日平均",
                        "avg_5d": "5日平均", "avg_10d": "10日平均",
                        "median_5d": "5日中央値",
                        "avg_mfe_10d": "10日MFE", "avg_mae_10d": "10日MAE",
                    })
                    st.dataframe(ps, hide_index=True, use_container_width=True)

            with right_bt:
                st.markdown("### 資金フロー別")
                if not regime_summary.empty:
                    rs = regime_summary.copy()
                    for col in [
                        "win_1d", "win_3d", "win_5d", "win_10d",
                        "avg_1d", "avg_3d", "avg_5d", "avg_10d",
                        "median_5d", "avg_mfe_10d", "avg_mae_10d",
                    ]:
                        rs[col] = rs[col].map(lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%")
                    rs = rs.rename(columns={
                        "regime": "資金フロー", "signals": "件数",
                        "win_1d": "1日勝率", "win_3d": "3日勝率",
                        "win_5d": "5日勝率", "win_10d": "10日勝率",
                        "avg_1d": "1日平均", "avg_3d": "3日平均",
                        "avg_5d": "5日平均", "avg_10d": "10日平均",
                        "median_5d": "5日中央値",
                        "avg_mfe_10d": "10日MFE", "avg_mae_10d": "10日MAE",
                    })
                    st.dataframe(rs, hide_index=True, use_container_width=True)

            st.markdown("### シグナル明細")
            bt_show = bt.copy()
            for col in ["signal_date", "entry_date"]:
                bt_show[col] = pd.to_datetime(bt_show[col], errors="coerce").dt.strftime("%Y-%m-%d")
            for col in ["ret_1d", "ret_3d", "ret_5d", "ret_10d", "mfe_10d", "mae_10d"]:
                bt_show[col] = bt_show[col].map(
                    lambda x: "—" if pd.isna(x) else f"{float(x):+.2f}%"
                )
            bt_cols = [
                "signal_date", "ticker", "name", "phase", "regime",
                "cover_score", "ignition_score", "long_demand_score",
                "entry_date", "ret_1d", "ret_3d", "ret_5d", "ret_10d",
                "mfe_10d", "mae_10d",
            ]
            bt_labels = {
                "signal_date": "シグナル日", "ticker": "コード", "name": "銘柄",
                "phase": "Phase", "regime": "資金フロー",
                "cover_score": "Cover", "ignition_score": "Ignition",
                "long_demand_score": "Long", "entry_date": "翌日エントリー",
                "ret_1d": "1日", "ret_3d": "3日", "ret_5d": "5日",
                "ret_10d": "10日", "mfe_10d": "MFE", "mae_10d": "MAE",
            }
            st.dataframe(
                bt_show[bt_cols].rename(columns=bt_labels),
                hide_index=True,
                use_container_width=True,
                height=420,
            )

            csv_data = bt.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ バックテストCSV",
                data=csv_data,
                file_name="short_cover_backtest.csv",
                mime="text/csv",
                key="short_cover_backtest_download",
            )

            st.markdown("## 🧬 最適条件ファインダー")
            st.caption(
                "時系列を前半の学習期間と後半の検証期間に分けます。"
                "閾値は学習期間だけで探索し、その後の検証期間で再現した条件を上位に表示します。"
            )
            optimizer = st.session_state.get("short_cover_optimizer")
            if optimizer is None:
                optimizer = optimize_short_cover_thresholds(
                    bt,
                    horizon=opt_horizon,
                    train_fraction=opt_train_fraction,
                    min_train_signals=8,
                    min_test_signals=4,
                    top_train_candidates=30,
                )
                st.session_state.short_cover_optimizer = optimizer

            if optimizer.empty:
                st.info("最適化に必要なシグナル数がまだ不足しています。検証営業日数や対象銘柄数を増やしてください。")
            else:
                best = optimizer.iloc[0]
                o1, o2, o3, o4 = st.columns(4)
                o1.metric("安定度", f"{best['stability_score']:.0f}/100")
                o2.metric("検証シグナル", f"{int(best['test_signals'])}件")
                o3.metric(
                    f"検証{opt_horizon}日平均",
                    "—" if pd.isna(best["test_avg"]) else f"{float(best['test_avg']):+.2f}%",
                )
                o4.metric(
                    f"検証{opt_horizon}日勝率",
                    "—" if pd.isna(best["test_win"]) else f"{float(best['test_win']):.1f}%",
                )

                st.markdown(
                    f"### {best['robustness']}｜候補条件  "
                    f"Cover ≥ **{best['cover_min']:.0f}** / "
                    f"Ignition ≥ **{best['ignition_min']:.0f}** / "
                    f"Long ≥ **{best['long_min']:.0f}** / "
                    f"Pressure ≥ **{best['pressure_min']:.0f}** / "
                    f"Confidence ≥ **{best['confidence_min']:.0f}**"
                )

                opt_show = optimizer.head(15).copy()
                for col in [
                    "train_win", "train_avg", "train_median",
                    "test_win", "test_avg", "test_median",
                    "test_mfe", "test_mae",
                ]:
                    opt_show[col] = opt_show[col].map(
                        lambda x: "—" if pd.isna(x) else f"{float(x):+.1f}%"
                    )
                opt_show["stability_score"] = opt_show["stability_score"].map(lambda x: f"{float(x):.0f}")
                opt_show["条件"] = opt_show.apply(
                    lambda r: (
                        f"C{int(r['cover_min'])} / I{int(r['ignition_min'])} / "
                        f"L{int(r['long_min'])} / P{int(r['pressure_min'])} / "
                        f"Q{int(r['confidence_min'])}"
                    ),
                    axis=1,
                )
                opt_cols = [
                    "robustness", "条件", "train_signals", "train_win", "train_avg",
                    "test_signals", "test_win", "test_avg", "test_mfe", "test_mae",
                    "stability_score",
                ]
                opt_labels = {
                    "robustness": "判定", "train_signals": "学習件数",
                    "train_win": "学習勝率", "train_avg": "学習平均",
                    "test_signals": "検証件数", "test_win": "検証勝率",
                    "test_avg": "検証平均", "test_mfe": "検証MFE",
                    "test_mae": "検証MAE", "stability_score": "安定度",
                }
                st.dataframe(
                    opt_show[opt_cols].rename(columns=opt_labels),
                    hide_index=True,
                    use_container_width=True,
                    height=min(600, 80 + 35 * len(opt_show)),
                )

                st.caption(
                    "🟢 ROBUST = 後半の未使用データでもプラス期待値・勝率50%以上・安定度60以上。"
                    "サンプル数が少ない条件は上位でも過信しない設計です。"
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

m1, m2, m3, m4, m5, m6 = st.columns(6)
m1.metric("Cover Score", f"{selected['cover_score']:.0f}")
m2.metric("Ignition", f"{selected['ignition_score']:.0f}")
m3.metric("Long Demand", f"{selected['long_demand_score']:.0f}")
m4.metric("Short Pressure", f"{selected['short_pressure']:.0f}")
m5.metric("公表空売り", fmt_num(selected.get("short_ratio"), 2, "%"))
m6.metric("DTC", fmt_num(selected.get("dtc"), 1, "日"))

st.markdown(f"### {selected['phase']}　{selected['name']}（{ticker}）")
st.caption(f"資金フロー判定：{selected.get('regime', '—')}")

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
        ["Ignition", selected.get("ignition_score")],
        ["資金フロー", selected.get("regime")],
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
### 初動昇格ランキング

前営業日と直近営業日の価格・出来高シグナルを同じShort Pressure条件で比較し、
Phase上昇、Cover 65突破、Ignition 65突破、新規5日高値突破、新規AVWAP回復を検出します。

**これは「前営業日の空売り残高を完全再現したバックテスト」ではありません。**
公表残高を固定したまま、価格・出来高側で何が新しく点火したかを見るためのデイリー変化検知です。

### 今日の最優先チェック

確認優先度は、次を統合した0〜100のスコアです。

- 検証済みROBUST / PROMISING条件との一致 **30%**
- 今日のPhase昇格・Cover/Ignition上昇 **25%**
- 現在Phase **20%**
- 資金フロー分類 **10%**
- データ信頼度 **10%**
- 出来高確認 **5%**

A+ / A / B / C は売買判断ではなく、**今日どの候補から確認するか**の順番です。

### 検証済み条件マッチ

最適条件ファインダーで ROBUST / PROMISING と判定された上位条件を、現在のランキングへ自動適用します。

- 5つの閾値をすべて満たした銘柄だけに **⭐ ROBUST MATCH** / **🟡 PROMISING MATCH**
- Match Strength は各閾値をどれだけ上回ったかを0〜100で表示
- 最適化をまだ実行していない場合は表示しません
- 過去成績を保証するものではなく、「検証済み条件との一致」を示します

### 最適条件ファインダー

- 前半期間だけで Cover / Ignition / Long / Pressure / Confidence の閾値を探索
- 後半期間は探索に使わず、ホールドアウト検証だけに使用
- 検証期間でもプラス期待値・勝率・サンプル数が保てた条件を ROBUST / PROMISING と表示
- 全期間を一度に最適化しないことで、過学習を抑えます

### バックテスト

- シグナルは当日終値で確定
- エントリーは翌営業日始値
- 1 / 3 / 5 / 10営業日後の終値リターン
- MFE = エントリー後10営業日の最大上昇率
- MAE = エントリー後10営業日の最大下落率
- JPX掲載日が取れる場合は掲載日より前のシグナル生成には使いません

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

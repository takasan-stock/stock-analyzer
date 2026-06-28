import streamlit as st
import pandas as pd
import plotly.express as px
import os
import datetime

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="銘柄管理ダッシュボード", layout="wide", page_icon="📊")

# データ保存用のローカルCSVファイル名
DATA_FILE = "portfolio_data.csv"

COLUMNS = [
    "ティッカー", "銘柄名", "セクター", "ステータス",
    "売上5y CAGR", "売上予想", "PER", "ネットキャッシュ",
    "投資家メモ", "更新日"
]

SECTOR_OPTIONS = ["IT・通信", "電気機器", "小売", "サービス", "金融", "その他", "未分類"]
STATUS_OPTIONS = ["監視中", "打診買い", "保有中", "見送り"]
NETCASH_OPTIONS = ["潤沢", "普通", "マイナス", "不明"]

# ==========================================
# データの読み込み・保存
# ==========================================
def load_data():
    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
        return df
    df = pd.read_csv(DATA_FILE, dtype={"ティッカー": str}, encoding="utf-8-sig")
    # 列が足りない場合（旧バージョンのCSVなど）は補完しておく
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]

def save_data(df):
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")

# st.session_state に持たせることで、フォーム送信などの再実行時にも
# 編集中のデータが消えないようにする
if "df" not in st.session_state:
    st.session_state.df = load_data()

st.title("📊 銘柄管理ダッシュボード")
st.caption("Obsidian（Dataview / Templater）の代わりに、ブラウザ上で動く株式管理ダッシュボードです。")

tab1, tab2, tab3 = st.tabs(["📋 一覧・編集", "📝 新規銘柄登録", "📊 分析"])

# ------------------------------------------
# タブ1：一覧表示 ＋ 編集・削除
# ------------------------------------------
with tab1:
    df = st.session_state.df
    st.subheader("保有・監視銘柄データ")

    if df.empty:
        st.info("現在登録されているデータがありません。「新規銘柄登録」タブから追加してください。")
    else:
        # --- 検索・絞り込み（表示用） ---
        col_f1, col_f2 = st.columns([2, 1])
        with col_f1:
            keyword = st.text_input("🔎 ティッカー・銘柄名で検索", "")
        with col_f2:
            status_options = sorted(df["ステータス"].dropna().astype(str).unique())
            status_filter = st.multiselect(
                "ステータスで絞り込み",
                options=status_options,
                default=status_options
            )

        view_df = df[df["ステータス"].isin(status_filter)]
        if keyword:
            mask = (
                view_df["ティッカー"].astype(str).str.contains(keyword, case=False, na=False)
                | view_df["銘柄名"].astype(str).str.contains(keyword, case=False, na=False)
            )
            view_df = view_df[mask]

        st.dataframe(view_df, use_container_width=True, hide_index=True)

        st.divider()

        # --- 編集・削除エリア ---
        st.markdown("##### ✏️ データの編集・削除")
        st.caption(
            "セルをダブルクリックすると直接編集できます。行頭にチェックを入れて削除、"
            "下に空欄の行が出たら新規行として入力もできます。編集後は必ず「変更を保存」を押してください。"
        )

        edited_df = st.data_editor(
            df,
            use_container_width=True,
            hide_index=True,
            num_rows="dynamic",
            column_config={
                "セクター": st.column_config.SelectboxColumn(options=SECTOR_OPTIONS),
                "ステータス": st.column_config.SelectboxColumn(options=STATUS_OPTIONS),
                "ネットキャッシュ": st.column_config.SelectboxColumn(options=NETCASH_OPTIONS),
            },
            key="full_editor"
        )

        col_b1, col_b2, col_b3 = st.columns(3)
        with col_b1:
            if st.button("💾 変更を保存", type="primary"):
                st.session_state.df = edited_df.reset_index(drop=True)
                save_data(st.session_state.df)
                st.success("✅ 変更を保存しました！")
                st.rerun()
        with col_b2:
            csv = df.to_csv(index=False).encode("utf-8-sig")
            st.download_button(
                "⬇️ CSVをダウンロード",
                data=csv,
                file_name=f"portfolio_data_{datetime.date.today()}.csv",
                mime="text/csv"
            )
        with col_b3:
            if st.button("🔄 保存済みデータを再読込"):
                st.session_state.df = load_data()
                st.rerun()

# ------------------------------------------
# タブ2：新規銘柄登録フォーム
# ------------------------------------------
with tab2:
    st.subheader("分析結果の入力")

    with st.form("register_form", clear_on_submit=True):
        col1, col2 = st.columns(2)

        with col1:
            ticker = st.text_input("ティッカーコード (例: 7974)")
            name = st.text_input("銘柄名 (例: 任天堂)")
            sector = st.selectbox("セクター", SECTOR_OPTIONS)
            status = st.selectbox("ステータス", STATUS_OPTIONS)

        with col2:
            cagr = st.text_input("売上5y CAGR (例: 15.2%)")
            forecast = st.text_input("売上予想 (例: 今期+10%成長)")
            per = st.text_input("PER (例: 15.5)")
            net_cash = st.selectbox("ネットキャッシュ", NETCASH_OPTIONS)

        memo = st.text_area("投資家メモ (決算の所感、チャートの形状、カタリストなど)")

        submitted = st.form_submit_button("💾 データベースに登録")

        if submitted:
            if ticker == "":
                st.error("⚠️ ティッカーコードは必須です！")
            else:
                existing_tickers = st.session_state.df["ティッカー"].astype(str).tolist()
                if ticker in existing_tickers:
                    st.warning(
                        f"⚠️ ティッカー {ticker} は既に登録されています。"
                        "新しい行として追加されます。既存データを更新したい場合は"
                        "「一覧・編集」タブで直接編集してください。"
                    )

                new_data = pd.DataFrame([{
                    "ティッカー": ticker,
                    "銘柄名": name,
                    "セクター": sector,
                    "ステータス": status,
                    "売上5y CAGR": cagr,
                    "売上予想": forecast,
                    "PER": per,
                    "ネットキャッシュ": net_cash,
                    "投資家メモ": memo,
                    "更新日": datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                }])

                st.session_state.df = pd.concat([st.session_state.df, new_data], ignore_index=True)
                save_data(st.session_state.df)

                st.success(f"✅ {ticker} ({name}) を登録しました！")
                st.rerun()

# ------------------------------------------
# タブ3：分析
# ------------------------------------------
with tab3:
    df = st.session_state.df
    st.subheader("ポートフォリオの分析")

    if df.empty:
        st.info("データが登録されると、ここにグラフや統計が表示されます。")
    else:
        # PERを数値に変換（変換できないものはNaNにする）
        per_numeric = pd.to_numeric(df["PER"], errors="coerce")

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("登録銘柄数", f"{len(df)} 件")
        m2.metric("保有中", f"{(df['ステータス'] == '保有中').sum()} 件")
        m3.metric("監視中", f"{(df['ステータス'] == '監視中').sum()} 件")
        avg_per = per_numeric.mean()
        m4.metric("平均PER", f"{avg_per:.1f}" if pd.notna(avg_per) else "—")

        col_c1, col_c2 = st.columns(2)

        with col_c1:
            sector_counts = df["セクター"].value_counts().reset_index()
            sector_counts.columns = ["セクター", "件数"]
            fig_sector = px.pie(
                sector_counts, names="セクター", values="件数",
                title="セクター別 銘柄数"
            )
            st.plotly_chart(fig_sector, use_container_width=True)

        with col_c2:
            status_counts = df["ステータス"].value_counts().reset_index()
            status_counts.columns = ["ステータス", "件数"]
            fig_status = px.bar(
                status_counts, x="ステータス", y="件数",
                title="ステータス別 銘柄数", text="件数"
            )
            st.plotly_chart(fig_status, use_container_width=True)

        if per_numeric.notna().sum() > 0:
            per_df = df.copy()
            per_df["PER（数値）"] = per_numeric
            per_df = per_df.dropna(subset=["PER（数値）"])
            fig_per = px.bar(
                per_df.sort_values("PER（数値）"),
                x="銘柄名", y="PER（数値）", color="セクター",
                title="銘柄別 PER"
            )
            st.plotly_chart(fig_per, use_container_width=True)

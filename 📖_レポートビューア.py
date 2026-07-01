"""
過去レポートの検索＆閲覧ビューア
GitHubリポジトリ（stock-reports）から直接Markdownを読み込んで表示する。
スマホからでも過去の分析をサクッと振り返れるシンプル閲覧専用ページ。
"""
import streamlit as st
import requests
import base64
from collections import defaultdict

st.set_page_config(
    page_title="レポートビューア",
    page_icon="📖",
    layout="wide"
)

# ==========================================
# GitHub API ヘルパー
# ==========================================
def get_github_config():
    try:
        return {
            "token": st.secrets["GITHUB_TOKEN"],
            "repo":  st.secrets["GITHUB_REPO"],
            "branch": st.secrets.get("GITHUB_BRANCH", "main"),
        }
    except Exception:
        return None

@st.cache_data(ttl=300)  # 5分キャッシュ（スマホ回線でも快適に）
def fetch_report_index(token: str, repo: str, branch: str) -> list:
    """
    GitHubの reports/ フォルダのファイル一覧を取得し、
    {'ticker', 'kind', 'key', 'path', 'download_url'} のリストを返す。
    ttl=300なのでボタンを押さなくても5分後に自動更新される。
    """
    url = f"https://api.github.com/repos/{repo}/contents/reports"
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
    }
    try:
        resp = requests.get(url, headers=headers, params={"ref": branch}, timeout=10)
    except requests.exceptions.RequestException as e:
        return []

    if resp.status_code != 200:
        return []

    items = resp.json()
    results = []
    for item in items:
        fname = item.get("name", "")
        # .md 以外・.migrated・_backup は除外
        if not fname.endswith(".md") or fname.endswith(".migrated"):
            continue
        # ファイル名を <ticker>_<kind>_<key>.md に分解
        parts = fname[:-3].split("_", 2)
        if len(parts) < 3:
            continue
        ticker, kind, key = parts
        if kind not in ("analysis", "earnings"):
            continue
        results.append({
            "ticker": ticker,
            "kind": kind,
            "key": key,
            "path": item.get("path", ""),
            "download_url": item.get("download_url", ""),
        })
    return results

@st.cache_data(ttl=300)
def fetch_report_content(download_url: str, token: str) -> str:
    """
    download_url からMarkdownの本文を取得して返す。
    Private リポジトリなので Authorization ヘッダーが必要。
    """
    headers = {"Authorization": f"Bearer {token}"}
    try:
        resp = requests.get(download_url, headers=headers, timeout=10)
        if resp.status_code == 200:
            return resp.text
        return f"⚠️ 取得失敗（HTTP {resp.status_code}）"
    except requests.exceptions.RequestException as e:
        return f"⚠️ 接続エラー（{e}）"

# ==========================================
# サイドバー：銘柄・種別・エントリ選択
# ==========================================
st.sidebar.title("📖 レポートビューア")
st.sidebar.caption("GitHubに保存した分析レポートをここで読めます。")

config = get_github_config()
if config is None:
    st.sidebar.error("GitHub連携が未設定です。\nSecretsにGITHUB_TOKEN / GITHUB_REPOを設定してください。")
    st.stop()

# インデックス取得
with st.spinner("GitHubからファイル一覧を取得中..."):
    index = fetch_report_index(config["token"], config["repo"], config["branch"])

if not index:
    st.sidebar.warning("レポートがまだ1件もありません。\nメインアプリで銘柄レポートを保存してください。")
    st.info("レポートが保存されると、ここに一覧が表示されます。")
    st.stop()

# ティッカーごとにグルーピング
by_ticker = defaultdict(list)
for item in index:
    by_ticker[item["ticker"]].append(item)

ticker_list = sorted(by_ticker.keys())

# メインアプリの portfolio_data.csv から銘柄名を取得（あれば表示名を日本語に）
try:
    import pandas as pd
    df_portfolio = pd.read_csv("portfolio_data.csv", dtype={"ティッカー": str}, encoding="utf-8-sig")
    name_map = {str(row["ティッカー"]): row["銘柄名"] for _, row in df_portfolio.iterrows()}
except Exception:
    name_map = {}

def ticker_label(ticker: str) -> str:
    name = name_map.get(ticker, "")
    return f"{ticker}　{name}" if name else ticker

# 銘柄選択
selected_ticker = st.sidebar.selectbox(
    "銘柄を選択",
    options=ticker_list,
    format_func=ticker_label,
)

# 種別選択（銘柄分析 / 決算分析）
ticker_entries = by_ticker[selected_ticker]
kinds_available = sorted(set(e["kind"] for e in ticker_entries))
kind_labels = {"analysis": "📝 銘柄分析レポート", "earnings": "📊 決算分析レポート"}

selected_kind = st.sidebar.radio(
    "種別",
    options=kinds_available,
    format_func=lambda k: kind_labels.get(k, k),
)

# エントリ選択（日付・四半期）
kind_entries = sorted(
    [e for e in ticker_entries if e["kind"] == selected_kind],
    key=lambda e: e["key"],
    reverse=True,   # 新しい順
)
entry_keys = [e["key"] for e in kind_entries]

selected_key = st.sidebar.selectbox(
    "日付 / 四半期",
    options=entry_keys,
)

# 強制リフレッシュボタン（キャッシュをクリアしてGitHubを再取得）
st.sidebar.divider()
if st.sidebar.button("🔄 最新データに更新"):
    fetch_report_index.clear()
    fetch_report_content.clear()
    st.rerun()

# ==========================================
# メインエリア：レポート表示
# ==========================================
selected_entry = next((e for e in kind_entries if e["key"] == selected_key), None)

if selected_entry is None:
    st.info("左のサイドバーで銘柄とエントリを選んでください。")
    st.stop()

# ヘッダー
company_name = name_map.get(selected_ticker, "")
kind_str = kind_labels.get(selected_entry["kind"], selected_entry["kind"])
st.title(f"{company_name}（{selected_ticker}）" if company_name else selected_ticker)
st.caption(f"{kind_str}　／　{selected_key}")
st.divider()

# 本文取得・表示
with st.spinner("レポートを読み込み中..."):
    content = fetch_report_content(selected_entry["download_url"], config["token"])

st.markdown(content)

# ダウンロードボタン（スマホでテキストコピーしやすいように）
st.divider()
st.download_button(
    "⬇️ このレポートをダウンロード（.md）",
    data=content.encode("utf-8"),
    file_name=f"{selected_ticker}_{selected_entry['kind']}_{selected_key}.md",
    mime="text/markdown",
)

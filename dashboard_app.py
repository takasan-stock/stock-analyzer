import streamlit as st
import pandas as pd
import plotly.express as px
import yfinance as yf
import requests
import base64
import os
import io
import zipfile
import re
import json
import datetime

# ==========================================
# ページ設定
# ==========================================
st.set_page_config(page_title="銘柄管理ダッシュボード", layout="wide", page_icon="📊")

# データ保存用のローカルCSVファイル名
DATA_FILE = "portfolio_data.csv"

# 個別銘柄レポート（Markdown）の保存先
REPORT_DIR = "reports"
REPORT_BACKUP_DIR = os.path.join(REPORT_DIR, "_backup")
os.makedirs(REPORT_BACKUP_DIR, exist_ok=True)

COLUMNS = [
    "ティッカー", "銘柄名", "セクター", "ステータス",
    "売上5y CAGR", "売上予想", "PER", "ネットキャッシュ",
    "投資家メモ", "更新日"
]

SECTOR_OPTIONS = ["IT・通信", "電気機器", "小売", "サービス", "金融", "その他", "未分類"]
STATUS_OPTIONS = ["監視中", "打診買い", "保有中", "見送り"]
NETCASH_OPTIONS = ["潤沢", "普通", "マイナス", "不明"]

# 個別銘柄レポート作成時にAIへ渡す分析プロンプトのテンプレート
ANALYST_PROMPT_TEMPLATE = """役割： あなたはプロの投資アナリストです。データを元に、SWOT分析を用いた深い洞察を提供してください。
分析指示：

1. 業績ハイライト：まず冒頭のレポート項目として「売上5y CAGR」と「今期売上予想」の数値を明記し、成長の質を評価してください。
2. SWOT分析: 強み・弱み・機会・脅威を整理してください。
3. キャッシュフロー分析（表形式）：直近3期分のキャッシュフローを調査し、Markdownの表形式で数値も入れてまとめてください。
4. 具体的な目標株価の算出：3つのシナリオ（弱気・ベース・強気）で算出してください。
5. 定性・財務: 決算短信、有価証券報告書、IR資料の要点をまとめてください。
6. 投資判断：長期投資としての適性を A〜D で判定し、提言を述べてください。
7. 最新のアナリストレポートを基に、企業の評価コメント、株価目標、および投資判断の推奨を3点に要約してください。
8. ピオトロスキーのFスコアを表示してください。"""

# 四半期ごとの決算分析用プロンプトのテンプレート（{}部分は実行時に埋める）
EARNINGS_PROMPT_TEMPLATE = """役割：
あなたはプロの投資アナリストです。
発表された決算数値、企業決算資料（IRのPDFまたは画像）を読み込み、市場の期待値を比較し、今後の株価への影響を初心者でも理解できる形で要約してください。
決算データ：
- 銘柄: {company_name}　{quarter}（{ticker}）
- 対象四半期: {quarter}
- 投資家の第一印象: {impression}
- メモ: {memo}
分析指示：
1. 進捗率分析: 通期予想に対する進捗率が、過去の季節性と比較して妥当か分析してください。
2. 上方修正の可能性: 今回の結果を受けて、今後上方修正が出る期待値を予測してください。
3. 売上高、営業利益、経常利益、純利益、EPS 営業利益率や前年比成長率、 配当金と配当性向、 キャッシュフロー（営業CF・投資CF・フリーCF）  の主要数値のまとめ（表形式：売上/営業利益/EPS/前年比など）
4. ポジティブ要因（3点）
5. ネガティブ要因（3点）
6. 総合コメント（初心者にも分かる言葉で）
7. 投資アナリストとして、この決算を踏まえた次の一手（例：押し目買い、様子見、利確検討など）"""

# ==========================================
# GitHub永続化レイヤー
# ==========================================
class GitHubStorage:
    """
    GitHub Contents API を使って、CSVやMarkdownファイルを
    GitHubリポジトリに永続保存するクラス。
    GitHub未設定時はローカルファイルにフォールバックする。
    """

    def __init__(self):
        self.config = self._load_config()

    def _load_config(self):
        try:
            return {
                "token":  st.secrets["GITHUB_TOKEN"],
                "repo":   st.secrets["GITHUB_REPO"],
                "branch": st.secrets.get("GITHUB_BRANCH", "main"),
            }
        except Exception:
            return None

    @property
    def enabled(self) -> bool:
        return self.config is not None

    def _headers(self):
        return {
            "Authorization": f"Bearer {self.config['token']}",
            "Accept": "application/vnd.github+json",
        }

    def _api_url(self, path_in_repo: str) -> str:
        return f"https://api.github.com/repos/{self.config['repo']}/contents/{path_in_repo}"

    def _get_sha(self, path_in_repo: str) -> str | None:
        """既存ファイルのSHAを取得する（更新時に必要）。存在しなければNoneを返す。"""
        try:
            resp = requests.get(
                self._api_url(path_in_repo), headers=self._headers(),
                params={"ref": self.config["branch"]}, timeout=10
            )
            if resp.status_code == 200:
                return resp.json().get("sha")
        except Exception:
            pass
        return None

    def read_text(self, path_in_repo: str) -> str | None:
        """
        GitHubからファイルの内容をテキストとして読み込む。
        失敗時はNoneを返す。
        """
        if not self.enabled:
            return None
        try:
            resp = requests.get(
                self._api_url(path_in_repo), headers=self._headers(),
                params={"ref": self.config["branch"]}, timeout=10
            )
            if resp.status_code == 200:
                b64 = resp.json().get("content", "").replace("\n", "")
                return base64.b64decode(b64).decode("utf-8-sig")
        except Exception:
            pass
        return None

    def write_text(self, path_in_repo: str, content: str, commit_message: str) -> tuple[bool, str]:
        """
        GitHubにテキストファイルを書き込む（新規作成 or 更新）。
        戻り値: (成功したか, メッセージ)
        """
        if not self.enabled:
            return False, "GitHub未設定"
        try:
            sha = self._get_sha(path_in_repo)
            payload = {
                "message": commit_message,
                "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
                "branch":  self.config["branch"],
            }
            if sha:
                payload["sha"] = sha
            resp = requests.put(
                self._api_url(path_in_repo), headers=self._headers(),
                json=payload, timeout=15
            )
            if resp.status_code in (200, 201):
                return True, f"GitHubに保存しました（{path_in_repo}）"
            elif resp.status_code == 401:
                return False, "GitHub認証エラー。Personal Access Tokenを確認してください。"
            elif resp.status_code == 404:
                return False, "リポジトリが見つかりません。GITHUB_REPOの設定を確認してください。"
            else:
                return False, f"GitHub保存失敗（HTTP {resp.status_code}）"
        except requests.exceptions.RequestException as e:
            return False, f"GitHub通信エラー（{e}）"

    def delete_file(self, path_in_repo: str, commit_message: str) -> tuple[bool, str]:
        """GitHubからファイルを削除する。"""
        if not self.enabled:
            return False, "GitHub未設定"
        try:
            sha = self._get_sha(path_in_repo)
            if not sha:
                return False, "ファイルが見つかりません"
            resp = requests.delete(
                self._api_url(path_in_repo), headers=self._headers(),
                json={"message": commit_message, "sha": sha, "branch": self.config["branch"]},
                timeout=10
            )
            return (True, "削除しました") if resp.status_code in (200, 204) else (False, f"HTTP {resp.status_code}")
        except requests.exceptions.RequestException as e:
            return False, str(e)

    def list_files(self, dir_in_repo: str) -> list[dict]:
        """
        GitHubのフォルダ内のファイル一覧を返す。
        戻り値: [{"name", "path", "sha", "download_url"}, ...]
        """
        if not self.enabled:
            return []
        try:
            resp = requests.get(
                self._api_url(dir_in_repo), headers=self._headers(),
                params={"ref": self.config["branch"]}, timeout=10
            )
            if resp.status_code == 200:
                return resp.json() if isinstance(resp.json(), list) else []
        except Exception:
            pass
        return []

    def read_binary_url(self, download_url: str) -> str | None:
        """download_urlから直接テキストを取得する（Private対応）。"""
        if not self.enabled:
            return None
        try:
            resp = requests.get(
                download_url, headers={"Authorization": f"Bearer {self.config['token']}"},
                timeout=10
            )
            return resp.text if resp.status_code == 200 else None
        except Exception:
            return None

# アプリ全体で共有するGitHubStorageインスタンス
# （毎回インスタンス化すると都度SecretsをパースするためSession stateに保持）
if "gh_storage" not in st.session_state:
    st.session_state.gh_storage = GitHubStorage()
gh = st.session_state.gh_storage

# ==========================================
# データの読み込み・保存（GitHub永続化対応版）
# ==========================================
def load_data():
    """
    起動時のデータ読み込み。
    GitHub設定あり → GitHubから取得してローカルにキャッシュ
    GitHub設定なし → ローカルCSVから読み込み（従来通り）
    """
    global DATA_FILE

    if gh.enabled:
        csv_text = gh.read_text(DATA_FILE)
        if csv_text:
            # GitHubから取得できたらローカルにも書いてキャッシュとして残す
            with open(DATA_FILE, "w", encoding="utf-8-sig") as f:
                f.write(csv_text)
        elif not os.path.exists(DATA_FILE):
            # GitHubにもローカルにもない → 空データで初期化
            csv_text = None

    if not os.path.exists(DATA_FILE):
        df = pd.DataFrame(columns=COLUMNS)
        df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")
        return df

    df = pd.read_csv(DATA_FILE, dtype={"ティッカー": str, "PER": str}, encoding="utf-8-sig")
    for col in COLUMNS:
        if col not in df.columns:
            df[col] = ""
    return df[COLUMNS]

def save_data(df):
    """
    データ保存。ローカルCSVに保存 + GitHub設定ありならGitHubにも保存。
    エラー時はローカル保存を維持してwarningを返す。
    """
    # まずローカルに保存（フォールバックとして常に実施）
    df.to_csv(DATA_FILE, index=False, encoding="utf-8-sig")

    if gh.enabled:
        with open(DATA_FILE, "r", encoding="utf-8-sig") as f:
            csv_text = f.read()
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
        ok, msg = gh.write_text(
            DATA_FILE, csv_text,
            f"Update portfolio_data.csv - {timestamp}"
        )
        if not ok:
            st.toast(f"⚠️ GitHub保存失敗（ローカルには保存済み）：{msg}", icon="⚠️")

# ==========================================
# reports/ の起動時GitHub同期
# ==========================================
def sync_reports_from_github():
    """
    起動時に一度だけ呼び出して、GitHubのreports/フォルダをローカルに同期する。
    すでに同期済み（session_stateにフラグあり）なら何もしない。
    """
    if st.session_state.get("reports_synced"):
        return
    if not gh.enabled:
        st.session_state.reports_synced = True
        return

    items = gh.list_files("reports")
    synced_count = 0
    for item in items:
        fname = item.get("name", "")
        if not fname.endswith(".md") or fname.endswith(".migrated"):
            continue
        local_path = os.path.join(REPORT_DIR, fname)
        if os.path.exists(local_path):
            continue  # すでにローカルにあればスキップ
        content = gh.read_binary_url(item.get("download_url", ""))
        if content:
            with open(local_path, "w", encoding="utf-8") as f:
                f.write(content)
            synced_count += 1

    st.session_state.reports_synced = True
    if synced_count > 0:
        st.toast(f"✅ GitHubからレポート {synced_count} 件を同期しました", icon="✅")



# ==========================================
# Yahoo!ファイナンスから銘柄情報を自動取得
# ==========================================
def make_yf_ticker(yf_symbol: str):
    """
    yfinance 1.5.x向けのTicker生成ヘルパー。
    自前のrequests.Sessionは渡さず session=None にすることで、
    yfinanceが内部でcurl_cffiベースのセッション（ブラウザTLS偽装付き）を
    自動生成する。これがYahooのBot検出を回避する正しい方法。
    レート制限対策として、日本語ロケール設定と短いスリープを入れる。
    """
    import time as _time
    try:
        yf.config.locale.lang = "ja-JP"
        yf.config.locale.region = "JP"
    except Exception:
        pass
    # 連続アクセス時のレート制限を避けるため、ごく短い間隔を空ける
    _time.sleep(0.3)
    return yf.Ticker(yf_symbol)  # session=None → curl_cffi自動使用

def contains_japanese(text) -> bool:
    """文字列にひらがな・カタカナ・漢字が含まれているか判定する"""
    if not text:
        return False
    return bool(re.search(r"[ぁ-んァ-ヶ一-龠]", text))

def pick_japanese_name(short_name, long_name) -> str:
    """
    shortName / longName のうち日本語表記の方を優先して返す。
    どちらも日本語でない場合（データが英語名しかない銘柄など）は shortName を優先してフォールバックする。
    """
    for candidate in (short_name, long_name):
        if contains_japanese(candidate):
            return candidate
    return short_name or long_name or ""

def fetch_stock_info(ticker_code: str):
    """
    日本株のティッカーコードから、銘柄名・PER・ネットキャッシュの簡易判定を取得する。
    取得できなかった項目は None になる。
    戻り値: (info_dict, error_message)
    """
    code = ticker_code.strip()
    if not code:
        return None, "ティッカーコードを入力してください。"

    # 「7974」のような数字だけのコードには .T（東証）を自動付与
    yf_symbol = code if "." in code else f"{code}.T"

    try:
        ticker_obj = make_yf_ticker(yf_symbol)

        # yfinance 1.5.x 以降は ticker.info が429を起こしやすいため、
        # まず軽量な fast_info で市場データを取得し、
        # 銘柄名だけ get_info() から取る2段階方式にする
        try:
            fast = ticker_obj.fast_info
            market_cap = getattr(fast, "market_cap", None) or 0
        except Exception:
            fast = None
            market_cap = 0

        # 銘柄名・PERはget_info()から取得（失敗時はフォールバック）
        try:
            info = ticker_obj.get_info()
        except Exception:
            info = {}

    except Exception as e:
        return None, f"取得処理でエラーが発生しました（{e}）。インターネット接続をご確認ください。"

    if not info and fast is None:
        return None, f"「{yf_symbol}」の情報が見つかりませんでした。ティッカーコードをご確認ください。"

    name = ""
    if info:
        # displayName は日本語名が入る場合がある（ja-JPロケール設定時）
        # shortName → longName の順でフォールバック
        display = info.get("displayName", "")
        name = (
            display if contains_japanese(display)
            else pick_japanese_name(info.get("shortName"), info.get("longName"))
        )

    # PERはget_info()から、取れなければget_valuation_measures()で補完
    per = info.get("trailingPE") if info else None
    if per is None:
        try:
            vm = ticker_obj.get_valuation_measures()
            if vm is not None and not vm.empty and "TrailingPE" in vm.columns:
                per_val = vm["TrailingPE"].dropna()
                per = float(per_val.iloc[-1]) if not per_val.empty else None
        except Exception:
            pass
    per_str = f"{per:.1f}" if isinstance(per, (int, float)) else ""

    # ネットキャッシュの簡易判定
    # fast_infoのmarket_capと、balance_sheetから現金・負債を取得
    net_cash_status = "不明"
    try:
        cash, debt = 0, 0
        try:
            bs = ticker_obj.get_balance_sheet()
            if bs is not None and not bs.empty:
                def get_bs_val(labels):
                    for label in labels:
                        if label in bs.index:
                            v = bs.loc[label].dropna()
                            if not v.empty:
                                return float(v.iloc[0])
                    return 0
                # yfinance 1.5.xの正しいラベル（スペースなし）
                cash = get_bs_val(["CashAndCashEquivalents",
                                   "CashCashEquivalentsAndShortTermInvestments",
                                   "CashFinancial", "CashEquivalents"])
                debt = get_bs_val(["TotalDebt", "LongTermDebt",
                                   "LongTermDebtAndCapitalLeaseObligation"])
        except Exception:
            pass

        # balance_sheetで取れなければget_info()の値で代替
        if (cash == 0 or debt == 0) and info:
            cash = cash or (info.get("totalCash") or 0)
            debt = debt or (info.get("totalDebt") or 0)

        if market_cap > 0 and (cash or debt):
            ratio = (cash - debt) / market_cap
            if ratio > 0.1:
                net_cash_status = "潤沢"
            elif ratio > -0.1:
                net_cash_status = "普通"
            else:
                net_cash_status = "マイナス"
    except Exception:
        pass

    # 何も取得できなかった場合のみエラーにする
    # 名前が取れなくてもPERや時価総額が取れていれば成功扱い（名前は手入力してもらう）
    if not name and per is None and market_cap == 0:
        return None, f"「{yf_symbol}」の情報が見つかりませんでした。ティッカーコードをご確認ください。"

    return {
        "name": name,
        "per": per_str,
        "net_cash": net_cash_status,
    }, None

# ==========================================
# 売上高の履歴取得 ＆ CAGR計算
# ==========================================
def fetch_revenue_history(ticker_code: str):
    """
    yfinanceから年次の売上高（Total Revenue）を取得する。
    Yahoo!ファイナンスの無料データは通常直近4期分程度しか提供されないため、
    「5年」分が取れるとは限らない点に注意。
    戻り値: (決算期と売上高のDataFrame または None, エラーメッセージ または None)
    """
    code = ticker_code.strip()
    if not code:
        return None, "ティッカーコードを入力してください。"

    yf_symbol = code if "." in code else f"{code}.T"

    try:
        ticker_obj = make_yf_ticker(yf_symbol)
        financials = ticker_obj.financials
    except Exception as e:
        return None, f"取得処理でエラーが発生しました（{e}）。インターネット接続をご確認ください。"

    if financials is None or financials.empty:
        return None, f"「{yf_symbol}」の決算データが見つかりませんでした。"

    revenue_row = None
    for label in ["Total Revenue", "TotalRevenue"]:
        if label in financials.index:
            revenue_row = financials.loc[label]
            break

    if revenue_row is None:
        return None, "売上高（Total Revenue）の項目が見つかりませんでした。"

    revenue_row = revenue_row.dropna().sort_index()  # 決算期が古い順に並べ替え

    if len(revenue_row) < 2:
        return None, "CAGRを算出するための決算データが2期分以上ありません。"

    df_rev = pd.DataFrame({
        "決算期": [d.strftime("%Y-%m") if hasattr(d, "strftime") else str(d) for d in revenue_row.index],
        "売上高（百万）": [round(v / 1_000_000, 1) for v in revenue_row.values],
    })
    df_rev["_date"] = list(revenue_row.index)
    df_rev["_raw"] = list(revenue_row.values)

    return df_rev, None

def fetch_next_earnings_date(ticker_code: str):
    """
    yfinanceから次回決算発表予定日を取得する。
    yfinanceのバージョンによって calendar が dict か DataFrame かが変わるため、両方に対応する。
    戻り値: (date または None, エラーメッセージ または None)
    """
    code = ticker_code.strip()
    if not code:
        return None, "ティッカーコードを入力してください。"

    yf_symbol = code if "." in code else f"{code}.T"

    try:
        ticker_obj = make_yf_ticker(yf_symbol)
        calendar = ticker_obj.calendar
    except Exception as e:
        return None, f"取得処理でエラーが発生しました（{e}）。インターネット接続をご確認ください。"

    if calendar is None or (hasattr(calendar, "empty") and calendar.empty) or calendar == {}:
        return None, f"「{yf_symbol}」の決算スケジュール情報が見つかりませんでした。"

    earnings_dates = None
    if isinstance(calendar, dict):
        earnings_dates = calendar.get("Earnings Date")
    else:
        try:
            if "Earnings Date" in calendar.index:
                earnings_dates = calendar.loc["Earnings Date"].dropna().tolist()
        except Exception:
            earnings_dates = None

    if not earnings_dates:
        return None, f"「{yf_symbol}」の次回決算発表予定日はまだ公表されていないようです。"

    next_date = earnings_dates[0] if isinstance(earnings_dates, (list, tuple)) else earnings_dates
    if hasattr(next_date, "date"):
        next_date = next_date.date()

    return next_date, None

def calc_cagr(start_val, end_val, years):
    """売上などのCAGR（年平均成長率）を計算する"""
    if start_val is None or end_val is None or start_val <= 0 or years is None or years <= 0:
        return None
    return (end_val / start_val) ** (1 / years) - 1

# ==========================================
# カンバンビュー用：株価一括取得 & カードHTML
# ==========================================
@st.cache_data(ttl=300)  # 5分キャッシュ
def fetch_prices_batch(tickers: tuple) -> dict:
    """
    複数銘柄の現在株価・前日比をyf.download()で一括取得する。
    ttl=300なので5分ごとに自動更新。手動更新は fetch_prices_batch.clear() で。
    戻り値: {ティッカー: {"price": float, "change_pct": float}}
    """
    if not tickers:
        return {}

    symbol_map = {(t if "." in t else f"{t}.T"): t for t in tickers}
    symbols = list(symbol_map.keys())
    result = {}

    try:
        data = yf.download(symbols, period="2d", interval="1d",
                            auto_adjust=True, progress=False)
        if data.empty:
            return {}

        close = data["Close"]
        if len(symbols) == 1:
            close = close.to_frame(name=symbols[0])

        for symbol, orig in symbol_map.items():
            if symbol not in close.columns:
                continue
            prices = close[symbol].dropna()
            if len(prices) >= 1:
                result[orig] = {
                    "price": float(prices.iloc[-1]),
                    "change_pct": float(
                        (prices.iloc[-1] - prices.iloc[-2]) / prices.iloc[-2] * 100
                    ) if len(prices) >= 2 else 0.0
                }
    except Exception:
        pass

    return result

def make_card_html(ticker: str, name: str, per: str, cagr: str,
                   net_cash: str, price=None, change_pct=None, memo: str = "") -> str:
    """銘柄カード1枚分のHTMLを生成する"""
    if price is not None:
        color = "#2ecc71" if change_pct >= 0 else "#e74c3c"
        sign  = "+" if change_pct >= 0 else ""
        price_html = (
            f'<div style="margin:4px 0">'
            f'<span style="font-size:1.05em;font-weight:bold">¥{price:,.0f}</span>'
            f'<span style="color:{color};font-size:0.82em;margin-left:6px">'
            f'{sign}{change_pct:.1f}%</span></div>'
        )
    else:
        price_html = '<div style="color:#888;font-size:0.78em;margin:4px 0">株価未取得</div>'

    badges = "".join(
        f'<span style="font-size:0.72em;background:#2a2a3e;padding:2px 6px;'
        f'border-radius:4px;margin-right:4px">{label}</span>'
        for label, val in [("PER", per), ("CAGR", cagr), ("💰", net_cash)]
        if val
    )

    memo_snippet = ""
    if memo:
        short = memo[:45] + ("…" if len(memo) > 45 else "")
        memo_snippet = (
            f'<div style="font-size:0.73em;color:#888;margin-top:5px;'
            f'line-height:1.4">{short}</div>'
        )

    return (
        f'<div style="background:#1e1e2e;border:1px solid #333;border-radius:8px;'
        f'padding:10px 12px;margin-bottom:8px">'
        f'<div style="font-size:0.72em;color:#888">{ticker}</div>'
        f'<div style="font-weight:bold;font-size:0.92em;margin:2px 0">{name}</div>'
        f'{price_html}'
        f'<div style="margin-top:5px;display:flex;flex-wrap:wrap;gap:3px">{badges}</div>'
        f'{memo_snippet}'
        f'</div>'
    )

# ==========================================
# 個別銘柄レポート（Markdown）の保存・バックアップ
# ==========================================
def safe_ticker_filename(ticker_code: str) -> str:
    """ファイル名として安全な文字だけに絞る（パス区切り文字などを除去）"""
    return "".join(c for c in str(ticker_code) if c.isalnum() or c in ("-", "_"))

# ==========================================
# 日付・四半期ごとの複数エントリ管理（1エントリ=1ファイルのMarkdown方式）
# ==========================================
def entry_filepath(ticker_code: str, kind: str, key: str) -> str:
    """
    エントリ1件分のファイルパスを返す。
    例: reports/7974_analysis_2026-05-14.md
        reports/7974_earnings_2026-4Q.md
    keyに使えない文字（/など）はハイフンに置換する。
    """
    safe_name = safe_ticker_filename(ticker_code)
    safe_key = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in key)
    return os.path.join(REPORT_DIR, f"{safe_name}_{kind}_{safe_key}.md")

def load_entries(ticker_code: str, kind: str) -> list:
    """
    reports/ フォルダから <ticker>_<kind>_<key>.md を検索して
    新しい順（key降順）で返す。
    各エントリは {"key": "2026-05-14", "content": "...", "updated_at": "..."} の形式。
    """
    safe_name = safe_ticker_filename(ticker_code)
    prefix = f"{safe_name}_{kind}_"
    entries = []
    for fname in os.listdir(REPORT_DIR):
        if fname.startswith(prefix) and fname.endswith(".md"):
            fpath = os.path.join(REPORT_DIR, fname)
            key = fname[len(prefix):-3]  # プレフィックスと .md を除いた部分
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    content = f.read()
                mtime = os.path.getmtime(fpath)
                updated_at = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
                entries.append({"key": key, "content": content, "updated_at": updated_at})
            except OSError:
                pass
    entries.sort(key=lambda e: e["key"], reverse=True)
    return entries

def save_entry(ticker_code: str, kind: str, key: str, content: str) -> None:
    """
    1エントリ=1ファイルとして保存する。
    上書き前に _backup/ にタイムスタンプ付きで退避する。
    """
    fpath = entry_filepath(ticker_code, kind, key)

    if os.path.exists(fpath):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        bname = os.path.splitext(os.path.basename(fpath))[0]
        backup_path = os.path.join(REPORT_BACKUP_DIR, f"{bname}_{timestamp}.md")
        try:
            with open(fpath, "r", encoding="utf-8") as f_old:
                old = f_old.read()
            with open(backup_path, "w", encoding="utf-8") as f_bk:
                f_bk.write(old)
        except Exception:
            pass

    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)

def delete_entry(ticker_code: str, kind: str, key: str) -> None:
    """エントリファイルを削除する（削除前に _backup/ に退避）"""
    fpath = entry_filepath(ticker_code, kind, key)
    if not os.path.exists(fpath):
        return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bname = os.path.splitext(os.path.basename(fpath))[0]
    backup_path = os.path.join(REPORT_BACKUP_DIR, f"{bname}_{timestamp}_deleted.md")
    try:
        with open(fpath, "r", encoding="utf-8") as f:
            content = f.read()
        with open(backup_path, "w", encoding="utf-8") as f_bk:
            f_bk.write(content)
    except Exception:
        pass
    os.remove(fpath)

def migrate_legacy_report_if_needed(ticker_code: str) -> None:
    """
    旧形式ファイルが残っていれば新形式に自動移行する。
    - reports/<ticker>.md          → analysis エントリ（ファイル更新日を日付キーに）
    - reports/<ticker>_analysis.json → 各エントリを個別 .md に展開
    - reports/<ticker>_earnings.json → 各エントリを個別 .md に展開
    """
    safe_name = safe_ticker_filename(ticker_code)

    # 旧①: 単一 .md ファイル
    legacy_md = os.path.join(REPORT_DIR, f"{safe_name}.md")
    if os.path.exists(legacy_md):
        with open(legacy_md, "r", encoding="utf-8") as f:
            content = f.read()
        if content.strip():
            mtime = os.path.getmtime(legacy_md)
            date_key = datetime.datetime.fromtimestamp(mtime).strftime("%Y-%m-%d")
            dest = entry_filepath(ticker_code, "analysis", date_key)
            if not os.path.exists(dest):
                save_entry(ticker_code, "analysis", date_key, content)
        os.rename(legacy_md, legacy_md + ".migrated")

    # 旧②③: JSON ファイル
    for kind in ("analysis", "earnings"):
        json_path = os.path.join(REPORT_DIR, f"{safe_name}_{kind}.json")
        if not os.path.exists(json_path):
            continue
        try:
            with open(json_path, "r", encoding="utf-8") as f:
                old_entries = json.load(f)
            for e in old_entries:
                key = e.get("key", "unknown")
                dest = entry_filepath(ticker_code, kind, key)
                if not os.path.exists(dest):
                    save_entry(ticker_code, kind, key, e.get("content", ""))
            os.rename(json_path, json_path + ".migrated")
        except Exception:
            pass

def create_reports_zip() -> io.BytesIO:
    """reports/ 内の .md ファイルをまとめてZIP化する（_backup/ は除く）"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        for fname in os.listdir(REPORT_DIR):
            fpath = os.path.join(REPORT_DIR, fname)
            if os.path.isfile(fpath) and fname.endswith(".md"):
                zf.write(fpath, arcname=fname)
    buf.seek(0)
    return buf

# ==========================================
# GitHub連携：レポートのMarkdownを自動コミット
# ==========================================
def get_github_config():
    """後方互換用ラッパー。gh.config を返す。"""
    return gh.config

@st.cache_data(ttl=120)
def github_fetch_entries(token: str, repo: str, branch: str, ticker_code: str, kind: str) -> list:
    """GitHubのreports/フォルダから <ticker>_<kind>_<key>.md を検索して新しい順で返す。"""
    safe_name = safe_ticker_filename(ticker_code)
    prefix = f"{safe_name}_{kind}_"
    # gh.list_filesを使う（GitHubStorageに集約）
    items = gh.list_files("reports")
    entries = []
    for item in items:
        fname = item.get("name", "")
        if not fname.startswith(prefix) or not fname.endswith(".md"):
            continue
        key = fname[len(prefix):-3]
        entries.append({
            "key": key,
            "path": item.get("path", ""),
            "sha": item.get("sha", ""),
            "download_url": item.get("download_url", ""),
        })
    entries.sort(key=lambda e: e["key"], reverse=True)
    return entries

@st.cache_data(ttl=120)
def github_fetch_content(token: str, download_url: str) -> str:
    """GitHubからMarkdownの本文を取得する。Privateリポジトリ対応。"""
    return gh.read_binary_url(download_url) or ""

def github_delete_file(token: str, repo: str, branch: str, path: str, sha: str, message: str) -> tuple[bool, str]:
    """GitHubのファイルを削除する（GitHubStorageに委譲）。"""
    return gh.delete_file(path, message)

def github_upload_report(file_stem: str, content: str, company_name: str = "", extension: str = "md") -> tuple[bool, str]:
    """レポートをGitHubに保存する（GitHubStorageに委譲）。"""
    if not gh.enabled:
        return False, "GitHub未設定"
    safe_name = safe_ticker_filename(file_stem)
    path = f"reports/{safe_name}.{extension}"
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
    return gh.write_text(path, content, f"Update report: {file_stem}（{company_name}）- {timestamp}")

# st.session_state に持たせることで、フォーム送信などの再実行時にも
# 編集中のデータが消えないようにする
if "df" not in st.session_state:
    st.session_state.df = load_data()

# 起動時にGitHubのreports/をローカルに同期（1セッションに1度だけ）
sync_reports_from_github()

st.title("📊 銘柄管理ダッシュボード")
st.caption("Obsidian（Dataview / Templater）の代わりに、ブラウザ上で動く株式管理ダッシュボードです。")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📋 一覧・編集", "📝 新規銘柄登録", "🧮 売上CAGR計算", "📊 分析", "📑 個別銘柄レポート"
])

# ------------------------------------------
# タブ1：一覧表示 ＋ 編集・削除
# ------------------------------------------
with tab1:
    df = st.session_state.df
    st.subheader("保有・監視銘柄データ")

    if df.empty:
        st.info("現在登録されているデータがありません。「新規銘柄登録」タブから追加してください。")
    else:
        # --- ビュー切り替え ---
        view_mode = st.radio(
            "表示モード",
            ["📋 テーブル", "🗂️ カンバン"],
            horizontal=True,
            key="view_mode"
        )

        st.divider()

        # ==========================================
        # カンバンビュー
        # ==========================================
        if view_mode == "🗂️ カンバン":
            col_kb1, col_kb2 = st.columns([3, 1])
            with col_kb1:
                st.caption("ステータスごとに銘柄カードを表示します。カード下のボタンでステータスを変更できます。")
            with col_kb2:
                if st.button("🔄 株価を更新", key="kanban_refresh"):
                    fetch_prices_batch.clear()
                    st.rerun()

            # 全銘柄の株価を一括取得（5分キャッシュ）
            all_tickers = tuple(df["ティッカー"].astype(str).tolist())
            with st.spinner("株価を取得中..."):
                prices = fetch_prices_batch(all_tickers)

            # ステータス順に列を並べる
            kanban_columns = [s for s in STATUS_OPTIONS if s in df["ステータス"].values]
            cols = st.columns(len(kanban_columns))

            STATUS_COLORS = {
                "監視中":  "#3498db",
                "打診買い": "#f39c12",
                "保有中":  "#2ecc71",
                "見送り":  "#95a5a6",
            }

            for col_ui, status in zip(cols, kanban_columns):
                with col_ui:
                    color = STATUS_COLORS.get(status, "#888")
                    status_df = df[df["ステータス"] == status]
                    st.markdown(
                        f'<div style="border-top:3px solid {color};padding-top:6px;'
                        f'margin-bottom:8px;font-weight:bold;font-size:0.95em">'
                        f'{status} <span style="color:#888;font-size:0.8em">({len(status_df)})</span></div>',
                        unsafe_allow_html=True
                    )

                    for _, row in status_df.iterrows():
                        ticker = str(row["ティッカー"])
                        p = prices.get(ticker, {})

                        # カードHTML表示
                        st.markdown(
                            make_card_html(
                                ticker=ticker,
                                name=str(row["銘柄名"]),
                                per=str(row["PER"]),
                                cagr=str(row["売上5y CAGR"]),
                                net_cash=str(row["ネットキャッシュ"]),
                                price=p.get("price"),
                                change_pct=p.get("change_pct"),
                                memo=str(row["投資家メモ"]),
                            ),
                            unsafe_allow_html=True
                        )

                        # ステータス変更ボタン（隣のステータスへ移動）
                        next_statuses = [s for s in STATUS_OPTIONS if s != status]
                        btn_cols = st.columns(len(next_statuses))
                        for bc, next_s in zip(btn_cols, next_statuses):
                            short = next_s[:3]  # ボタンを短く
                            if bc.button(f"→{short}", key=f"kanban_mv_{ticker}_{next_s}",
                                         help=f"{next_s}へ移動"):
                                idx = st.session_state.df[
                                    st.session_state.df["ティッカー"].astype(str) == ticker
                                ].index
                                if len(idx) > 0:
                                    st.session_state.df.at[idx[0], "ステータス"] = next_s
                                    st.session_state.df.at[idx[0], "更新日"] = \
                                        datetime.datetime.now().strftime("%Y-%m-%d %H:%M")
                                    save_data(st.session_state.df)
                                    st.rerun()

        # ==========================================
        # テーブルビュー（従来通り）
        # ==========================================
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

            st.dataframe(view_df, width='stretch', hide_index=True)

            st.divider()

            # --- 編集・削除エリア ---
            st.markdown("##### ✏️ データの編集・削除")
            st.caption(
                "セルをダブルクリックすると直接編集できます。行頭にチェックを入れて削除、"
                "下に空欄の行が出たら新規行として入力もできます。編集後は必ず「変更を保存」を押してください。"
            )

            edited_df = st.data_editor(
                df,
                width='stretch',
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

    # 自動取得した値をクリアするフラグ処理（ウィジェット生成前に実行する必要がある）
    if st.session_state.get("reset_new_form", False):
        for k in ["new_ticker", "reg_name", "reg_per", "reg_cagr",
                  "reg_forecast", "reg_memo"]:
            st.session_state[k] = ""
        st.session_state.reg_netcash = "不明"
        st.session_state.reset_new_form = False

    # reg_netcashのデフォルト値を初期化（selectboxのkey管理のため）
    if "reg_netcash" not in st.session_state:
        st.session_state.reg_netcash = "不明"

    st.markdown("##### ① ティッカーコードを入力して自動取得（任意）")
    col_t1, col_t2 = st.columns([2, 1])
    with col_t1:
        ticker = st.text_input("ティッカーコード (例: 7974)", key="new_ticker")
    with col_t2:
        st.write("")
        fetch_clicked = st.button("🔍 銘柄情報をWeb上から自動取得")

    if fetch_clicked:
        with st.spinner("Yahoo!ファイナンスから取得中..."):
            data, error = fetch_stock_info(ticker)
        if error:
            st.error(f"⚠️ {error}")
        else:
            # フォームのwidget keyを直接更新（value=引数では反映されないため）
            st.session_state.reg_name = data["name"]
            st.session_state.reg_per = data["per"]
            if data["net_cash"] in NETCASH_OPTIONS:
                st.session_state.reg_netcash = data["net_cash"]
            st.session_state.fetch_msg = {
                "name": data["name"], "has_name": bool(data["name"])
            }
            st.rerun()

    # 取得結果メッセージ（rerun後に表示）
    if "fetch_msg" in st.session_state:
        _fm = st.session_state.fetch_msg
        if _fm["has_name"]:
            st.success(
                f"✅「{_fm['name']}」の情報を取得しました。銘柄名・PER・ネットキャッシュを下のフォームに反映しています。"
                "セクターと売上関連の項目は自動取得の対象外なので、ご自身で入力してください。"
            )
        else:
            st.warning(
                "⚠️ 銘柄名は取得できませんでしたが、PER・ネットキャッシュは反映しています。"
                "銘柄名は下のフォームで手入力してください（yfinanceが日本語名を持っていない銘柄です）。"
            )
        del st.session_state.fetch_msg

    st.caption(
        "※ 銘柄名・PER・ネットキャッシュ（簡易判定）のみ自動取得します。"
        "セクター分類や売上CAGR・売上予想はWeb上の単純な数値取得では精度が出ないため、手動入力のままにしています。"
    )

    st.divider()
    st.markdown("##### ② 内容を確認して登録")

    with st.container():
        col1, col2 = st.columns(2)

        with col1:
            st.text_input("ティッカーコード（①で入力した値）", value=ticker, disabled=True, key="reg_ticker_display")
            name = st.text_input("銘柄名", key="reg_name")
            sector = st.selectbox("セクター", SECTOR_OPTIONS, key="reg_sector")
            status = st.selectbox("ステータス", STATUS_OPTIONS, key="reg_status")

        with col2:
            cagr = st.text_input("売上5y CAGR (例: 15.2%) ※自動取得対象外", key="reg_cagr")
            forecast = st.text_input("売上予想 (例: 今期+10%成長) ※自動取得対象外", key="reg_forecast")
            per = st.text_input("PER (例: 15.5)", key="reg_per")
            net_cash = st.selectbox(
                "ネットキャッシュ", NETCASH_OPTIONS,
                key="reg_netcash"
            )

        memo = st.text_area("投資家メモ (決算の所感、チャートの形状、カタリストなど)", key="reg_memo")

        submitted = st.button("💾 データベースに登録", type="primary", key="reg_submit")

        if submitted:
            if ticker == "":
                st.error("⚠️ ①でティッカーコードを入力してください！")
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
                st.session_state.reset_new_form = True
                st.rerun()

# ------------------------------------------
# タブ3：売上CAGR計算
# ------------------------------------------
with tab3:
    st.subheader("🧮 売上CAGR計算")
    st.caption(
        "売上5y CAGR（年平均成長率）を計算するためのタブです。"
        "ティッカーからの自動取得、またはIR資料などの数値を手入力して計算できます。"
    )

    calc_mode = st.radio(
        "計算方法",
        ["🌐 ティッカーから自動取得", "✍️ 手入力で計算"],
        horizontal=True
    )

    st.divider()

    if calc_mode == "🌐 ティッカーから自動取得":
        col_c1, col_c2 = st.columns([2, 1])
        with col_c1:
            cagr_ticker = st.text_input("ティッカーコード (例: 7974)", key="cagr_ticker")
        with col_c2:
            st.write("")
            cagr_fetch_clicked = st.button("📊 決算データを取得")

        if cagr_fetch_clicked:
            with st.spinner("決算データを取得中..."):
                df_rev, error = fetch_revenue_history(cagr_ticker)

            if error:
                st.error(f"⚠️ {error}")
                st.info(
                    "Yahoo!ファイナンスの無料データは小型株や新興企業では取得できない場合があります。"
                    "その場合は「手入力で計算」をお試しください。"
                )
            else:
                st.markdown("##### 年度別 売上高")
                st.dataframe(
                    df_rev[["決算期", "売上高（百万）"]],
                    hide_index=True, width='stretch'
                )

                fig_rev = px.bar(
                    df_rev, x="決算期", y="売上高（百万）",
                    title="年度別 売上高の推移", text="売上高（百万）"
                )
                st.plotly_chart(fig_rev, width='stretch')

                start_val = df_rev["_raw"].iloc[0]
                end_val = df_rev["_raw"].iloc[-1]
                start_date = df_rev["_date"].iloc[0]
                end_date = df_rev["_date"].iloc[-1]
                years_span = (end_date - start_date).days / 365.25

                cagr_value = calc_cagr(start_val, end_val, years_span)

                n_periods = len(df_rev)
                if n_periods < 6:
                    st.warning(
                        f"⚠️ Yahoo!ファイナンスから取得できたのは直近 {n_periods} 期分（約{years_span:.1f}年分）のデータです。"
                        "「5年」CAGRとしてはやや短いので、ラベルや精度には注意してください。"
                    )

                if cagr_value is not None:
                    st.metric(
                        f"実質 約{years_span:.1f}年 CAGR",
                        f"{cagr_value:+.1%}"
                    )
                    st.caption("👇 この値を「新規銘柄登録」タブの『売上5y CAGR』欄にコピーして使ってください")
                    st.code(f"{cagr_value*100:.1f}%", language=None)
                else:
                    st.error("CAGRを計算できませんでした（データが不正、または期間が0年です）。")

    else:
        st.markdown("##### IR資料・決算短信などの数値を入力してください")
        col_m1, col_m2, col_m3 = st.columns(3)
        with col_m1:
            start_rev = st.number_input(
                "5年前（または開始期）の売上高", min_value=0.0, value=0.0, step=1.0,
                help="単位は百万円・億円など何でもOKです（開始と終了で揃えてください）"
            )
        with col_m2:
            end_rev = st.number_input(
                "直近期の売上高", min_value=0.0, value=0.0, step=1.0
            )
        with col_m3:
            n_years_manual = st.number_input(
                "年数", min_value=1, max_value=20, value=5, step=1
            )

        if st.button("🧮 CAGRを計算", type="primary"):
            cagr_manual = calc_cagr(start_rev, end_rev, n_years_manual)
            if cagr_manual is None:
                st.error("⚠️ 売上高は0より大きい値を入力してください。")
            else:
                st.metric(f"{n_years_manual}年 CAGR", f"{cagr_manual:+.1%}")
                st.caption("👇 この値を「新規銘柄登録」タブの『売上5y CAGR』欄にコピーして使ってください")
                st.code(f"{cagr_manual*100:.1f}%", language=None)

    # ------------------------------------------
    # ネットキャッシュ / NCR / PER(CN) / DPUP 計算
    # ------------------------------------------
    st.divider()
    st.subheader("💰 ネットキャッシュ・DPUP計算")
    st.caption(
        "貸借対照表の数値（百万円単位）を入力すると、ネットキャッシュ・NCR・PER(CN)・DPUPを一括計算します。"
        "ティッカーを入力してyfinanceから自動取得することもできます（精度は保証できません）。"
    )

    # --- ティッカー自動取得 ---
    col_nc_t1, col_nc_t2 = st.columns([2, 1])
    with col_nc_t1:
        nc_ticker = st.text_input("ティッカーコード（自動取得する場合）", key="nc_ticker",
                                   placeholder="例: 7974（空欄の場合は手入力）")
    with col_nc_t2:
        st.write("")
        nc_fetch = st.button("🔍 財務データを自動取得", key="nc_fetch_btn")

    if nc_fetch and nc_ticker.strip():
        yf_sym = nc_ticker.strip()
        if "." not in yf_sym:
            yf_sym = f"{yf_sym}.T"
        with st.spinner("yfinanceから財務データを取得中..."):
            try:
                _t = make_yf_ticker(yf_sym)

                _bs = _t.get_balance_sheet()
                _is = _t.get_income_stmt()
                _info = _t.get_info() or {}

                def _get(_df, labels):
                    """複数のラベル候補から最初に見つかった値を返す（円→百万円）"""
                    if _df is None or _df.empty:
                        return None
                    for lbl in labels:
                        if lbl in _df.index:
                            v = _df.loc[lbl].dropna()
                            if not v.empty:
                                return float(v.iloc[0]) / 1e6
                    return None

                def _sum(_df, labels):
                    """複数ラベルの値を合算する（投資有価証券のように分散している項目用）"""
                    if _df is None or _df.empty:
                        return None
                    total = 0.0
                    found = False
                    for lbl in labels:
                        if lbl in _df.index:
                            v = _df.loc[lbl].dropna()
                            if not v.empty:
                                total += float(v.iloc[0])
                                found = True
                    return total / 1e6 if found else None

                # yfinance 1.5.xの実際のラベルはスペースなし（例: CurrentAssets）
                _ca  = _get(_bs, ["CurrentAssets", "TotalCurrentAssets"])
                _inv = _get(_bs, ["Inventory", "OtherInventories"])
                # 投資有価証券は複数科目に分散するため合算する
                _inv_sec = _sum(_bs, [
                    "AvailableForSaleSecurities", "LongTermEquityInvestment",
                    "HeldToMaturitySecurities", "InvestmentsAndAdvances",
                    "OtherInvestments", "InvestmentinFinancialAssets"])
                _liab = _get(_bs, ["TotalLiabilitiesNetMinorityInterest"])
                _eq   = _get(_bs, ["StockholdersEquity", "CommonStockEquity",
                                   "TotalEquityGrossMinorityInterest"])
                _debt = _get(_bs, ["TotalDebt", "LongTermDebt",
                                   "LongTermDebtAndCapitalLeaseObligation"])
                _ebit = _get(_is, ["EBIT", "OperatingIncome",
                                   "TotalOperatingIncomeAsReported"])
                _mcap = (_info.get("marketCap") or 0) / 1e6

                st.session_state.nc_prefill = {
                    "ca": _ca, "inv": _inv, "inv_sec": _inv_sec,
                    "liab": _liab, "eq": _eq, "debt": _debt,
                    "ebit": _ebit, "mcap": _mcap,
                }
                # number_inputのkeyに直接値を書き込む（既存ウィジェットを上書きするため）
                key_map = {
                    "nc_ca": _ca, "nc_inv": _inv, "nc_isec": _inv_sec,
                    "nc_liab": _liab, "nc_mcap": _mcap,
                    "nc_ebit": _ebit, "nc_eq": _eq, "nc_debt": _debt,
                }
                for wkey, wval in key_map.items():
                    if wval is not None:
                        st.session_state[wkey] = float(wval)

                labels_jp = {
                    "ca": "流動資産", "inv": "棚卸資産", "inv_sec": "投資有価証券",
                    "liab": "負債合計", "eq": "純資産", "debt": "有利子負債",
                    "ebit": "EBIT", "mcap": "時価総額",
                }
                got = [labels_jp[k] for k, v in st.session_state.nc_prefill.items() if v is not None]
                missing = [labels_jp[k] for k, v in st.session_state.nc_prefill.items() if v is None]
                st.session_state.nc_fetch_msg = {"got": got, "missing": missing}
                st.rerun()
            except Exception as e:
                st.error(f"⚠️ 取得に失敗しました（{e}）")

    # 取得結果メッセージ（rerun後に表示）
    if "nc_fetch_msg" in st.session_state:
        _msg = st.session_state.nc_fetch_msg
        if _msg["got"]:
            st.success(f"✅ 取得できた項目: {', '.join(_msg['got'])}（単位: 百万円）")
        if _msg["missing"]:
            st.warning(
                f"⚠️ 取得できなかった項目: {', '.join(_msg['missing'])}。"
                "yfinanceにデータがない項目です。IR資料を見て手入力してください。"
            )
        st.caption("⚠️ 特に投資有価証券はyfinanceのデータが実態と乖離しやすいので、IR資料で確認することをおすすめします。")
        del st.session_state.nc_fetch_msg

    # number_inputの初期値をsession_stateで管理（自動取得時に上書きされる）
    for _k in ["nc_ca", "nc_inv", "nc_isec", "nc_liab", "nc_mcap", "nc_per", "nc_ebit", "nc_eq", "nc_debt"]:
        if _k not in st.session_state:
            st.session_state[_k] = 0.0

    st.markdown("##### 入力（百万円単位）")
    col1, col2, col3 = st.columns(3)
    with col1:
        nc_ca   = st.number_input("流動資産",       step=1.0, key="nc_ca")
        nc_inv  = st.number_input("棚卸資産",       step=1.0, key="nc_inv")
    with col2:
        nc_isec = st.number_input("投資有価証券",   step=1.0, key="nc_isec")
        nc_liab = st.number_input("負債合計",       step=1.0, key="nc_liab")
    with col3:
        nc_mcap = st.number_input("時価総額",       step=1.0, key="nc_mcap")
        nc_per  = st.number_input("PER（倍）",     step=0.1, key="nc_per")

    st.markdown("##### ROIC計算用（yfinance自動取得 or 手入力）")
    col4, col5, col6 = st.columns(3)
    with col4:
        nc_ebit = st.number_input("EBIT（営業利益）",  step=1.0, key="nc_ebit")
    with col5:
        nc_eq   = st.number_input("純資産",            step=1.0, key="nc_eq")
    with col6:
        nc_debt = st.number_input("有利子負債",         step=1.0, key="nc_debt")

    if st.button("🧮 まとめて計算する", type="primary", key="nc_calc"):
        # ネットキャッシュ
        net_cash_val = nc_ca - (nc_inv * 0.3 + nc_isec * 0.7) - nc_liab

        # NCR
        ncr_val = (net_cash_val / nc_mcap * 100) if nc_mcap > 0 else None

        # PER(CN)
        per_cn_val = None
        if ncr_val is not None and ncr_val < 100 and nc_per > 0:
            per_cn_val = nc_per * (100 - ncr_val) / 100

        # ROIC
        invested_cap = nc_eq + nc_debt
        roic_val = (nc_ebit / invested_cap * 100) if invested_cap > 0 and nc_ebit > 0 else None

        # DPUP
        dpup_val = None
        if roic_val is not None and per_cn_val is not None and per_cn_val != 0:
            dpup_val = round(roic_val / per_cn_val, 2)

        st.divider()
        st.markdown("##### 📊 計算結果")

        r1, r2, r3 = st.columns(3)
        r1.metric("ネットキャッシュ（百万円）", f"{net_cash_val:,.0f}")
        r2.metric("NCR（%）", f"{ncr_val:.1f}%" if ncr_val is not None else "—")
        r3.metric("PER(CN)", f"{per_cn_val:.2f}" if per_cn_val is not None else "—")

        r4, r5, r6 = st.columns(3)
        r4.metric("ROIC（%）", f"{roic_val:.2f}%" if roic_val is not None else "—")
        r5.metric("DPUP", f"{dpup_val}" if dpup_val is not None else "—")
        r6.metric("投下資本（百万円）", f"{invested_cap:,.0f}" if invested_cap > 0 else "—")

        # 判定コメント
        if dpup_val is not None:
            if dpup_val >= 1.5:
                st.success(f"✅ DPUP {dpup_val} → 割安圏の目安（1.5以上）")
            elif dpup_val >= 1.0:
                st.info(f"ℹ️ DPUP {dpup_val} → やや割安（1.0〜1.5）")
            else:
                st.warning(f"⚠️ DPUP {dpup_val} → 割安感は薄い（1.0未満）")

        # コピー用サマリー
        st.caption("👇 メモ用にコピーできます")
        summary = (
            f"ネットキャッシュ: {net_cash_val:,.0f} 百万円\n"
            f"NCR: {f'{ncr_val:.1f}%' if ncr_val is not None else '—'}\n"
            f"PER(CN): {f'{per_cn_val:.2f}' if per_cn_val is not None else '—'}\n"
            f"ROIC: {f'{roic_val:.2f}%' if roic_val is not None else '—'}\n"
            f"DPUP: {dpup_val if dpup_val is not None else '—'}"
        )
        st.code(summary, language=None)

        if ncr_val is not None and ncr_val >= 100:
            st.info("ℹ️ NCR ≥ 100% のため PER(CN) は計算されません（ネットキャッシュが時価総額を超えています）。")

# ------------------------------------------
# タブ4：分析
# ------------------------------------------
with tab4:
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
            st.plotly_chart(fig_sector, width='stretch')

        with col_c2:
            status_counts = df["ステータス"].value_counts().reset_index()
            status_counts.columns = ["ステータス", "件数"]
            fig_status = px.bar(
                status_counts, x="ステータス", y="件数",
                title="ステータス別 銘柄数", text="件数"
            )
            st.plotly_chart(fig_status, width='stretch')

        if per_numeric.notna().sum() > 0:
            per_df = df.copy()
            per_df["PER（数値）"] = per_numeric
            per_df = per_df.dropna(subset=["PER（数値）"])
            fig_per = px.bar(
                per_df.sort_values("PER（数値）"),
                x="銘柄名", y="PER（数値）", color="セクター",
                title="銘柄別 PER"
            )
            st.plotly_chart(fig_per, width='stretch')

# ------------------------------------------
# タブ5：個別銘柄レポート（Markdown）
# ------------------------------------------
with tab5:
    df = st.session_state.df
    st.subheader("📑 個別銘柄レポート")
    st.caption(
        "銘柄ごとに、Markdown形式の詳細な分析メモ（決算所感、SWOT分析、AIレポートの貼り付けなど）を残せます。"
    )

    if df.empty:
        st.info("ダッシュボードに銘柄が登録されていません。まずは「新規銘柄登録」タブから追加してください。")
    else:
        # 表示名 → ティッカー の対応をdictで持つ（銘柄名に" : "が含まれても誤動作しないようにする）
        ticker_label_map = {
            f"{row['ティッカー']} : {row['銘柄名']}": str(row["ティッカー"])
            for _, row in df.iterrows()
        }
        selected_label = st.selectbox("レポートを表示する銘柄を選択してください", list(ticker_label_map.keys()))
        selected_ticker = ticker_label_map[selected_label]

        matched_rows = df[df["ティッカー"].astype(str) == selected_ticker]
        selected_row = matched_rows.iloc[0] if not matched_rows.empty else None

        st.divider()

        # --- 直近の決算スケジュール（カウントダウン） ---
        st.markdown("##### 📅 直近の決算スケジュール")
        col_e1, col_e2 = st.columns([1, 2])
        with col_e1:
            earnings_clicked = st.button("📅 決算スケジュールを取得", key=f"earnings_btn_{selected_ticker}")

        if earnings_clicked:
            with st.spinner("取得中..."):
                next_date, error = fetch_next_earnings_date(selected_ticker)
            if error:
                st.session_state[f"earnings_result_{selected_ticker}"] = ("error", error)
            else:
                st.session_state[f"earnings_result_{selected_ticker}"] = ("ok", next_date)

        result = st.session_state.get(f"earnings_result_{selected_ticker}")
        if result:
            status, value = result
            if status == "error":
                st.error(f"⚠️ {value}")
                st.caption("自動取得できなかったので、決算予定日が分かっていれば手動で入力してください。")

                manual_date = st.date_input(
                    "決算発表予定日（手動入力）",
                    value=None,
                    key=f"earnings_manual_date_{selected_ticker}"
                )
                if manual_date:
                    if st.button("この日付で確定する", key=f"earnings_manual_confirm_{selected_ticker}"):
                        st.session_state[f"earnings_result_{selected_ticker}"] = ("manual", manual_date)
                        st.rerun()
            else:
                today = datetime.date.today()
                delta = (value - today).days
                col_d1, col_d2 = st.columns(2)
                date_label = "次回決算発表予定日" if status == "ok" else "次回決算発表予定日（手動入力）"
                col_d1.metric(date_label, value.strftime("%Y-%m-%d"))
                if delta < 0:
                    col_d2.metric("状況", "発表済み／要確認")
                elif delta == 0:
                    col_d2.metric("残り日数", "本日！")
                else:
                    col_d2.metric("残り日数", f"{delta} 日")

                if status == "manual":
                    if st.button("🔄 やり直す（自動取得を再試行 / 手動入力をクリア）", key=f"earnings_reset_{selected_ticker}"):
                        st.session_state.pop(f"earnings_result_{selected_ticker}", None)
                        st.rerun()
        else:
            st.caption("ボタンを押すと、Yahoo!ファイナンスから次回決算発表予定日を取得します。")

        st.divider()

        # --- 基本データカード ---
        st.markdown("##### 📊 基本データ")
        if selected_row is not None:
            with st.container(border=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("ステータス", selected_row["ステータス"] or "—")
                c1.metric("コード", selected_row["ティッカー"])
                c2.metric("セクター", selected_row["セクター"] or "—")
                c2.metric("PER", selected_row["PER"] or "—")
                c3.metric("売上5y CAGR", selected_row["売上5y CAGR"] or "—")
                c3.metric("ネットキャッシュ", selected_row["ネットキャッシュ"] or "—")
                if selected_row["投資家メモ"]:
                    st.caption(f"💬 投資家メモ：{selected_row['投資家メモ']}")

        st.divider()

        # --- AI分析用プロンプト ---
        st.markdown("##### 🤖 AI分析用プロンプト")
        st.caption("下のプロンプトをコピーして、ChatGPT・Gemini・Claudeなどに貼り付けて分析してもらってください。")

        include_context = st.checkbox(
            "ダッシュボードの登録データ（セクター・PER・売上5y CAGRなど）をプロンプトに含める",
            value=True,
            key=f"include_context_{selected_ticker}"
        )

        if selected_row is not None:
            company_name = selected_row["銘柄名"]
        else:
            company_name = ""

        if include_context and selected_row is not None:
            context_lines = "\n".join(
                f"- {col}: {selected_row[col]}"
                for col in ["ティッカー", "銘柄名", "セクター", "ステータス",
                            "売上5y CAGR", "売上予想", "PER", "ネットキャッシュ", "投資家メモ"]
            )
            full_prompt = (
                f"{ANALYST_PROMPT_TEMPLATE}\n\n"
                f"【ダッシュボード登録データ】\n{context_lines}\n\n"
                f"対象企業：{company_name}（{selected_ticker}）\n"
                f"※上記の登録データはあくまで現時点の記録値です。最新の決算短信・有価証券報告書・IR資料等を"
                f"自分で調査したうえで分析してください。"
            )
        else:
            full_prompt = f"{ANALYST_PROMPT_TEMPLATE}\n\n対象企業：{company_name}（{selected_ticker}）"

        st.code(full_prompt, language=None)

        st.divider()

        # --- 決算分析用プロンプト ---
        st.markdown("##### 📈 決算分析用プロンプト")
        st.caption(
            "決算発表のたびに使うプロンプトです。対象四半期・第一印象・メモを入力すると、"
            "決算データを埋め込んだプロンプトが生成されます。発表されたIR資料（PDFや画像）と一緒にAIへ渡してください。"
        )

        col_q1, col_q2 = st.columns(2)
        with col_q1:
            quarter_input = st.text_input(
                "対象四半期 (例: 2026-4Q)",
                key=f"earnings_quarter_{selected_ticker}"
            )
        with col_q2:
            impression_input = st.selectbox(
                "投資家の第一印象",
                ["ポジティブ", "中立", "ネガティブ"],
                index=1,
                key=f"earnings_impression_{selected_ticker}"
            )

        memo_input = st.text_area(
            "メモ（決算を見た直後の所感、印象的だった発言など）",
            key=f"earnings_memo_{selected_ticker}",
            height=80
        )

        earnings_prompt = EARNINGS_PROMPT_TEMPLATE.format(
            company_name=company_name or "（銘柄名未登録）",
            quarter=quarter_input or "（対象四半期を入力してください）",
            ticker=selected_ticker,
            impression=impression_input,
            memo=memo_input or "（特になし）"
        )

        st.code(earnings_prompt, language=None)

        st.divider()

        # GitHub連携の設定（両レポートセクションで共通利用）
        github_config = get_github_config()
        if not github_config:
            st.caption(
                "💡 GitHub連携は未設定です。SecretsにGITHUB_TOKEN / GITHUB_REPOを設定すると、"
                "再起動してもレポートが消えなくなります。"
            )

        # 旧形式（1銘柄1ファイル）が残っていれば、新形式（日付ごとの複数エントリ）に自動移行
        migrate_legacy_report_if_needed(selected_ticker)

        def load_entries_smart(ticker: str, kind: str) -> list:
            """
            GitHub設定があればGitHubから直接読み込む。
            なければローカルの reports/ フォルダから読み込む（フォールバック）。
            GitHub読み込み成功時は、ローカルにも同期してキャッシュとして残す。
            """
            if github_config:
                gh_entries = github_fetch_entries(
                    github_config["token"], github_config["repo"],
                    github_config["branch"], ticker, kind
                )
                # 本文をGitHubから取得してローカルにキャッシュ
                full_entries = []
                for e in gh_entries:
                    content = github_fetch_content(github_config["token"], e["download_url"])
                    # ローカルにも書き込んで再起動後の一瞬のギャップを埋める
                    local_path = entry_filepath(ticker, kind, e["key"])
                    if content and not os.path.exists(local_path):
                        try:
                            with open(local_path, "w", encoding="utf-8") as f:
                                f.write(content)
                        except OSError:
                            pass
                    full_entries.append({
                        "key": e["key"],
                        "content": content,
                        "updated_at": e.get("updated_at", ""),
                        "_sha": e["sha"],
                        "_path": e["path"],
                    })
                return full_entries
            else:
                return load_entries(ticker, kind)

        # ------------------------------------------
        # 銘柄分析レポート（日付ごと・複数エントリ・折りたたみ）
        # ------------------------------------------
        st.markdown("##### 📝 銘柄レポート（日付ごとに記録）")
        st.caption(
            "SWOT分析などのAIレポートを、日付ごとに複数貼り付けて記録できます。"
            "後から見返して『あの時はこう思っていたが…』を振り返るのに使ってください。"
        )

        analysis_entries = load_entries_smart(selected_ticker, "analysis")

        with st.expander("➕ 新しい銘柄レポートを追加", expanded=(len(analysis_entries) == 0)):
            new_analysis_date = st.date_input(
                "記録する日付",
                value=datetime.date.today(),
                key=f"analysis_new_date_{selected_ticker}"
            )
            new_analysis_content = st.text_area(
                "Markdown形式のレポート本文",
                height=400,
                placeholder="# 銘柄分析レポート\n\nここに分析結果やSWOT分析を貼り付けます...",
                key=f"analysis_new_content_{selected_ticker}"
            )

            if github_config:
                st.caption(f"保存時にGitHub（`{github_config['repo']}`）にも自動コミットされます。")

            if st.button("💾 このレポートを保存する", type="primary", key=f"analysis_save_{selected_ticker}"):
                if not new_analysis_content.strip():
                    st.error("⚠️ レポート本文が空です。")
                else:
                    date_key = new_analysis_date.strftime("%Y-%m-%d")
                    save_entry(selected_ticker, "analysis", date_key, new_analysis_content)
                    st.success(f"✅ {date_key} のレポートを保存しました！")

                    if github_config:
                        with st.spinner("GitHubにも保存中..."):
                            file_stem = f"{safe_ticker_filename(selected_ticker)}_analysis_{date_key}"
                            gh_ok, gh_message = github_upload_report(
                                file_stem, new_analysis_content, company_name
                            )
                        if gh_ok:
                            st.success(f"✅ {gh_message}")
                            github_fetch_entries.clear()
                            github_fetch_content.clear()
                        else:
                            st.warning(f"⚠️ ローカル保存は成功しましたが、GitHubへの保存に失敗しました：{gh_message}")

                    st.rerun()

        if analysis_entries:
            st.caption(f"📚 記録済み: {len(analysis_entries)} 件（新しい順）")
            for entry in analysis_entries:
                with st.expander(f"📅 {entry['key']}（最終更新: {entry.get('updated_at', '不明')}）"):
                    st.markdown(entry["content"])
                    col_ae1, col_ae2 = st.columns([1, 1])
                    with col_ae1:
                        st.download_button(
                            "⬇️ ダウンロード",
                            data=entry["content"].encode("utf-8"),
                            file_name=f"{safe_ticker_filename(selected_ticker)}_{entry['key']}.md",
                            mime="text/markdown",
                            key=f"analysis_dl_{selected_ticker}_{entry['key']}"
                        )
                    with col_ae2:
                        if st.button("🗑️ このエントリを削除", key=f"analysis_del_{selected_ticker}_{entry['key']}"):
                            delete_entry(selected_ticker, "analysis", entry["key"])
                            if github_config and entry.get("_sha") and entry.get("_path"):
                                with st.spinner("GitHubからも削除中..."):
                                    gh_ok, gh_msg = github_delete_file(
                                        github_config["token"], github_config["repo"],
                                        github_config["branch"], entry["_path"], entry["_sha"],
                                        f"Delete report: {entry['_path']}"
                                    )
                                if gh_ok:
                                    github_fetch_entries.clear()
                                    github_fetch_content.clear()
                                else:
                                    st.warning(f"⚠️ ローカルからは削除しましたが、GitHubからの削除に失敗しました：{gh_msg}")
                            st.success(f"{entry['key']} のレポートを削除しました。")
                            st.rerun()
        else:
            st.info("まだこの銘柄の銘柄レポートは登録されていません。上の「➕ 新しい銘柄レポートを追加」から登録してください。")

        st.divider()

        # ------------------------------------------
        # 決算分析レポート（四半期ごと・複数エントリ・折りたたみ）
        # ------------------------------------------
        st.markdown("##### 📈 決算分析レポート（四半期ごとに記録）")
        st.caption(
            "決算分析プロンプトの結果（進捗率分析・上方修正予測など）を、四半期ごとに分けて記録できます。"
        )

        earnings_entries = load_entries_smart(selected_ticker, "earnings")

        with st.expander("➕ 新しい決算分析レポートを追加", expanded=(len(earnings_entries) == 0)):
            new_earnings_quarter = st.text_input(
                "対象四半期（例: 2026-4Q）",
                value=quarter_input,
                key=f"earnings_new_quarter_{selected_ticker}"
            )
            new_earnings_content = st.text_area(
                "Markdown形式の決算分析レポート本文",
                height=400,
                placeholder="# 決算分析レポート\n\n進捗率分析、ポジティブ/ネガティブ要因などを貼り付けます...",
                key=f"earnings_new_content_{selected_ticker}"
            )

            if github_config:
                st.caption(f"保存時にGitHub（`{github_config['repo']}`）にも自動コミットされます。")

            if st.button("💾 この決算分析レポートを保存する", type="primary", key=f"earnings_save_{selected_ticker}"):
                if not new_earnings_quarter.strip():
                    st.error("⚠️ 対象四半期を入力してください（例: 2026-4Q）。")
                elif not new_earnings_content.strip():
                    st.error("⚠️ レポート本文が空です。")
                else:
                    quarter_key = new_earnings_quarter.strip()
                    save_entry(selected_ticker, "earnings", quarter_key, new_earnings_content)
                    st.success(f"✅ {quarter_key} の決算分析レポートを保存しました！")

                    if github_config:
                        with st.spinner("GitHubにも保存中..."):
                            safe_q = "".join(c if c.isalnum() or c in ("-", "_") else "-" for c in quarter_key)
                            file_stem = f"{safe_ticker_filename(selected_ticker)}_earnings_{safe_q}"
                            gh_ok, gh_message = github_upload_report(
                                file_stem, new_earnings_content, company_name
                            )
                        if gh_ok:
                            st.success(f"✅ {gh_message}")
                            github_fetch_entries.clear()
                            github_fetch_content.clear()
                        else:
                            st.warning(f"⚠️ ローカル保存は成功しましたが、GitHubへの保存に失敗しました：{gh_message}")

                    st.rerun()

        if earnings_entries:
            st.caption(f"📚 記録済み: {len(earnings_entries)} 件（新しい順）")
            for entry in earnings_entries:
                with st.expander(f"📊 {entry['key']}（最終更新: {entry.get('updated_at', '不明')}）"):
                    st.markdown(entry["content"])
                    col_ee1, col_ee2 = st.columns([1, 1])
                    with col_ee1:
                        st.download_button(
                            "⬇️ ダウンロード",
                            data=entry["content"].encode("utf-8"),
                            file_name=f"{safe_ticker_filename(selected_ticker)}_{entry['key']}.md",
                            mime="text/markdown",
                            key=f"earnings_dl_{selected_ticker}_{entry['key']}"
                        )
                    with col_ee2:
                        if st.button("🗑️ このエントリを削除", key=f"earnings_del_{selected_ticker}_{entry['key']}"):
                            delete_entry(selected_ticker, "earnings", entry["key"])
                            if github_config and entry.get("_sha") and entry.get("_path"):
                                with st.spinner("GitHubからも削除中..."):
                                    gh_ok, gh_msg = github_delete_file(
                                        github_config["token"], github_config["repo"],
                                        github_config["branch"], entry["_path"], entry["_sha"],
                                        f"Delete report: {entry['_path']}"
                                    )
                                if gh_ok:
                                    github_fetch_entries.clear()
                                    github_fetch_content.clear()
                                else:
                                    st.warning(f"⚠️ ローカルからは削除しましたが、GitHubからの削除に失敗しました：{gh_msg}")
                            st.success(f"{entry['key']} の決算分析レポートを削除しました。")
                            st.rerun()
        else:
            st.info("まだこの銘柄の決算分析レポートは登録されていません。上の「➕ 新しい決算分析レポートを追加」から登録してください。")

        st.divider()

        # ------------------------------------------
        # ニュース・クリップ（Obsidian Web Clipper貼り付け・日付ごと）
        # ------------------------------------------
        st.markdown("##### 📰 ニュース・クリップ（1記事=1エントリ）")
        st.caption(
            "株探などのニュース記事をObsidian Web Clipperでクリップした.mdをそのまま貼り付けて保存できます。"
            "1記事=1エントリとして日付ごとに蓄積されます。"
        )

        news_entries = load_entries_smart(selected_ticker, "news")

        with st.expander("➕ 新しいニュースを追加", expanded=(len(news_entries) == 0)):
            col_n1, col_n2 = st.columns([1, 1])
            with col_n1:
                new_news_date = st.date_input(
                    "記事の日付",
                    value=datetime.date.today(),
                    key=f"news_new_date_{selected_ticker}"
                )
            with col_n2:
                new_news_source = st.text_input(
                    "ソース（例: 株探、日経、IR）",
                    key=f"news_new_source_{selected_ticker}"
                )

            new_news_content = st.text_area(
                "Obsidian Web Clipperでクリップした内容をここに貼り付け",
                height=350,
                placeholder=(
                    "# 記事タイトル\n\n"
                    "source:: https://kabutan.jp/...\n\n"
                    "## 本文\n\nここにクリップした内容が入ります..."
                ),
                key=f"news_new_content_{selected_ticker}"
            )

            if github_config:
                st.caption(f"保存時にGitHub（`{github_config['repo']}`）にも自動コミットされます。")

            if st.button("💾 このニュースを保存する", type="primary", key=f"news_save_{selected_ticker}"):
                if not new_news_content.strip():
                    st.error("⚠️ 本文が空です。クリップした内容を貼り付けてください。")
                else:
                    date_key = new_news_date.strftime("%Y-%m-%d")
                    # 同じ日付に複数記事がある場合、時刻をサフィックスとして付与して区別する
                    existing_keys = {e["key"] for e in news_entries}
                    if date_key in existing_keys:
                        date_key = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")

                    # ソースをメタデータとして先頭行に追加
                    source_str = new_news_source.strip()
                    content_to_save = (
                        f"<!-- source: {source_str} -->\n\n{new_news_content}"
                        if source_str else new_news_content
                    )

                    save_entry(selected_ticker, "news", date_key, content_to_save)
                    st.success(f"✅ {date_key} のニュースを保存しました！")

                    if github_config:
                        with st.spinner("GitHubにも保存中..."):
                            file_stem = f"{safe_ticker_filename(selected_ticker)}_news_{date_key}"
                            gh_ok, gh_message = github_upload_report(
                                file_stem, content_to_save, company_name
                            )
                        if gh_ok:
                            st.success(f"✅ {gh_message}")
                            github_fetch_entries.clear()
                            github_fetch_content.clear()
                        else:
                            st.warning(f"⚠️ ローカル保存は成功しましたが、GitHubへの保存に失敗しました：{gh_message}")

                    st.rerun()

        if news_entries:
            st.caption(f"📚 記録済み: {len(news_entries)} 件（新しい順）")
            for entry in news_entries:
                # ソース名をコメントから抽出して表示名に使う
                source_display = ""
                first_line = entry["content"].split("\n")[0]
                if first_line.startswith("<!-- source:") and first_line.endswith("-->"):
                    source_display = first_line[len("<!-- source:"):][:-len("-->")].strip()

                label = f"📰 {entry['key']}"
                if source_display:
                    label += f"　{source_display}"
                label += f"（{entry.get('updated_at', '')}）"

                with st.expander(label):
                    # ソースコメント行は表示せず本文だけ表示
                    display_content = entry["content"]
                    if display_content.startswith("<!-- source:"):
                        display_content = "\n".join(display_content.split("\n")[2:])
                    st.markdown(display_content)

                    col_ne1, col_ne2 = st.columns([1, 1])
                    with col_ne1:
                        st.download_button(
                            "⬇️ ダウンロード",
                            data=entry["content"].encode("utf-8"),
                            file_name=f"{safe_ticker_filename(selected_ticker)}_news_{entry['key']}.md",
                            mime="text/markdown",
                            key=f"news_dl_{selected_ticker}_{entry['key']}"
                        )
                    with col_ne2:
                        if st.button("🗑️ このエントリを削除", key=f"news_del_{selected_ticker}_{entry['key']}"):
                            delete_entry(selected_ticker, "news", entry["key"])
                            if github_config and entry.get("_sha") and entry.get("_path"):
                                with st.spinner("GitHubからも削除中..."):
                                    gh_ok, gh_msg = github_delete_file(
                                        github_config["token"], github_config["repo"],
                                        github_config["branch"], entry["_path"], entry["_sha"],
                                        f"Delete news: {entry['_path']}"
                                    )
                                if gh_ok:
                                    github_fetch_entries.clear()
                                    github_fetch_content.clear()
                                else:
                                    st.warning(f"⚠️ ローカルからは削除しましたが、GitHubからの削除に失敗しました：{gh_msg}")
                            st.success(f"{entry['key']} のニュースを削除しました。")
                            st.rerun()
        else:
            st.info("まだこの銘柄のニュースは登録されていません。上の「➕ 新しいニュースを追加」から登録してください。")

        st.divider()

        # --- バックアップ ---
        st.markdown("##### 📦 バックアップ")
        report_files = [
            f for f in os.listdir(REPORT_DIR)
            if os.path.isfile(os.path.join(REPORT_DIR, f)) and (f.endswith(".md") or f.endswith(".json"))
        ]
        if report_files:
            st.download_button(
                f"📦 全レポートをZIPでダウンロード（{len(report_files)}件）",
                data=create_reports_zip(),
                file_name=f"reports_backup_{datetime.date.today()}.zip",
                mime="application/zip"
            )
        else:
            st.caption("まだ保存されたレポートがありません。")
from __future__ import annotations

import datetime
import html
import json
import os
import xml.etree.ElementTree as ET
from email.utils import parsedate_to_datetime
from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from portfolio_quality import build_company_news_query, is_relevant_company_news
from sidebar_nav import render_sidebar_navigation
from stock_action_status import evaluate_stock_action_status
from site_ui import render_v3_page_header, render_v3_section
from site_context import (
    get_active_stock,
    load_portfolio_stock_options,
    normalize_site_ticker,
    set_active_stock,
)


st.set_page_config(
    page_title="銘柄詳細｜たかさん日本株分析 V3",
    page_icon="🔬",
    layout="wide",
)
render_sidebar_navigation()

ME_FILES = [
    "data/multiple_expansion/me_screener_latest.csv",
    "data/multiple_expansion/me_universe_snapshot.csv",
    "data/multiple_expansion/mex_latest.csv",
]
SHORT_HISTORY_FILE = "data/short_cover_alert_history.csv"
ENTRY_STATUS_FILE = "data/short_cover_entry_status.json"
PORTFOLIO_FILE = "portfolio_data.csv"
REPORT_DIR = Path("reports")


def _safe_text(value, fallback="—") -> str:
    if value is None:
        return fallback
    try:
        if pd.isna(value):
            return fallback
    except Exception:
        pass
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return fallback
    return text


def _safe_float(value):
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _fmt_num(value, digits=1, suffix="") -> str:
    num = _safe_float(value)
    if num is None:
        return "—"
    return f"{num:.{digits}f}{suffix}"


def _fmt_pct_ratio(value, digits=1) -> str:
    num = _safe_float(value)
    if num is None:
        return "—"
    return f"{num * 100:.{digits}f}%"


def _fmt_pct_raw(value, digits=1) -> str:
    num = _safe_float(value)
    if num is None:
        return "—"
    return f"{num:.{digits}f}%"


@st.cache_data(ttl=300, show_spinner=False)
def _load_csv(path: str, dtype=None) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        return pd.read_csv(path, dtype=dtype)
    except UnicodeDecodeError:
        try:
            return pd.read_csv(path, dtype=dtype, encoding="utf-8-sig")
        except Exception:
            return pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def _ticker_mask(series: pd.Series, ticker: str) -> pd.Series:
    normalized = (
        series.astype(str)
        .str.replace(".0", "", regex=False)
        .str.replace(".T", "", regex=False)
        .str.strip()
    )
    return normalized == normalize_site_ticker(ticker)


@st.cache_data(ttl=300, show_spinner=False)
def _load_me_row(ticker: str) -> dict:
    code = normalize_site_ticker(ticker)
    for path in ME_FILES:
        frame = _load_csv(path, dtype={"ticker": str})
        if frame.empty or "ticker" not in frame.columns:
            continue
        matched = frame[_ticker_mask(frame["ticker"], code)].copy()
        if matched.empty:
            continue
        for date_col in ("trade_date", "data_asof", "available_date"):
            if date_col in matched.columns:
                matched["_sort_date"] = pd.to_datetime(
                    matched[date_col],
                    errors="coerce",
                )
                matched = matched.sort_values("_sort_date", ascending=False)
                break
        return matched.iloc[0].to_dict()
    return {}


@st.cache_data(ttl=300, show_spinner=False)
def _load_short_row(ticker: str) -> dict:
    frame = _load_csv(SHORT_HISTORY_FILE, dtype={"ticker": str})
    if frame.empty or "ticker" not in frame.columns:
        return {}
    matched = frame[_ticker_mask(frame["ticker"], ticker)].copy()
    if matched.empty:
        return {}
    matched["_alert_dt"] = pd.to_datetime(
        matched.get("alert_date"),
        errors="coerce",
    )
    matched["_updated_dt"] = pd.to_datetime(
        matched.get("last_updated"),
        errors="coerce",
    )
    matched = matched.sort_values(
        ["_alert_dt", "_updated_dt"],
        ascending=False,
    )
    return matched.iloc[0].to_dict()


@st.cache_data(ttl=120, show_spinner=False)
def _load_live_entry_row(ticker: str) -> dict:
    if not os.path.exists(ENTRY_STATUS_FILE):
        return {}
    try:
        with open(ENTRY_STATUS_FILE, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return {}

    code = normalize_site_ticker(ticker)
    for row in payload.get("rows", []) or []:
        if normalize_site_ticker(row.get("ticker", "")) == code:
            return dict(row)
    return {}


@st.cache_data(ttl=900, show_spinner=False)
def _price_snapshot(ticker: str) -> dict:
    symbol = ticker if "." in ticker else f"{ticker}.T"
    try:
        data = yf.download(
            symbol,
            period="10d",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = data.columns.get_level_values(0)
        data = data.dropna(subset=["Close"])
        if data.empty:
            return {}
        close = float(data["Close"].iloc[-1])
        previous = float(data["Close"].iloc[-2]) if len(data) >= 2 else None
        change = ((close / previous) - 1.0) * 100.0 if previous else None
        return {
            "price": close,
            "change_pct": change,
            "date": pd.Timestamp(data.index[-1]).strftime("%Y-%m-%d"),
        }
    except Exception:
        return {}


@st.cache_data(ttl=21600, show_spinner=False)
def _earnings_date(ticker: str):
    symbol = ticker if "." in ticker else f"{ticker}.T"
    try:
        calendar = yf.Ticker(symbol).calendar
        value = None
        if isinstance(calendar, dict):
            value = calendar.get("Earnings Date")
        elif isinstance(calendar, pd.DataFrame) and not calendar.empty:
            if "Earnings Date" in calendar.index:
                value = calendar.loc["Earnings Date"].iloc[0]
            elif "Earnings Date" in calendar.columns:
                value = calendar["Earnings Date"].iloc[0]

        if isinstance(value, (list, tuple)) and value:
            value = value[0]
        if value is None:
            return None
        ts = pd.Timestamp(value)
        return ts.date()
    except Exception:
        return None


@st.cache_data(ttl=1800, show_spinner=False)
def _fetch_news(ticker: str, company_name: str, max_items: int = 5) -> list[dict]:
    name = str(company_name or "").strip()
    if not name:
        return []

    try:
        query_text = build_company_news_query(ticker, name)
        url = (
            "https://news.google.com/rss/search?q="
            + quote(query_text)
            + "&hl=ja&gl=JP&ceid=JP:ja"
        )
        response = requests.get(
            url,
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        if response.status_code != 200:
            return []

        root = ET.fromstring(response.content)
        items = []
        for item in root.findall(".//item")[: max_items * 4]:
            title = (item.findtext("title") or "").strip()
            link = (item.findtext("link") or "").strip()
            pub_date_str = (item.findtext("pubDate") or "").strip()
            source_el = item.find("source")
            source = (
                source_el.text.strip()
                if source_el is not None and source_el.text
                else ""
            )
            if not is_relevant_company_news(title, ticker, name, source):
                continue

            date_text = ""
            if pub_date_str:
                try:
                    dt = parsedate_to_datetime(pub_date_str)
                    date_text = dt.strftime("%Y-%m-%d %H:%M")
                except Exception:
                    date_text = ""

            items.append(
                {
                    "title": title,
                    "link": link,
                    "source": source,
                    "date": date_text,
                }
            )
            if len(items) >= max_items:
                break
        return items
    except Exception:
        return []


def _load_portfolio_row(ticker: str) -> dict:
    frame = _load_csv(PORTFOLIO_FILE, dtype={"ティッカー": str})
    if frame.empty or "ティッカー" not in frame.columns:
        return {}
    matched = frame[_ticker_mask(frame["ティッカー"], ticker)]
    if matched.empty:
        return {}
    return matched.iloc[0].to_dict()


def _report_counts(ticker: str) -> tuple[int, int]:
    if not REPORT_DIR.exists():
        return 0, 0
    code = normalize_site_ticker(ticker)
    analysis = list(REPORT_DIR.glob(f"{code}_analysis_*.md"))
    earnings = list(REPORT_DIR.glob(f"{code}_earnings_*.md"))
    return len(analysis), len(earnings)


active_ticker, active_name, active_source = get_active_stock()
stock_options = load_portfolio_stock_options()

if not active_ticker and stock_options:
    active_ticker, active_name = stock_options[0]
    set_active_stock(active_ticker, active_name, source="銘柄詳細")

if not active_ticker:
    st.title("🔬 銘柄詳細")
    st.info(
        "表示する銘柄がまだ選ばれていません。"
        "左サイドバーの「現在の銘柄」で銘柄を選択してください。"
    )
    st.stop()

portfolio_row = _load_portfolio_row(active_ticker)
me_row = _load_me_row(active_ticker)
short_row = _load_short_row(active_ticker)
live_entry = _load_live_entry_row(active_ticker)

company_name = (
    _safe_text(portfolio_row.get("銘柄名"), "")
    or _safe_text(me_row.get("company_name"), "")
    or active_name
)
set_active_stock(active_ticker, company_name, source="銘柄詳細")

price = _price_snapshot(active_ticker)
earnings_date = _earnings_date(active_ticker)
analysis_count, earnings_count = _report_counts(active_ticker)
action_status = evaluate_stock_action_status(
    portfolio_row=portfolio_row,
    me_row=me_row,
    short_row=short_row,
    live_entry=live_entry,
    earnings_date=earnings_date,
)

render_v3_page_header(
    title=f"{company_name or active_ticker}｜{active_ticker}",
    subtitle="基本情報・FCF・ME・Entry・材料・レポートを1画面で確認し、必要な専門ツールへそのまま深掘りします。",
    stage="DETAIL",
    icon="🔬",
    badge="STOCK DETAIL HUB",
)

top1, top2, top3, top4, top5 = st.columns(5)
top1.metric(
    "現在値",
    "—" if not price else f"¥{price['price']:,.0f}",
    None if not price or price.get("change_pct") is None else f"{price['change_pct']:+.1f}%",
)
top2.metric("ステータス", _safe_text(portfolio_row.get("ステータス")))
top3.metric("PER", _safe_text(portfolio_row.get("PER")))
top4.metric("ME", _fmt_num(me_row.get("mex_score"), 0))
top5.metric(
    "Entry",
    _safe_text(
        live_entry.get("status")
        or short_row.get("alert_tier")
    ),
)

if price:
    st.caption(
        f"株価基準日: {price.get('date', '—')}"
        + (
            f" ｜ このページへは「{active_source}」から引き継ぎ"
            if active_source and active_source != "銘柄詳細"
            else ""
        )
    )

_status_palette = {
    "PRETRADE": ("#ecfdf3", "#067647", "#abefc6"),
    "ENTRY_MONITOR": ("#eff8ff", "#175cd3", "#b2ddff"),
    "MONITOR": ("#f5f3ff", "#6941c6", "#d9d6fe"),
    "RESEARCH": ("#fff7ed", "#b54708", "#fedf89"),
    "DATA_CHECK": ("#f8fafc", "#475467", "#d0d5dd"),
    "EVENT_CHECK": ("#fff1f3", "#c01048", "#fecdd6"),
}
_status_bg, _status_fg, _status_border = _status_palette.get(
    action_status.stage,
    ("#f8fafc", "#344054", "#d0d5dd"),
)
_reasons_text = " ｜ ".join(action_status.reasons) if action_status.reasons else "強い追い風シグナルはまだ未確認"
_blockers_text = " ｜ ".join(action_status.blockers) if action_status.blockers else "大きなブロッカーなし"

st.markdown(
    f"""
    <div style="
        margin:14px 0 12px 0;
        padding:16px 18px;
        border:1px solid {_status_border};
        border-radius:16px;
        background:{_status_bg};
    ">
      <div style="font-size:.72rem;font-weight:800;color:{_status_fg};letter-spacing:.08em">
        V3 ACTION STATUS
      </div>
      <div style="display:flex;align-items:baseline;gap:10px;flex-wrap:wrap;margin-top:3px">
        <div style="font-size:1.35rem;font-weight:850;color:{_status_fg}">
          {action_status.icon} {html.escape(action_status.label)}
        </div>
        <div style="font-size:.78rem;color:#667085;font-weight:700">
          Readiness {action_status.readiness_score}/100 ｜ Confidence {action_status.confidence}
        </div>
      </div>
      <div style="font-size:.86rem;color:#344054;margin-top:7px;line-height:1.65">
        {html.escape(action_status.summary)}
      </div>
      <div style="font-size:.76rem;color:#475467;margin-top:8px;line-height:1.6">
        <b>追い風:</b> {html.escape(_reasons_text)}<br>
        <b>確認事項:</b> {html.escape(_blockers_text)}
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

_action_col1, _action_col2 = st.columns([1, 3])
with _action_col1:
    if action_status.stage == "EVENT_CHECK":
        st.info("📰 下の「材料・イベント」タブを優先確認")
    else:
        st.page_link(
            action_status.next_page,
            label=action_status.next_label,
            icon=action_status.next_icon,
            use_container_width=True,
        )
with _action_col2:
    st.caption(
        "Readinessは売買推奨スコアではなく、Growth / FCF / ME / Entry / "
        "イベント確認がどこまで揃っているかを示す研究・監視の進行度です。"
    )

render_v3_section("次のアクション", "現在のステータスから専門画面へ深掘りします。", icon="🚦")
a1, a2, a3, a4, a5 = st.columns(5)
with a1:
    st.page_link(
        "pages/14_Multiple_Expansion_Hunter.py",
        label="ME Hunter",
        icon="🚀",
        use_container_width=True,
    )
with a2:
    st.page_link(
        "pages/8_Short_Cover_Hunter.py",
        label="Entry Hunter",
        icon="🎯",
        use_container_width=True,
    )
with a3:
    st.page_link(
        "pages/9_Pre_Trade_Check.py",
        label="購入前チェック",
        icon="🛡️",
        use_container_width=True,
    )
with a4:
    st.page_link(
        "dashboard_app.py",
        label="AI分析・保存",
        icon="🤖",
        use_container_width=True,
    )
with a5:
    st.page_link(
        "pages/📖_レポートビューア.py",
        label="レポート",
        icon="📖",
        use_container_width=True,
    )

overview_tab, engine_tab, entry_tab, event_tab, record_tab = st.tabs(
    [
        "📋 概要",
        "💵 FCF / ME",
        "🎯 Entry / 需給",
        "📰 材料・イベント",
        "📚 記録",
    ]
)

with overview_tab:
    render_v3_section("基本情報", "登録情報と基礎的な成長・財務指標を確認します。", icon="📋")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("セクター", _safe_text(portfolio_row.get("セクター") or me_row.get("sector33_name")))
    c2.metric("売上5y CAGR", _safe_text(portfolio_row.get("売上5y CAGR")))
    c3.metric("ROIC", _safe_text(portfolio_row.get("ROIC")))
    c4.metric("ネットキャッシュ", _safe_text(portfolio_row.get("ネットキャッシュ")))

    c5, c6, c7, c8 = st.columns(4)
    c5.metric("売上予想", _safe_text(portfolio_row.get("売上予想")))
    c6.metric("DPUP", _safe_text(portfolio_row.get("DPUP")))
    c7.metric("目標株価", _safe_text(portfolio_row.get("目標株価")))
    c8.metric("損切りライン", _safe_text(portfolio_row.get("損切りライン")))

    memo = _safe_text(portfolio_row.get("投資家メモ"), "")
    if memo:
        st.info(f"💬 投資家メモ: {memo}")

with engine_tab:
    render_v3_section("FCF / Quality", "企業価値のエンジンとキャッシュ創出力を確認します。", icon="💵")
    if not me_row:
        st.info("この銘柄は現在のME/FCFスナップショットにありません。")
    else:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("FCF Engine", _fmt_num(me_row.get("fcf_engine_score"), 1))
        f2.metric("Quality", _safe_text(me_row.get("fcf_quality_grade")))
        f3.metric("P/FCF", _fmt_num(me_row.get("p_fcf"), 1, "x"))
        f4.metric("FCF Coverage", _fmt_pct_ratio(me_row.get("fcf_engine_coverage")))

        f5, f6, f7, f8 = st.columns(4)
        f5.metric("FCF Margin", _fmt_pct_ratio(me_row.get("fcf_margin")))
        f6.metric("Cash Conversion", _fmt_num(me_row.get("cash_conversion"), 2, "x"))
        f7.metric("Operating Margin", _fmt_pct_ratio(me_row.get("operating_margin")))
        f8.metric("Net Cash Ratio", _fmt_pct_ratio(me_row.get("net_cash_ratio")))

        render_v3_section("Multiple Expansion", "市場評価の変化と再加速状態を確認します。", icon="🚀")
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("MEX Score", _fmt_num(me_row.get("mex_score"), 1))
        m2.metric("MEX Coverage", _fmt_pct_ratio(me_row.get("mex_coverage")))
        m3.metric("State", _safe_text(me_row.get("state_confirmed")))
        m4.metric("SW Decision", _safe_text(me_row.get("sw_decision")))

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Second Wave", _safe_text(me_row.get("second_wave_state")))
        m6.metric("RS Rating", _fmt_num(me_row.get("rs_rating"), 0))
        m7.metric("Volume Ratio", _fmt_num(me_row.get("volume_ratio"), 2, "x"))
        m8.metric("52W Confirm", "✅" if bool(me_row.get("confirm_52w")) else "—")

        st.caption(
            f"MEデータ基準: {_safe_text(me_row.get('data_asof') or me_row.get('trade_date'))}"
            f" ｜ Data Quality: {_safe_text(me_row.get('data_quality'))}"
        )

with entry_tab:
    render_v3_section("Entry / Short Cover", "需給・点火・Entry状態を確認します。", icon="🎯")
    if not short_row and not live_entry:
        st.info("現在、この銘柄のEntry / Short Cover監視データはありません。")
    else:
        e1, e2, e3, e4 = st.columns(4)
        e1.metric(
            "Live Status",
            _safe_text(live_entry.get("status")),
        )
        e2.metric("Alert Tier", _safe_text(short_row.get("alert_tier")))
        e3.metric("Alert Score", _fmt_num(short_row.get("alert_score"), 1))
        e4.metric("Phase", _safe_text(short_row.get("phase")))

        e5, e6, e7, e8 = st.columns(4)
        e5.metric("Cover Score", _fmt_num(short_row.get("cover_score"), 1))
        e6.metric("Ignition", _fmt_num(short_row.get("ignition_score"), 1))
        e7.metric("RS Watch", _fmt_num(short_row.get("rs_watch"), 0))
        e8.metric("Volume", _fmt_num(short_row.get("vol_ratio"), 1, "x"))

        reason = _safe_text(
            live_entry.get("reason")
            or short_row.get("alert_reason"),
            "",
        )
        if reason:
            st.info(f"🎯 監視理由: {reason}")
        st.caption(
            f"直近Alert: {_safe_text(short_row.get('alert_date'))}"
            f" ｜ Outcome: {_safe_text(short_row.get('outcome_status'))}"
        )

with event_tab:
    render_v3_section("決算・材料", "直近イベントと登録材料を確認します。", icon="📅")
    x1, x2, x3 = st.columns(3)
    x1.metric(
        "次回決算予定",
        earnings_date.isoformat() if earnings_date else "取得不可 / 未確認",
    )
    if earnings_date:
        days = (earnings_date - datetime.date.today()).days
        x2.metric("決算まで", f"{days}日" if days >= 0 else "発表済み")
    else:
        x2.metric("決算まで", "—")
    x3.metric("材料発生日", _safe_text(portfolio_row.get("材料発生日")))

    catalyst = _safe_text(portfolio_row.get("材料メモ"), "")
    if catalyst:
        st.info(f"💡 登録材料: {catalyst}")

    render_v3_section("最新ニュース", "銘柄に関連する直近ニュースを確認します。", icon="📰")
    news_items = _fetch_news(active_ticker, company_name)
    if not news_items:
        st.caption("関連ニュースを取得できませんでした。")
    else:
        for item in news_items:
            title = html.escape(item.get("title", ""))
            link = html.escape(item.get("link", ""), quote=True)
            source = html.escape(item.get("source", ""))
            date_text = html.escape(item.get("date", ""))
            st.markdown(
                f'<div style="padding:8px 0;border-bottom:1px solid rgba(128,128,128,.15)">'
                f'<a href="{link}" target="_blank" style="text-decoration:none;font-weight:700">{title}</a><br>'
                f'<span style="font-size:.78em;opacity:.65">{source}　{date_text}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )

with record_tab:
    render_v3_section("AI分析・リサーチ記録", "保存済み分析と決算レポートへつなぎます。", icon="📚")
    r1, r2, r3 = st.columns(3)
    r1.metric("銘柄分析レポート", analysis_count)
    r2.metric("決算分析レポート", earnings_count)
    r3.metric("更新日", _safe_text(portfolio_row.get("更新日")))

    st.caption(
        "AI分析の作成・保存は銘柄管理ダッシュボード、"
        "過去レポートの閲覧はレポートビューアで行います。"
        "共通銘柄はそのまま引き継がれます。"
    )
    q1, q2 = st.columns(2)
    with q1:
        st.page_link(
            "dashboard_app.py",
            label="AI分析・銘柄リサーチ記録へ",
            icon="🤖",
            use_container_width=True,
        )
    with q2:
        st.page_link(
            "pages/📖_レポートビューア.py",
            label="過去レポートを見る",
            icon="📖",
            use_container_width=True,
        )

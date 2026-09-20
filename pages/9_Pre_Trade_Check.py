from __future__ import annotations

import base64
import io
import os

import pandas as pd
import requests
import streamlit as st
import yfinance as yf

from pretrade import build_technical_snapshot, evaluate_pretrade
from short_cover import build_entry_hunter_snapshot, normalize_ticker


st.set_page_config(page_title="Pre-Trade Check", page_icon="🛡️", layout="wide")


@st.cache_data(ttl=300, show_spinner=False)
def load_daily(ticker: str) -> pd.DataFrame:
    symbol = ticker if "." in ticker else f"{ticker}.T"
    try:
        df = yf.download(
            symbol,
            period="18mo",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=900, show_spinner=False)
def load_benchmark() -> pd.DataFrame:
    for symbol in ("^TOPX", "^N225"):
        try:
            df = yf.download(
                symbol,
                period="18mo",
                interval="1d",
                auto_adjust=True,
                progress=False,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.dropna(subset=["Close"])
            if not df.empty:
                return df
        except Exception:
            continue
    return pd.DataFrame()


@st.cache_data(ttl=120, show_spinner=False)
def load_intraday(ticker: str) -> pd.DataFrame:
    symbol = ticker if "." in ticker else f"{ticker}.T"
    try:
        df = yf.download(
            symbol,
            period="10d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df.dropna(subset=["Close"])
    except Exception:
        return pd.DataFrame()


def _github_config():
    try:
        return {
            "token": st.secrets["GITHUB_TOKEN"],
            "repo": st.secrets["GITHUB_REPO"],
            "branch": st.secrets.get("GITHUB_BRANCH", "main"),
        }
    except Exception:
        return None


def load_portfolio() -> pd.DataFrame:
    path = "portfolio_data.csv"
    config = _github_config()

    if config:
        url = f"https://api.github.com/repos/{config['repo']}/contents/{path}"
        try:
            resp = requests.get(
                url,
                headers={
                    "Authorization": f"Bearer {config['token']}",
                    "Accept": "application/vnd.github+json",
                },
                params={"ref": config["branch"]},
                timeout=10,
            )
            if resp.status_code == 200:
                raw = resp.json().get("content", "").replace("\n", "")
                text = base64.b64decode(raw).decode("utf-8-sig")
                return pd.read_csv(io.StringIO(text), dtype=str).fillna("")
        except Exception:
            pass

    if os.path.exists(path):
        try:
            return pd.read_csv(path, dtype=str).fillna("")
        except Exception:
            pass
    return pd.DataFrame()


def _price(value, fallback=0.0) -> float:
    try:
        text = str(value).replace(",", "").replace("円", "").strip()
        return float(text) if text else float(fallback)
    except Exception:
        return float(fallback)


def _yen(v) -> str:
    try:
        return f"¥{float(v):,.0f}"
    except Exception:
        return "—"


st.title("🛡️ Pre-Trade Check v1")
st.caption(
    "注文前の最終チェック。チャートの質・買い位置・リスク管理を分離して採点します。"
    "これは売買推奨ではなく、確認漏れを減らすための実行支援ツールです。"
)

portfolio = load_portfolio()

if portfolio.empty or "ティッカー" not in portfolio.columns:
    st.warning("登録銘柄を読み込めませんでした。ティッカーを直接入力してください。")
    ticker = normalize_ticker(st.text_input("証券コード", placeholder="例: 4063"))
    row = {}
else:
    options = []
    rows = {}
    for _, r in portfolio.iterrows():
        code = normalize_ticker(r.get("ティッカー", ""))
        if not code:
            continue
        name = str(r.get("銘柄名", "")).strip()
        label = f"{name}（{code}）" if name else code
        options.append(label)
        rows[label] = r.to_dict()

    selected = st.selectbox("購入前チェックする銘柄", options)
    row = rows.get(selected, {})
    ticker = normalize_ticker(row.get("ティッカー", ""))

if not ticker:
    st.stop()

with st.spinner(f"{ticker} の価格データを取得中..."):
    daily = load_daily(ticker)
    benchmark = load_benchmark()
    intraday = load_intraday(ticker)

if daily.empty:
    st.error("日足データを取得できませんでした。")
    st.stop()

tech = build_technical_snapshot(daily, benchmark)
hunter = build_entry_hunter_snapshot(daily, intraday) if not intraday.empty else None

if not tech.get("data_ok"):
    st.error(tech.get("reason", "価格データ不足"))
    st.stop()

price = float(tech["price"])
stop_suggestions = tech.get("stop_suggestions", {})

st.markdown("### ① 自動チェック")
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("現在値", _yen(price), f"{tech.get('change_pct'):+.1f}%" if tech.get("change_pct") is not None else None)
c2.metric("Stage", str(tech.get("stage", "—")))
c3.metric("RS Proxy", f"{tech.get('rs_proxy'):.0f}" if tech.get("rs_proxy") is not None else "—")
c4.metric("出来高比", f"{tech.get('volume_ratio'):.2f}x" if tech.get("volume_ratio") is not None else "—")
c5.metric("20EMA乖離", f"{tech.get('ema20_gap_pct'):+.1f}%" if tech.get("ema20_gap_pct") is not None else "—")

with st.expander("📊 テクニカル詳細", expanded=False):
    d1, d2, d3, d4, d5 = st.columns(5)
    d1.metric("20EMA", _yen(tech.get("ema20")))
    d2.metric("50MA", _yen(tech.get("sma50")))
    d3.metric("150MA", _yen(tech.get("sma150")))
    d4.metric("200MA", _yen(tech.get("sma200")))
    d5.metric("ATR14", _yen(tech.get("atr14")))
    st.write(
        f"20日高値ブレイク: {'✅' if tech.get('breakout20') else '—'}　"
        f"20EMA押し目反発: {'✅' if tech.get('pullback20') else '—'}　"
        f"寄り付きギャップ: {tech.get('gap_pct'):+.1f}%"
        if tech.get("gap_pct") is not None else ""
    )

st.markdown("### ② Entry Hunter")
if hunter:
    h1, h2, h3, h4 = st.columns(4)
    h1.metric("状態", hunter.get("status", "—"))
    h2.metric("Hunter Score", f"{hunter.get('score', 0):.0f}")
    h3.metric("VWAP", _yen(hunter.get("vwap")))
    h4.metric("15分出来高", f"{hunter.get('relvol15'):.2f}x" if hunter.get("relvol15") is not None else "—")
    if hunter.get("reason"):
        st.caption(f"成立: {hunter['reason']}")
    if hunter.get("risk"):
        st.caption(f"注意: {hunter['risk']}")
else:
    st.info("Entry Hunterの5分足データは取得できませんでした。日足ベースで判定します。")

st.markdown("### ③ 売買プラン")

registered_stop = _price(row.get("損切りライン", ""), 0)
registered_target = _price(row.get("目標株価", ""), 0)

default_stop = registered_stop
if default_stop <= 0 and stop_suggestions:
    # Prefer a balanced ATR stop for the first proposal.
    default_stop = float(
        stop_suggestions.get("ATR 1.5倍")
        or stop_suggestions.get("20EMA下")
        or next(iter(stop_suggestions.values()))
    )

default_target = registered_target
if default_target <= price and default_stop > 0:
    default_target = price + max(price - default_stop, 0) * 2.0
if default_target <= price:
    default_target = price * 1.08

p1, p2, p3 = st.columns(3)
with p1:
    entry = st.number_input(
        "Entry",
        min_value=0.0,
        value=float(round(price, 1)),
        step=max(1.0, round(price * 0.001, 1)),
    )
with p2:
    stop = st.number_input(
        "Stop",
        min_value=0.0,
        value=float(round(default_stop, 1)) if default_stop > 0 else 0.0,
        step=max(1.0, round(price * 0.001, 1)),
    )
with p3:
    target = st.number_input(
        "Target",
        min_value=0.0,
        value=float(round(default_target, 1)),
        step=max(1.0, round(price * 0.001, 1)),
    )

if stop_suggestions:
    st.caption(
        "Stop候補: "
        + "　".join(f"{k} {_yen(v)}" for k, v in stop_suggestions.items())
    )

r1, r2, r3 = st.columns(3)
with r1:
    capital = st.number_input(
        "運用資金",
        min_value=100_000.0,
        value=1_000_000.0,
        step=100_000.0,
        format="%.0f",
    )
with r2:
    risk_percent = st.number_input(
        "1回の許容損失（資金比%）",
        min_value=0.1,
        max_value=5.0,
        value=1.0,
        step=0.1,
    )
with r3:
    earnings_mode = st.selectbox(
        "決算まで",
        ["未確認", "当日", "1-2日", "3-5日", "6-10日", "11日以上"],
        index=0,
    )

earnings_days_map = {
    "未確認": None,
    "当日": 0,
    "1-2日": 2,
    "3-5日": 4,
    "6-10日": 8,
    "11日以上": 20,
}
earnings_days = earnings_days_map[earnings_mode]

st.markdown("#### チャート形状（TradingViewで確認したものだけON）")
f1, f2, f3 = st.columns(3)
with f1:
    vcp = st.checkbox("VCP")
with f2:
    pp = st.checkbox("PP / Pivot")
with f3:
    c3 = st.checkbox("3C")

result = evaluate_pretrade(
    tech,
    entry=entry,
    stop=stop,
    target=target,
    capital=capital,
    risk_percent=risk_percent,
    earnings_days=earnings_days,
    entry_hunter=hunter,
    pattern_flags={"VCP": vcp, "PP": pp, "3C": c3},
    lot_size=100,
)

st.divider()
st.markdown("## ④ SYSTEM VERDICT")

v1, v2, v3, v4 = st.columns([1.3, 1, 1, 1])
v1.metric("判定", result["verdict"])
v2.metric("総合", f"{result['total_score']:.0f} / 100")
v3.metric("RR", f"1:{result['rr']:.2f}" if result.get("rr") is not None else "—")
v4.metric("推奨株数", f"{int(result['position']['shares']):,}株")

s1, s2, s3 = st.columns(3)
s1.metric("Setup Quality", f"{result['setup_score']:.0f} / 40")
s2.metric("Entry Timing", f"{result['entry_score']:.0f} / 35")
s3.metric("Risk Control", f"{result['risk_score']:.0f} / 25")

if result.get("blocked"):
    st.error(
        "TRADE BLOCKED: " + " / ".join(result.get("block_reasons", [])),
        icon="⛔",
    )

plan1, plan2, plan3, plan4 = st.columns(4)
plan1.metric("Entry", _yen(entry))
plan2.metric("Stop", _yen(stop))
plan3.metric("Target", _yen(target))
plan4.metric(
    "最大想定損失",
    _yen(result["position"].get("actual_max_loss")),
)

col_good, col_risk = st.columns(2)
with col_good:
    st.markdown("### ✅ Positive")
    positives = result.get("positive", [])
    if positives:
        for x in positives:
            st.write(f"✓ {x}")
    else:
        st.caption("強い加点要因はまだありません。")

with col_risk:
    st.markdown("### ⚠️ Risk / Check")
    risks = result.get("risks", [])
    if risks:
        for x in risks:
            st.write(f"• {x}")
    else:
        st.caption("大きな注意項目は検出されていません。")

st.info(
    "運用順序：今日見るべき銘柄 → Entry Hunter → Pre-Trade Check → "
    "Entry / Stop / Target / 株数を確認 → 発注判断。",
    icon="🧭",
)

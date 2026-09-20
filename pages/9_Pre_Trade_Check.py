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


PLAN_HISTORY_FILE = "data/pretrade_trade_plans.csv"
PLAN_HISTORY_COLUMNS = [
    "plan_id", "confirmed_at", "ticker", "name", "source",
    "verdict", "total_score", "setup_score", "entry_score", "risk_score",
    "stage", "rs_proxy", "volume_ratio", "ema20_gap_pct",
    "hunter_status", "hunter_score", "earnings_mode",
    "capital", "risk_percent", "entry", "stop", "target", "rr",
    "shares", "position_value", "actual_max_loss", "memo",
]


def _github_headers(config):
    return {
        "Authorization": f"Bearer {config['token']}",
        "Accept": "application/vnd.github+json",
    }


def load_trade_plan_history() -> pd.DataFrame:
    config = _github_config()
    if config:
        url = f"https://api.github.com/repos/{config['repo']}/contents/{PLAN_HISTORY_FILE}"
        try:
            resp = requests.get(
                url,
                headers=_github_headers(config),
                params={"ref": config["branch"]},
                timeout=10,
            )
            if resp.status_code == 200:
                raw = resp.json().get("content", "").replace("\n", "")
                text = base64.b64decode(raw).decode("utf-8-sig")
                df = pd.read_csv(io.StringIO(text), dtype=str).fillna("")
                for col in PLAN_HISTORY_COLUMNS:
                    if col not in df.columns:
                        df[col] = ""
                return df[PLAN_HISTORY_COLUMNS]
        except Exception:
            pass

    if os.path.exists(PLAN_HISTORY_FILE):
        try:
            df = pd.read_csv(PLAN_HISTORY_FILE, dtype=str).fillna("")
            for col in PLAN_HISTORY_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            return df[PLAN_HISTORY_COLUMNS]
        except Exception:
            pass

    return pd.DataFrame(columns=PLAN_HISTORY_COLUMNS)


def save_trade_plan_history(df: pd.DataFrame) -> tuple[bool, str]:
    os.makedirs(os.path.dirname(PLAN_HISTORY_FILE), exist_ok=True)
    out = df.copy()
    for col in PLAN_HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = ""
    out = out[PLAN_HISTORY_COLUMNS]
    out.to_csv(PLAN_HISTORY_FILE, index=False, encoding="utf-8-sig")

    config = _github_config()
    if not config:
        return True, "ローカル保存"

    url = f"https://api.github.com/repos/{config['repo']}/contents/{PLAN_HISTORY_FILE}"
    current_sha = None
    try:
        get_resp = requests.get(
            url,
            headers=_github_headers(config),
            params={"ref": config["branch"]},
            timeout=10,
        )
        if get_resp.status_code == 200:
            current_sha = get_resp.json().get("sha")

        csv_bytes = out.to_csv(index=False).encode("utf-8-sig")
        payload = {
            "message": "Save confirmed pre-trade plan",
            "content": base64.b64encode(csv_bytes).decode("ascii"),
            "branch": config["branch"],
        }
        if current_sha:
            payload["sha"] = current_sha

        put_resp = requests.put(
            url,
            headers=_github_headers(config),
            json=payload,
            timeout=15,
        )
        if put_resp.status_code in (200, 201):
            return True, "GitHubへ保存"
        return True, f"ローカル保存（GitHub保存失敗: {put_resp.status_code}）"
    except Exception as exc:
        return True, f"ローカル保存（GitHub保存失敗: {exc}）"


def append_trade_plan(record: dict) -> tuple[bool, str]:
    history = load_trade_plan_history()
    new_row = {col: record.get(col, "") for col in PLAN_HISTORY_COLUMNS}
    history = pd.concat([history, pd.DataFrame([new_row])], ignore_index=True)
    return save_trade_plan_history(history)


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

incoming_ticker = normalize_ticker(st.session_state.pop("pretrade_ticker", ""))
incoming_source = str(st.session_state.pop("pretrade_source", "") or "").strip()
incoming_name = str(st.session_state.pop("pretrade_name", "") or "").strip()
incoming_earnings_days = st.session_state.pop("pretrade_earnings_days", None)

portfolio = load_portfolio()
row = {}

if portfolio.empty or "ティッカー" not in portfolio.columns:
    st.warning("登録銘柄を読み込めませんでした。ティッカーを直接入力してください。")
    ticker = normalize_ticker(
        st.text_input(
            "証券コード",
            value=incoming_ticker,
            placeholder="例: 4063",
            key="pretrade_direct_ticker",
        )
    )
else:
    options = []
    rows = {}
    label_by_code = {}

    for _, r in portfolio.iterrows():
        code = normalize_ticker(r.get("ティッカー", ""))
        if not code:
            continue
        name = str(r.get("銘柄名", "")).strip()
        label = f"{name}（{code}）" if name else code
        options.append(label)
        rows[label] = r.to_dict()
        label_by_code[code] = label

    # Entry Hunterなどから渡された未登録銘柄も、そのままチェックできるようにする。
    if incoming_ticker and incoming_ticker not in label_by_code:
        incoming_label = (
            f"{incoming_name}（{incoming_ticker}）"
            if incoming_name
            else f"未登録銘柄（{incoming_ticker}）"
        )
        options.insert(0, incoming_label)
        rows[incoming_label] = {
            "ティッカー": incoming_ticker,
            "銘柄名": incoming_name,
        }
        label_by_code[incoming_ticker] = incoming_label

    if not options:
        st.warning("チェック可能な銘柄がありません。")
        st.stop()

    if incoming_ticker and incoming_ticker in label_by_code:
        st.session_state["pretrade_selected_label"] = label_by_code[incoming_ticker]
        if incoming_source:
            st.session_state["pretrade_active_source"] = incoming_source
            st.session_state["pretrade_source_ticker"] = incoming_ticker
        if incoming_earnings_days is not None:
            st.session_state["pretrade_incoming_earnings_days"] = incoming_earnings_days

    selected = st.selectbox(
        "購入前チェックする銘柄",
        options,
        key="pretrade_selected_label",
    )
    row = rows.get(selected, {})
    ticker = normalize_ticker(row.get("ティッカー", ""))

if not ticker:
    st.stop()

_active_source = str(st.session_state.get("pretrade_active_source", "") or "")
_source_ticker = normalize_ticker(st.session_state.get("pretrade_source_ticker", ""))
if _active_source and _source_ticker == ticker:
    st.success(f"🔗 {_active_source} から {ticker} を引き継ぎました。")

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
        key=f"pretrade_entry_{ticker}",
    )
with p2:
    stop = st.number_input(
        "Stop",
        min_value=0.0,
        value=float(round(default_stop, 1)) if default_stop > 0 else 0.0,
        step=max(1.0, round(price * 0.001, 1)),
        key=f"pretrade_stop_{ticker}",
    )
with p3:
    target = st.number_input(
        "Target",
        min_value=0.0,
        value=float(round(default_target, 1)),
        step=max(1.0, round(price * 0.001, 1)),
        key=f"pretrade_target_{ticker}",
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
    _incoming_days = st.session_state.pop("pretrade_incoming_earnings_days", None)
    _earnings_options = ["未確認", "当日", "1-2日", "3-5日", "6-10日", "11日以上"]
    if _incoming_days is None:
        _earnings_index = 0
    else:
        try:
            _d = int(_incoming_days)
            if _d <= 0:
                _earnings_index = 1
            elif _d <= 2:
                _earnings_index = 2
            elif _d <= 5:
                _earnings_index = 3
            elif _d <= 10:
                _earnings_index = 4
            else:
                _earnings_index = 5
        except (TypeError, ValueError):
            _earnings_index = 0

    earnings_mode = st.selectbox(
        "決算まで",
        _earnings_options,
        index=_earnings_index,
        key=f"pretrade_earnings_mode_{ticker}",
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
    vcp = st.checkbox("VCP", key=f"pretrade_vcp_{ticker}")
with f2:
    pp = st.checkbox("PP / Pivot", key=f"pretrade_pp_{ticker}")
with f3:
    c3 = st.checkbox("3C", key=f"pretrade_3c_{ticker}")

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

st.divider()
st.markdown("## ⑤ 注文直前5秒チェック")

_rr_value = result.get("rr")
_shares = int(result["position"].get("shares") or 0)
_confirmation_gate_ok = (
    not result.get("blocked")
    and _rr_value is not None
    and float(_rr_value) >= 2.0
    and _shares >= 100
)

g1, g2, g3 = st.columns(3)
g1.metric(
    "強制NG",
    "✅ なし" if not result.get("blocked") else "⛔ あり",
)
g2.metric(
    "RR確認ゲート",
    f"✅ 1:{float(_rr_value):.2f}" if _rr_value is not None and float(_rr_value) >= 2.0
    else (f"⚠️ 1:{float(_rr_value):.2f}" if _rr_value is not None else "⚠️ 未計算"),
)
g3.metric(
    "単元株",
    f"✅ {_shares:,}株" if _shares >= 100 else "⚠️ 100株未満",
)

if not _confirmation_gate_ok:
    st.warning(
        "TRADE PLAN CONFIRMの条件は「強制NGなし・RR 1:2以上・100株以上」です。"
        "条件未達の場合は、Entry / Stop / Target / 許容損失を見直してください。",
        icon="⚠️",
    )

q1, q2 = st.columns(2)
with q1:
    check_stop = st.checkbox(
        f"損切り価格 {_yen(stop)} を確認した",
        key=f"pretrade_check_stop_{ticker}",
    )
    check_target = st.checkbox(
        f"利確目標 {_yen(target)} を確認した",
        key=f"pretrade_check_target_{ticker}",
    )
    check_earnings = st.checkbox(
        f"決算日を確認した（{earnings_mode}）",
        key=f"pretrade_check_earnings_{ticker}",
    )
with q2:
    check_chase = st.checkbox(
        "追いかけ買いではないことを確認した",
        key=f"pretrade_check_chase_{ticker}",
    )
    check_loss = st.checkbox(
        f"最大想定損失 {_yen(result['position'].get('actual_max_loss'))} を許容できる",
        key=f"pretrade_check_loss_{ticker}",
    )
    check_shares = st.checkbox(
        f"発注株数 {_shares:,}株 を確認した",
        key=f"pretrade_check_shares_{ticker}",
    )

all_manual_checks = all([
    check_stop, check_target, check_earnings,
    check_chase, check_loss, check_shares,
])

trade_memo = st.text_area(
    "発注前メモ（任意）",
    placeholder="例：寄り後15分高値を維持した場合のみ。VWAP割れで見送り。",
    key=f"pretrade_memo_{ticker}",
    height=80,
)

confirm_disabled = not (_confirmation_gate_ok and all_manual_checks)
if st.button(
    "✅ TRADE PLAN CONFIRMED",
    type="primary",
    use_container_width=True,
    disabled=confirm_disabled,
    key=f"pretrade_confirm_{ticker}",
):
    now = pd.Timestamp.now(tz="Asia/Tokyo")
    company_name = str(row.get("銘柄名", "") or incoming_name or "")
    record = {
        "plan_id": f"{now.strftime('%Y%m%d-%H%M%S')}-{ticker}",
        "confirmed_at": now.isoformat(),
        "ticker": ticker,
        "name": company_name,
        "source": _active_source if _source_ticker == ticker else "Pre-Trade Check",
        "verdict": result.get("verdict", ""),
        "total_score": result.get("total_score", ""),
        "setup_score": result.get("setup_score", ""),
        "entry_score": result.get("entry_score", ""),
        "risk_score": result.get("risk_score", ""),
        "stage": tech.get("stage", ""),
        "rs_proxy": tech.get("rs_proxy", ""),
        "volume_ratio": tech.get("volume_ratio", ""),
        "ema20_gap_pct": tech.get("ema20_gap_pct", ""),
        "hunter_status": hunter.get("status", "") if hunter else "",
        "hunter_score": hunter.get("score", "") if hunter else "",
        "earnings_mode": earnings_mode,
        "capital": capital,
        "risk_percent": risk_percent,
        "entry": entry,
        "stop": stop,
        "target": target,
        "rr": result.get("rr", ""),
        "shares": _shares,
        "position_value": result["position"].get("position_value", ""),
        "actual_max_loss": result["position"].get("actual_max_loss", ""),
        "memo": trade_memo.strip(),
    }
    ok, message = append_trade_plan(record)
    if ok:
        st.session_state[f"pretrade_last_confirmed_{ticker}"] = record["plan_id"]
        st.success(
            f"✅ TRADE PLAN CONFIRMED｜{ticker}｜{_shares:,}株｜"
            f"Entry {_yen(entry)} / Stop {_yen(stop)} / Target {_yen(target)}"
            f"（{message}）"
        )
    else:
        st.error(f"売買計画を保存できませんでした：{message}")

_last_plan_id = st.session_state.get(f"pretrade_last_confirmed_{ticker}")
if _last_plan_id:
    st.caption(f"このセッションの最終確認済みPlan ID: {_last_plan_id}")

with st.expander("📚 確認済み売買プラン履歴", expanded=False):
    _history = load_trade_plan_history()
    if _history.empty:
        st.caption("まだ確認済みプランはありません。")
    else:
        _ticker_history = _history[_history["ticker"].astype(str) == str(ticker)].copy()
        if _ticker_history.empty:
            st.caption(f"{ticker} の確認済みプランはまだありません。")
        else:
            _ticker_history = _ticker_history.tail(20).iloc[::-1]
            show_cols = [
                "confirmed_at", "verdict", "total_score", "entry", "stop",
                "target", "rr", "shares", "actual_max_loss", "memo",
            ]
            st.dataframe(
                _ticker_history[show_cols].rename(columns={
                    "confirmed_at": "確認日時",
                    "verdict": "判定",
                    "total_score": "Score",
                    "entry": "Entry",
                    "stop": "Stop",
                    "target": "Target",
                    "rr": "RR",
                    "shares": "株数",
                    "actual_max_loss": "最大損失",
                    "memo": "メモ",
                }),
                hide_index=True,
                use_container_width=True,
            )

st.info(
    "運用順序：今日見るべき銘柄 → Entry Hunter → Pre-Trade Check → "
    "5秒チェック → TRADE PLAN CONFIRMED。"
    "CONFIRMEDは『発注前の計画を確認した記録』で、実際の約定記録とは分けて保存します。",
    icon="🧭",
)

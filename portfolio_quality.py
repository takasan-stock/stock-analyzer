from __future__ import annotations

import math
import re
from typing import Any

import pandas as pd


PERCENT_FIELDS = ("売上5y CAGR", "ROIC")
NUMBER_FIELDS = ("PER", "DPUP")


_EMPTY_MARKERS = {
    "",
    "-",
    "—",
    "none",
    "nan",
    "null",
    "n/a",
    "na",
}


_CORPORATE_NEWS_KEYWORDS = (
    "株",
    "株価",
    "決算",
    "業績",
    "ir",
    "上方修正",
    "下方修正",
    "増益",
    "減益",
    "売上",
    "営業利益",
    "経常利益",
    "純利益",
    "eps",
    "配当",
    "自社株",
    "株主",
    "証券",
    "投資",
    "市場",
    "上場",
    "時価総額",
    "per",
    "pbr",
    "受注",
    "提携",
    "m&a",
    "tob",
    "新サービス",
    "新商品",
    "発売",
    "導入",
    "採用",
    "コラボ",
    "事業",
    "企業",
    "会社",
    "earnings",
    "revenue",
    "profit",
    "shares",
    "stock",
)


_TRUSTED_FINANCE_SOURCES = (
    "株探",
    "日本経済新聞",
    "日経",
    "ロイター",
    "reuters",
    "bloomberg",
    "ブルームバーグ",
    "yahoo!ファイナンス",
    "みんかぶ",
    "minkabu",
    "会社四季報",
    "東洋経済",
    "時事通信",
)


def normalize_ticker_code(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().upper()
    if text.lower() in _EMPTY_MARKERS:
        return ""
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"\s+", "", text)
    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)
    return text


def normalize_metric_text(value: Any) -> str:
    """Return a canonical unit-less numeric string when the value is numeric."""
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    text = str(value).strip()
    if text.lower() in _EMPTY_MARKERS:
        return ""

    cleaned = (
        text.replace(",", "")
        .replace("％", "")
        .replace("%", "")
        .replace("倍", "")
        .strip()
    )
    cleaned = re.sub(r"(?i)x$", "", cleaned).strip()

    try:
        number = float(cleaned)
    except (TypeError, ValueError):
        return text

    if not math.isfinite(number):
        return ""

    if abs(number - round(number)) < 1e-12:
        return str(int(round(number)))
    return f"{number:.6f}".rstrip("0").rstrip(".")


def normalize_portfolio_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()

    out = df.copy()
    if "ティッカー" in out.columns:
        out["ティッカー"] = out["ティッカー"].map(normalize_ticker_code)

    for col in PERCENT_FIELDS + NUMBER_FIELDS:
        if col in out.columns:
            out[col] = out[col].map(normalize_metric_text)

    return out


def format_percent_metric(value: Any) -> str:
    normalized = normalize_metric_text(value)
    if not normalized:
        return "—"
    return f"{normalized}%"


def format_number_metric(value: Any) -> str:
    normalized = normalize_metric_text(value)
    return normalized or "—"


def display_portfolio_dataframe(df: pd.DataFrame | None) -> pd.DataFrame:
    """Format a portfolio dataframe for read-only display without mutating storage."""
    if df is None:
        return pd.DataFrame()

    out = normalize_portfolio_dataframe(df)

    for col in PERCENT_FIELDS:
        if col in out.columns:
            out[col] = out[col].map(format_percent_metric)

    for col in NUMBER_FIELDS:
        if col in out.columns:
            out[col] = out[col].map(format_number_metric)

    for col in out.columns:
        out[col] = out[col].map(
            lambda v: "—"
            if (
                v is None
                or (isinstance(v, float) and math.isnan(v))
                or str(v).strip().lower() in _EMPTY_MARKERS
            )
            else v
        )
    return out


def find_duplicate_tickers(df: pd.DataFrame | None) -> list[str]:
    if df is None or df.empty or "ティッカー" not in df.columns:
        return []

    tickers = df["ティッカー"].map(normalize_ticker_code)
    tickers = tickers[tickers != ""]
    dupes = tickers[tickers.duplicated(keep=False)]
    return sorted(set(dupes.tolist()))


def _company_aliases(company_name: Any) -> list[str]:
    name = str(company_name or "").strip()
    if not name or name.lower() in _EMPTY_MARKERS:
        return []

    aliases = {name}
    compact = re.sub(
        r"(株式会社|有限会社|ホールディングス|HD)$",
        "",
        name,
        flags=re.IGNORECASE,
    ).strip()
    if compact:
        aliases.add(compact)
    return sorted(aliases, key=len, reverse=True)


def build_company_news_query(ticker: Any, company_name: Any) -> str:
    ticker_text = normalize_ticker_code(ticker)
    aliases = _company_aliases(company_name)
    name = aliases[0] if aliases else ""
    base = f'"{name}"' if name else ticker_text
    context = "(株 OR 決算 OR IR OR 業績 OR 上方修正 OR 受注 OR 提携 OR 配当 OR 上場)"
    if ticker_text:
        return f"{base} {ticker_text} {context}".strip()
    return f"{base} {context}".strip()


def is_relevant_company_news(
    title: Any,
    ticker: Any,
    company_name: Any,
    source: Any = "",
) -> bool:
    """Conservative filter for ambiguous company names such as NOTE or スタメン."""
    title_text = str(title or "").strip()
    if not title_text:
        return False

    title_lower = title_text.lower()
    source_lower = str(source or "").strip().lower()
    ticker_text = normalize_ticker_code(ticker)

    if ticker_text and re.search(
        rf"(?<![0-9A-Z]){re.escape(ticker_text)}(?![0-9A-Z])",
        title_text.upper(),
    ):
        return True

    aliases = [a.lower() for a in _company_aliases(company_name) if len(a.strip()) >= 2]
    name_hit = any(alias in title_lower for alias in aliases)
    if not name_hit:
        return False

    context_hit = any(keyword.lower() in title_lower for keyword in _CORPORATE_NEWS_KEYWORDS)
    trusted_source = any(source_name.lower() in source_lower for source_name in _TRUSTED_FINANCE_SOURCES)
    return context_hit or trusted_source

from __future__ import annotations

import csv
from pathlib import Path

import streamlit as st


ACTIVE_TICKER_KEY = "site_active_ticker"
ACTIVE_NAME_KEY = "site_active_name"
ACTIVE_SOURCE_KEY = "site_active_source"


def normalize_site_ticker(value: object) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    if text.endswith(".0"):
        text = text[:-2]
    if text.upper().endswith(".T"):
        text = text[:-2]
    return text.strip()


def get_active_stock() -> tuple[str, str, str]:
    return (
        normalize_site_ticker(st.session_state.get(ACTIVE_TICKER_KEY, "")),
        str(st.session_state.get(ACTIVE_NAME_KEY, "") or "").strip(),
        str(st.session_state.get(ACTIVE_SOURCE_KEY, "") or "").strip(),
    )


def set_active_stock(ticker: object, name: object = "", source: str = "") -> None:
    code = normalize_site_ticker(ticker)
    if not code:
        return

    st.session_state[ACTIVE_TICKER_KEY] = code

    clean_name = str(name or "").strip()
    if clean_name:
        st.session_state[ACTIVE_NAME_KEY] = clean_name
    elif normalize_site_ticker(st.session_state.get(ACTIVE_TICKER_KEY, "")) != code:
        st.session_state[ACTIVE_NAME_KEY] = ""

    if source:
        st.session_state[ACTIVE_SOURCE_KEY] = str(source).strip()


def load_portfolio_stock_options(path: str = "portfolio_data.csv") -> list[tuple[str, str]]:
    file_path = Path(path)
    if not file_path.exists():
        return []

    rows: list[tuple[str, str]] = []
    seen: set[str] = set()

    try:
        with file_path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                code = normalize_site_ticker(row.get("ティッカー", ""))
                if not code or code in seen:
                    continue
                name = str(row.get("銘柄名", "") or "").strip()
                rows.append((code, name))
                seen.add(code)
    except OSError:
        return []

    return rows


def option_index(options: list[str], ticker: object) -> int:
    code = normalize_site_ticker(ticker)
    if code and code in options:
        return options.index(code)
    return 0

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests

JPX_BASE = "https://www.jpx.co.jp"
JPX_MARGIN_INDEX = f"{JPX_BASE}/markets/statistics-equities/margin/01.html"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/154 Safari/537.36"
)


@dataclass
class CreditSupplyLoadResult:
    balances: pd.DataFrame
    files_found: int
    files_loaded: int
    errors: list[str]


def normalize_ticker(value) -> str:
    s = str(value or "").strip().upper()
    if s.endswith(".T"):
        s = s[:-2]
    s = re.sub(r"[^0-9A-Z]", "", s)
    return s


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update(
        {
            "User-Agent": USER_AGENT,
            "Accept-Language": "ja,en;q=0.8",
        }
    )
    return s


def _clean_text(value) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    s = s.replace("\n", "").replace("\r", "").replace("\t", "")
    s = s.replace(" ", "").replace("　", "")
    s = s.replace("（", "(").replace("）", ")")
    return s.strip()


def _date_from_text(text: str) -> pd.Timestamp | None:
    m = re.search(r"(20\d{2})[年/\-.](\d{1,2})[月/\-.](\d{1,2})", text)
    if not m:
        return None
    try:
        return pd.Timestamp(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    except Exception:
        return None


def discover_jpx_margin_files(
    timeout: int = 20,
) -> tuple[list[tuple[str, pd.Timestamp | None]], list[str]]:
    """Discover the downloadable JPX issue-level margin balance workbooks."""
    errors: list[str] = []
    items: list[tuple[str, pd.Timestamp | None]] = []
    seen: set[str] = set()

    try:
        r = _session().get(JPX_MARGIN_INDEX, timeout=timeout)
        r.raise_for_status()
        html = r.text
    except requests.exceptions.RequestException as exc:
        return [], [f"JPX信用残ページ取得エラー: {exc}"]

    for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", html, flags=re.I | re.S):
        hrefs = re.findall(
            r'href=["\']([^"\']+\.(?:xlsx|xls|csv)(?:\?[^"\']*)?)["\']',
            row_html,
            flags=re.I,
        )
        if not hrefs:
            continue
        pub_date = _date_from_text(re.sub(r"<[^>]+>", " ", row_html))
        for href in hrefs:
            full = urljoin(JPX_MARGIN_INDEX, href)
            if full in seen:
                continue
            seen.add(full)
            items.append((full, pub_date))

    if not items:
        for href in re.findall(
            r'href=["\']([^"\']+\.(?:xlsx|xls|csv)(?:\?[^"\']*)?)["\']',
            html,
            flags=re.I,
        ):
            full = urljoin(JPX_MARGIN_INDEX, href)
            if full in seen:
                continue
            seen.add(full)
            items.append((full, None))

    items.sort(key=lambda x: x[1] or pd.Timestamp.min, reverse=True)
    if not items:
        errors.append("JPX信用残ファイルを発見できませんでした。ページ構造変更の可能性があります。")
    return items, errors


def _flatten_headers(raw: pd.DataFrame, header_row: int, depth: int) -> list[str]:
    pieces: list[list[str]] = []
    for offset in range(depth):
        row = [_clean_text(v) for v in raw.iloc[header_row + offset].tolist()]
        if offset == 0:
            filled: list[str] = []
            last = ""
            for value in row:
                if value:
                    last = value
                filled.append(last)
            row = filled
        pieces.append(row)

    labels: list[str] = []
    for col in range(raw.shape[1]):
        vals: list[str] = []
        for layer in pieces:
            v = layer[col]
            if v and v not in vals:
                vals.append(v)
        labels.append("".join(vals))
    return labels


def _find_header(raw: pd.DataFrame) -> tuple[int, int] | None:
    max_rows = min(len(raw), 30)
    for idx in range(max_rows):
        row = "|".join(_clean_text(v) for v in raw.iloc[idx].tolist())
        has_code = "コード" in row
        has_buy = "買" in row
        has_sell = "売" in row
        if has_code and has_buy and has_sell:
            next_row = ""
            if idx + 1 < len(raw):
                next_row = "|".join(_clean_text(v) for v in raw.iloc[idx + 1].tolist())
            depth = 2 if ("一般" in next_row or "制度" in next_row or "合計" in next_row) else 1
            return idx, depth
    return None


def _pick_column(columns: Iterable[str], *, side: str, kind: str = "total") -> str | None:
    side_tokens = ("買", "買残") if side == "long" else ("売", "売残")
    candidates: list[tuple[int, str]] = []

    for col in columns:
        label = _clean_text(col)
        if not any(tok in label for tok in side_tokens):
            continue
        if "残" not in label and "残高" not in label:
            continue

        score = 0
        if kind == "general":
            if "一般" not in label:
                continue
            score += 20
        elif kind == "standard":
            if "制度" not in label:
                continue
            score += 20
        else:
            if "合計" in label or "計" in label or "総" in label:
                score += 30
            if "一般" in label or "制度" in label:
                score -= 20
        if "株" in label:
            score += 2
        if "残高" in label:
            score += 3
        candidates.append((score, col))

    if not candidates:
        return None
    candidates.sort(key=lambda x: (x[0], len(x[1])), reverse=True)
    return candidates[0][1]


def _to_number(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(",", "", regex=False)
        .str.replace("株", "", regex=False)
        .str.replace("－", "", regex=False)
        .str.replace("-", "", regex=False)
        .str.strip()
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _parse_margin_sheet(raw: pd.DataFrame, default_date: pd.Timestamp | None) -> pd.DataFrame:
    found = _find_header(raw)
    if not found:
        return pd.DataFrame()
    header_row, depth = found
    columns = _flatten_headers(raw, header_row, depth)
    data = raw.iloc[header_row + depth :].copy()
    data.columns = columns
    data = data.dropna(how="all")

    code_col = next((c for c in columns if "銘柄コード" in _clean_text(c)), None)
    if code_col is None:
        code_col = next((c for c in columns if "コード" in _clean_text(c)), None)
    name_col = next((c for c in columns if "銘柄名" in _clean_text(c)), None)

    if code_col is None:
        return pd.DataFrame()

    long_total_col = _pick_column(columns, side="long", kind="total")
    short_total_col = _pick_column(columns, side="short", kind="total")
    long_general_col = _pick_column(columns, side="long", kind="general")
    long_standard_col = _pick_column(columns, side="long", kind="standard")
    short_general_col = _pick_column(columns, side="short", kind="general")
    short_standard_col = _pick_column(columns, side="short", kind="standard")

    if not any([long_total_col, long_general_col, long_standard_col]):
        return pd.DataFrame()
    if not any([short_total_col, short_general_col, short_standard_col]):
        return pd.DataFrame()

    out = pd.DataFrame()
    out["ticker"] = data[code_col].map(normalize_ticker)
    out["name"] = data[name_col].astype(str).str.strip() if name_col else ""

    def series_or_nan(col: str | None) -> pd.Series:
        if col and col in data.columns:
            return _to_number(data[col])
        return pd.Series(pd.NA, index=data.index, dtype="Float64")

    out["long_general"] = series_or_nan(long_general_col)
    out["long_standard"] = series_or_nan(long_standard_col)
    out["short_general"] = series_or_nan(short_general_col)
    out["short_standard"] = series_or_nan(short_standard_col)

    long_total = series_or_nan(long_total_col)
    short_total = series_or_nan(short_total_col)

    if long_total.isna().all():
        long_total = out[["long_general", "long_standard"]].sum(axis=1, min_count=1)
    if short_total.isna().all():
        short_total = out[["short_general", "short_standard"]].sum(axis=1, min_count=1)

    out["long_balance"] = pd.to_numeric(long_total, errors="coerce")
    out["short_balance"] = pd.to_numeric(short_total, errors="coerce")

    inferred_date = default_date
    if inferred_date is None:
        sample = " ".join(_clean_text(v) for v in raw.head(12).to_numpy().flatten())
        inferred_date = _date_from_text(sample)
    out["date"] = inferred_date

    out = out[out["ticker"].str.len().between(4, 5)]
    out = out[out[["long_balance", "short_balance"]].notna().any(axis=1)]
    return out.reset_index(drop=True)


def parse_jpx_margin_workbook(
    content: bytes,
    default_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse both the new daily format and legacy weekly workbook format."""
    frames: list[pd.DataFrame] = []
    try:
        book = pd.ExcelFile(io.BytesIO(content))
    except Exception:
        return pd.DataFrame()

    for sheet in book.sheet_names:
        try:
            raw = pd.read_excel(book, sheet_name=sheet, header=None)
        except Exception:
            continue
        parsed = _parse_margin_sheet(raw, default_date)
        if not parsed.empty:
            frames.append(parsed)

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["date", "ticker"], keep="last")
    return out


def load_jpx_margin_history(
    max_files: int = 16,
    timeout: int = 25,
) -> CreditSupplyLoadResult:
    items, errors = discover_jpx_margin_files(timeout=timeout)
    found = len(items)
    loaded = 0
    frames: list[pd.DataFrame] = []
    s = _session()

    for url, file_date in items[: max(1, int(max_files))]:
        try:
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            content = r.content

            if url.lower().split("?")[0].endswith(".csv"):
                try:
                    raw = pd.read_csv(io.BytesIO(content), header=None, encoding="utf-8-sig")
                except UnicodeDecodeError:
                    raw = pd.read_csv(io.BytesIO(content), header=None, encoding="cp932")
                frame = _parse_margin_sheet(raw, file_date)
            else:
                frame = parse_jpx_margin_workbook(content, default_date=file_date)

            if frame.empty:
                errors.append(f"信用残ファイル解析失敗: {url.rsplit('/', 1)[-1]}")
                continue
            frame["source_url"] = url
            frames.append(frame)
            loaded += 1
        except requests.exceptions.RequestException as exc:
            errors.append(f"信用残ファイル取得失敗: {url.rsplit('/', 1)[-1]} / {exc}")
        except Exception as exc:
            errors.append(f"信用残ファイル処理失敗: {url.rsplit('/', 1)[-1]} / {exc}")

    balances = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(
            columns=[
                "date",
                "ticker",
                "name",
                "long_balance",
                "short_balance",
                "long_general",
                "long_standard",
                "short_general",
                "short_standard",
                "source_url",
            ]
        )
    )
    if not balances.empty:
        balances["date"] = pd.to_datetime(balances["date"], errors="coerce")
        balances = balances.sort_values(["ticker", "date"]).reset_index(drop=True)

    return CreditSupplyLoadResult(
        balances=balances,
        files_found=found,
        files_loaded=loaded,
        errors=errors,
    )


def _safe_float(value) -> float | None:
    try:
        x = float(value)
        if pd.isna(x):
            return None
        return x
    except Exception:
        return None


def _ratio(long_balance, short_balance) -> float | None:
    long_v = _safe_float(long_balance)
    short_v = _safe_float(short_balance)
    if long_v is None or short_v is None or short_v <= 0:
        return None
    return long_v / short_v


def _score_volume_days(days: float | None) -> float | None:
    if days is None:
        return None
    if days < 0.5:
        return 100.0
    if days < 1.0:
        return 90.0
    if days < 2.0:
        return 70.0
    if days < 3.0:
        return 50.0
    if days < 5.0:
        return 30.0
    return 10.0


def _weighted_score(parts: list[tuple[float | None, float]]) -> tuple[float | None, float]:
    available = [(v, w) for v, w in parts if v is not None]
    if not available:
        return None, 0.0
    weight = sum(w for _, w in available)
    score = sum(v * w for v, w in available) / weight
    total_weight = sum(w for _, w in parts)
    coverage = weight / total_weight if total_weight else 0.0
    return round(score, 1), round(coverage * 100.0, 1)


def analyze_credit_supply(
    history: pd.DataFrame,
    ticker: str,
    *,
    latest_close: float | None = None,
    previous_close: float | None = None,
    avg_volume_5: float | None = None,
    avg_volume_25: float | None = None,
    shares_outstanding: float | None = None,
) -> dict:
    ticker_n = normalize_ticker(ticker)
    h = history.copy()
    if "ticker" not in h.columns:
        h = pd.DataFrame()
    else:
        h["ticker"] = h["ticker"].map(normalize_ticker)
        h = h[h["ticker"] == ticker_n].copy()
    if h.empty:
        return {
            "ticker": ticker_n,
            "status": "NO DATA",
            "score": None,
            "coverage": 0.0,
            "reasons": [],
            "risks": ["JPX信用残データがありません"],
            "history": h,
        }

    h["date"] = pd.to_datetime(h["date"], errors="coerce")
    h["long_balance"] = pd.to_numeric(h["long_balance"], errors="coerce")
    h["short_balance"] = pd.to_numeric(h["short_balance"], errors="coerce")
    h = h.dropna(subset=["date"]).sort_values("date").drop_duplicates("date", keep="last")
    h["credit_ratio"] = [
        _ratio(lb, sb) for lb, sb in zip(h["long_balance"], h["short_balance"])
    ]

    cur = h.iloc[-1]
    prev = h.iloc[-2] if len(h) >= 2 else None

    long_now = _safe_float(cur.get("long_balance"))
    short_now = _safe_float(cur.get("short_balance"))
    ratio_now = _ratio(long_now, short_now)

    long_prev = _safe_float(prev.get("long_balance")) if prev is not None else None
    short_prev = _safe_float(prev.get("short_balance")) if prev is not None else None
    ratio_prev = _ratio(long_prev, short_prev) if prev is not None else None

    long_delta = None if long_now is None or long_prev is None else long_now - long_prev
    short_delta = None if short_now is None or short_prev is None else short_now - short_prev
    ratio_delta = None if ratio_now is None or ratio_prev is None else ratio_now - ratio_prev

    latest_close = _safe_float(latest_close)
    previous_close = _safe_float(previous_close)
    price_change_pct = None
    if latest_close is not None and previous_close not in (None, 0):
        price_change_pct = (latest_close / previous_close - 1.0) * 100.0

    avg_volume_5 = _safe_float(avg_volume_5)
    avg_volume_25 = _safe_float(avg_volume_25)
    volume_days_5 = (
        long_now / avg_volume_5
        if long_now is not None and avg_volume_5 not in (None, 0)
        else None
    )
    volume_days_25 = (
        long_now / avg_volume_25
        if long_now is not None and avg_volume_25 not in (None, 0)
        else None
    )

    shares_outstanding = _safe_float(shares_outstanding)
    long_pct_shares = (
        long_now / shares_outstanding * 100.0
        if long_now is not None and shares_outstanding not in (None, 0)
        else None
    )
    short_pct_shares = (
        short_now / shares_outstanding * 100.0
        if short_now is not None and shares_outstanding not in (None, 0)
        else None
    )

    parts: list[tuple[float | None, float]] = []

    long_score = None
    if long_delta is not None:
        long_score = 100.0 if long_delta < 0 else (50.0 if long_delta == 0 else 10.0)
    parts.append((long_score, 25.0))

    short_score = None
    if short_delta is not None:
        if short_delta > 0 and (price_change_pct is None or price_change_pct >= 0):
            short_score = 90.0
        elif short_delta > 0:
            short_score = 45.0
        elif short_delta == 0:
            short_score = 50.0
        else:
            short_score = 35.0
    parts.append((short_score, 10.0))

    ratio_score = None
    if ratio_delta is not None:
        ratio_score = 100.0 if ratio_delta < 0 else (50.0 if ratio_delta == 0 else 15.0)
    parts.append((ratio_score, 15.0))

    parts.append((_score_volume_days(volume_days_25), 20.0))

    combo_score = None
    if price_change_pct is not None and long_delta is not None:
        if price_change_pct > 0 and long_delta < 0:
            combo_score = 100.0
        elif price_change_pct > 0 and long_delta >= 0:
            combo_score = 60.0
        elif price_change_pct <= 0 and long_delta > 0:
            combo_score = 5.0
        else:
            combo_score = 50.0
    parts.append((combo_score, 25.0))

    trend_score = None
    recent = h.tail(5)["long_balance"].dropna()
    if len(recent) >= 3:
        first = float(recent.iloc[0])
        last = float(recent.iloc[-1])
        if last < first:
            trend_score = 90.0
        elif last > first:
            trend_score = 30.0
        else:
            trend_score = 50.0
    parts.append((trend_score, 5.0))

    score, coverage = _weighted_score(parts)

    if score is None:
        status = "NO DATA"
    elif score >= 80:
        status = "🟢 改善優位"
    elif score >= 65:
        status = "🟢 やや改善"
    elif score >= 50:
        status = "🟡 中立"
    else:
        status = "🟠 悪化注意"

    reasons: list[str] = []
    risks: list[str] = []

    if long_delta is not None:
        if long_delta < 0:
            reasons.append(f"信用買残が前回比 {long_delta:+,.0f}株減少")
        elif long_delta > 0:
            risks.append(f"信用買残が前回比 {long_delta:+,.0f}株増加")

    if short_delta is not None:
        if short_delta > 0:
            if price_change_pct is not None and price_change_pct < 0:
                risks.append("信用売残は増加したが株価は下落しており、売り方優勢の可能性もある")
            else:
                reasons.append(f"信用売残が前回比 {short_delta:+,.0f}株増加")
        elif short_delta < 0:
            risks.append(f"信用売残が前回比 {short_delta:+,.0f}株減少")

    if ratio_delta is not None:
        if ratio_delta < 0:
            reasons.append(f"信用倍率が {ratio_prev:.2f}倍 → {ratio_now:.2f}倍へ低下")
        elif ratio_delta > 0:
            risks.append(f"信用倍率が {ratio_prev:.2f}倍 → {ratio_now:.2f}倍へ上昇")

    if price_change_pct is not None and long_delta is not None:
        if price_change_pct > 0 and long_delta < 0:
            reasons.append("株価上昇と信用買残減少が同時進行")
        elif price_change_pct <= 0 and long_delta > 0:
            risks.append("株価下落中に信用買残が増加")

    if volume_days_25 is not None:
        if volume_days_25 < 1.0:
            reasons.append(f"買残は25日平均出来高の {volume_days_25:.1f}日分で軽い")
        elif volume_days_25 >= 3.0:
            risks.append(f"買残は25日平均出来高の {volume_days_25:.1f}日分で重い")

    return {
        "ticker": ticker_n,
        "name": str(cur.get("name", "") or ""),
        "date": pd.Timestamp(cur["date"]),
        "previous_date": pd.Timestamp(prev["date"]) if prev is not None else None,
        "long_balance": long_now,
        "short_balance": short_now,
        "long_delta": long_delta,
        "short_delta": short_delta,
        "credit_ratio": ratio_now,
        "credit_ratio_prev": ratio_prev,
        "credit_ratio_delta": ratio_delta,
        "price_change_pct": price_change_pct,
        "volume_days_5": volume_days_5,
        "volume_days_25": volume_days_25,
        "long_pct_shares": long_pct_shares,
        "short_pct_shares": short_pct_shares,
        "score": score,
        "coverage": coverage,
        "status": status,
        "reasons": reasons,
        "risks": risks,
        "history": h,
    }

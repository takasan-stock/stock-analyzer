from __future__ import annotations

import io
import math
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from typing import Iterable
from urllib.parse import urljoin

import pandas as pd
import requests
import yfinance as yf

JPX_BASE = "https://www.jpx.co.jp"
JPX_INDEX = f"{JPX_BASE}/markets/public/short-selling/index.html"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/153 Safari/537.36"
)


@dataclass
class JpxLoadResult:
    events: pd.DataFrame
    files_found: int
    files_loaded: int
    pages_scanned: int
    errors: list[str]


def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": USER_AGENT, "Accept-Language": "ja,en;q=0.8"})
    return s


def _archive_url(index: int) -> str:
    return f"{JPX_BASE}/markets/public/short-selling/00-archives-{index:02d}.html"


def discover_jpx_excel_urls(
    archive_pages: int = 2,
    timeout: int = 20,
) -> tuple[list[tuple[str, pd.Timestamp | None]], list[str]]:
    """Discover JPX short-position Excel URLs plus their webpage publication date."""
    pages = [JPX_INDEX] + [_archive_url(i) for i in range(1, archive_pages + 1)]
    items: list[tuple[str, pd.Timestamp | None]] = []
    errors: list[str] = []
    seen: set[str] = set()
    s = _session()

    for page in pages:
        try:
            r = s.get(page, timeout=timeout)
            r.raise_for_status()
            text = r.text

            # Prefer row-level extraction so the date shown on the JPX page can be
            # treated as the first date the workbook was publicly available.
            found_in_rows = False
            for row_html in re.findall(r"<tr[^>]*>(.*?)</tr>", text, flags=re.I | re.S):
                hrefs = re.findall(
                    r'href=["\']([^"\']+\.(?:xlsx|xls)(?:\?[^"\']*)?)["\']',
                    row_html,
                    flags=re.I,
                )
                if not hrefs:
                    continue
                date_match = re.search(r"(20\d{2})[/-](\d{1,2})[/-](\d{1,2})", row_html)
                pub_date = None
                if date_match:
                    pub_date = pd.Timestamp(
                        year=int(date_match.group(1)),
                        month=int(date_match.group(2)),
                        day=int(date_match.group(3)),
                    )
                for href in hrefs:
                    full = urljoin(page, href)
                    if full not in seen:
                        seen.add(full)
                        items.append((full, pub_date))
                        found_in_rows = True

            # Fallback for future JPX HTML changes where table rows are not preserved.
            if not found_in_rows:
                hrefs = re.findall(
                    r'href=["\']([^"\']+\.(?:xlsx|xls)(?:\?[^"\']*)?)["\']',
                    text,
                    flags=re.I,
                )
                for href in hrefs:
                    full = urljoin(page, href)
                    if full not in seen:
                        seen.add(full)
                        items.append((full, None))
        except Exception as e:
            errors.append(f"{page}: {e}")

    return items, errors


def _clean_header(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return re.sub(r"\s+", "", str(value)).replace("\n", "")


def _pick_column(columns: Iterable, keyword_groups: list[list[str]]) -> str | None:
    pairs = [(c, _clean_header(c)) for c in columns]
    for keywords in keyword_groups:
        for original, normalized in pairs:
            if all(k.lower() in normalized.lower() for k in keywords):
                return original
    return None


def _to_float(value) -> float | None:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip().replace(",", "").replace("株", "")
    if not text or text.lower() in {"nan", "none", "-"}:
        return None
    try:
        return float(text)
    except ValueError:
        return None


def _to_percent(value) -> float | None:
    """Normalize Excel ratio values to percentage points (0.50 means 0.50%)."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return None
    if isinstance(value, str):
        text = value.strip().replace(",", "")
        if not text or text.lower() in {"nan", "none", "-"}:
            return None
        if "%" in text:
            try:
                return float(text.replace("%", ""))
            except ValueError:
                return None
        try:
            value = float(text)
        except ValueError:
            return None
    x = float(value)
    # Excel percentages such as 0.50% are usually stored as 0.005.
    if abs(x) <= 0.20:
        return x * 100.0
    return x


def normalize_ticker(value) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    text = str(value).strip().upper()
    if text.endswith(".0") and text[:-2].isdigit():
        text = text[:-2]
    text = re.sub(r"[^0-9A-Z]", "", text)
    if text.isdigit() and len(text) < 4:
        text = text.zfill(4)
    return text


def _detect_header_row(raw: pd.DataFrame, max_rows: int = 25) -> int | None:
    for i in range(min(max_rows, len(raw))):
        cells = [_clean_header(v) for v in raw.iloc[i].tolist()]
        joined = "|".join(cells)
        has_code = ("銘柄コード" in joined) or ("IssueCode".lower() in joined.lower())
        has_ratio = ("残高割合" in joined) or ("RatioofShort".lower() in joined.lower())
        if has_code and has_ratio:
            return i
    return None


def parse_jpx_excel(
    content: bytes,
    source_url: str = "",
    publication_date: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Parse JPX disclosure workbook into normalized event rows.

    The workbook format has changed over time, so headers are detected by keywords.
    Returned ratios are percentage points, e.g. 0.72 means 0.72%.
    """
    frames: list[pd.DataFrame] = []
    book = pd.ExcelFile(io.BytesIO(content))

    for sheet in book.sheet_names:
        raw = pd.read_excel(book, sheet_name=sheet, header=None, dtype=object)
        header_row = _detect_header_row(raw)
        if header_row is None:
            continue

        header = [str(v).strip() if pd.notna(v) else f"_blank_{j}" for j, v in enumerate(raw.iloc[header_row])]
        df = raw.iloc[header_row + 1 :].copy()
        df.columns = header
        df = df.dropna(how="all")
        if df.empty:
            continue

        code_col = _pick_column(df.columns, [["銘柄", "コード"], ["Issue", "Code"], ["Code"]])
        name_col = _pick_column(df.columns, [["銘柄", "名"], ["Issue", "Name"], ["NameofIssue"]])
        seller_col = _pick_column(
            df.columns,
            [["商号"], ["名称", "氏名"], ["Short", "Seller"], ["NameofShort"]],
        )
        date_col = _pick_column(df.columns, [["計算", "年月日"], ["Calculation", "Date"], ["Calculated", "Date"]])
        ratio_col = _pick_column(
            df.columns,
            [["空売り", "残高", "割合"], ["Short", "Position", "Ratio"], ["残高割合"]],
        )
        shares_col = _pick_column(
            df.columns,
            [["空売り", "残高", "数量"], ["Short", "Position", "Number"], ["残高数量"]],
        )

        if code_col is None or ratio_col is None:
            continue

        out = pd.DataFrame()
        out["ticker"] = df[code_col].map(normalize_ticker)
        out["name"] = df[name_col].fillna("").astype(str).str.strip() if name_col else ""
        out["seller"] = df[seller_col].fillna("不明").astype(str).str.strip() if seller_col else "不明"
        out["calc_date"] = pd.to_datetime(df[date_col], errors="coerce") if date_col else pd.NaT
        out["short_ratio"] = df[ratio_col].map(_to_percent)
        out["short_shares"] = df[shares_col].map(_to_float) if shares_col else None
        out["source_url"] = source_url
        out["publication_date"] = publication_date
        out = out[(out["ticker"] != "") & out["short_ratio"].notna()]
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=[
            "ticker", "name", "seller", "calc_date", "short_ratio",
            "short_shares", "source_url", "publication_date",
        ])

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["ticker", "seller", "calc_date", "short_ratio", "short_shares"], keep="last")
    return result


def load_jpx_events(archive_pages: int = 2, max_files: int = 70, workers: int = 8, timeout: int = 25) -> JpxLoadResult:
    discovered, errors = discover_jpx_excel_urls(archive_pages=archive_pages, timeout=timeout)
    discovered = discovered[:max_files]
    frames: list[pd.DataFrame] = []

    def fetch_one(item: tuple[str, pd.Timestamp | None]) -> tuple[str, pd.DataFrame | None, str | None]:
        url, publication_date = item
        try:
            s = _session()
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            return url, parse_jpx_excel(
                r.content,
                source_url=url,
                publication_date=publication_date,
            ), None
        except Exception as e:
            return url, None, str(e)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 10))) as ex:
        futures = [ex.submit(fetch_one, item) for item in discovered]
        for fut in as_completed(futures):
            url, frame, err = fut.result()
            if err:
                errors.append(f"{url}: {err}")
            elif frame is not None and not frame.empty:
                frames.append(frame)

    events = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=[
            "ticker", "name", "seller", "calc_date", "short_ratio",
            "short_shares", "source_url", "publication_date",
        ])
    )
    if not events.empty:
        events["calc_date"] = pd.to_datetime(events["calc_date"], errors="coerce")
        events = events.sort_values(["ticker", "seller", "calc_date"]).drop_duplicates(
            subset=["ticker", "seller", "calc_date", "short_ratio", "short_shares"], keep="last"
        )

    return JpxLoadResult(
        events=events,
        files_found=len(discovered),
        files_loaded=len(frames),
        pages_scanned=1 + archive_pages,
        errors=errors,
    )


def load_uploaded_workbooks(files) -> pd.DataFrame:
    frames = []
    for f in files or []:
        try:
            content = f.getvalue() if hasattr(f, "getvalue") else f.read()
            frame = parse_jpx_excel(content, source_url=getattr(f, "name", "uploaded"))
            if not frame.empty:
                frames.append(frame)
        except Exception:
            continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def build_short_metrics(
    events: pd.DataFrame,
    window_days: int = 75,
    as_of_date: pd.Timestamp | str | None = None,
) -> pd.DataFrame:
    """Build per-ticker disclosed short-position metrics from report events.

    Important: JPX daily files are report events, not a complete daily snapshot. We therefore
    label the result as an observed-window reconstruction and expose coverage age/counts.
    """
    columns = [
        "ticker", "name", "short_ratio", "short_shares", "institution_count", "delta_short",
        "cover_breadth", "build_breadth", "report_events", "last_calc_date", "data_age_days",
        "threshold_exit_count", "observed_sellers", "short_pressure_base",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    ev = events.copy()
    ev["calc_date"] = pd.to_datetime(ev["calc_date"], errors="coerce")
    if "publication_date" in ev.columns:
        ev["publication_date"] = pd.to_datetime(ev["publication_date"], errors="coerce")
    else:
        ev["publication_date"] = pd.NaT
    ev = ev.dropna(subset=["calc_date"])
    if ev.empty:
        return pd.DataFrame(columns=columns)

    # publication_date is the no-lookahead availability date where known.
    # Uploaded legacy workbooks may not carry it, so calculation date is the fallback.
    ev["_available_date"] = ev["publication_date"].fillna(ev["calc_date"]).dt.normalize()

    if as_of_date is not None:
        anchor_date = pd.Timestamp(as_of_date).normalize()
        ev = ev[ev["_available_date"] <= anchor_date].copy()
        if ev.empty:
            return pd.DataFrame(columns=columns)
    else:
        anchor_date = ev["_available_date"].max()

    cutoff = anchor_date - pd.Timedelta(days=window_days)
    ev = ev[ev["_available_date"] >= cutoff].copy()

    rows = []
    for ticker, tg in ev.groupby("ticker"):
        tg = tg.sort_values(["seller", "calc_date"])
        seller_latest = []
        decreases = 0
        increases = 0
        comparable = 0
        exits = 0
        delta_total = 0.0

        for seller, sg in tg.groupby("seller"):
            sg = sg.sort_values("calc_date")
            latest = sg.iloc[-1]
            prev = sg.iloc[-2] if len(sg) >= 2 else None
            current_ratio = float(latest["short_ratio"])
            current_shares = _to_float(latest.get("short_shares"))
            seller_latest.append((seller, current_ratio, current_shares, latest["calc_date"]))

            if prev is not None and pd.notna(prev.get("short_ratio")):
                prev_ratio = float(prev["short_ratio"])
                d = current_ratio - prev_ratio
                delta_total += d
                comparable += 1
                if d < -0.0001:
                    decreases += 1
                elif d > 0.0001:
                    increases += 1
                if prev_ratio >= 0.5 and current_ratio < 0.5:
                    exits += 1

        # The latest observed report for each seller represents the last known state in this window.
        active = [x for x in seller_latest if x[1] >= 0.5]
        ratio_sum = sum(x[1] for x in active)
        shares_sum = sum((x[2] or 0.0) for x in active)
        names = tg["name"].replace("nan", "").replace("None", "")
        name = next((str(v).strip() for v in reversed(names.tolist()) if str(v).strip()), "")
        last_date = max(x[3] for x in seller_latest)
        last_available = tg.loc[tg["calc_date"] == last_date, "_available_date"].max()
        age_base = last_available if pd.notna(last_available) else pd.Timestamp(last_date).normalize()
        age = max(0, (anchor_date - pd.Timestamp(age_base).normalize()).days)
        cover_breadth = decreases / comparable * 100 if comparable else None
        build_breadth = increases / comparable * 100 if comparable else None

        # Base pressure from disclosed size and number of institutions only; DTC is added later.
        ratio_points = min(45.0, ratio_sum / 4.0 * 45.0) if ratio_sum > 0 else 0.0
        institution_points = min(20.0, len(active) / 5.0 * 20.0)
        build_points = 0.0
        if delta_total > 0:
            build_points = min(20.0, delta_total / 0.8 * 20.0)
        freshness_points = max(0.0, 15.0 * (1 - min(age, 30) / 30.0))
        pressure_base = min(100.0, ratio_points + institution_points + build_points + freshness_points)

        rows.append({
            "ticker": ticker,
            "name": name,
            "short_ratio": ratio_sum,
            "short_shares": shares_sum if shares_sum > 0 else None,
            "institution_count": len(active),
            "delta_short": delta_total if comparable else None,
            "cover_breadth": cover_breadth,
            "build_breadth": build_breadth,
            "report_events": len(tg),
            "last_calc_date": pd.Timestamp(last_date),
            "data_age_days": age,
            "threshold_exit_count": exits,
            "observed_sellers": len(seller_latest),
            "short_pressure_base": pressure_base,
        })

    return pd.DataFrame(rows).sort_values(["short_pressure_base", "short_ratio"], ascending=False)


def _download_prices(tickers: list[str], period: str = "6mo") -> dict[str, pd.DataFrame]:
    if not tickers:
        return {}
    symbol_map = {f"{t}.T" if "." not in t else t: t for t in tickers}
    symbols = list(symbol_map)
    result: dict[str, pd.DataFrame] = {}
    try:
        data = yf.download(symbols, period=period, interval="1d", auto_adjust=True, progress=False, group_by="column", threads=True)
        if data is None or data.empty:
            return result
    except Exception:
        return result

    fields = ["Open", "High", "Low", "Close", "Volume"]
    for symbol, ticker in symbol_map.items():
        try:
            if isinstance(data.columns, pd.MultiIndex):
                frame = pd.DataFrame({f: data[f][symbol] for f in fields if f in data.columns.get_level_values(0) and symbol in data[f].columns})
            else:
                frame = data[[f for f in fields if f in data.columns]].copy()
            frame = frame.dropna(subset=["Close"])
            if not frame.empty:
                result[ticker] = frame
        except Exception:
            continue
    return result


def _price_features(frame: pd.DataFrame) -> dict:
    out = {
        "price": None, "change_pct": None, "vol_ratio": None, "avg_volume20": None,
        "dtc": None, "breakout5": False, "failed_breakdown": False, "avwap": None,
        "above_avwap": False, "ret20": None, "ret60": None, "rs_accel": False,
    }
    if frame is None or frame.empty or len(frame) < 6:
        return out

    f = frame.copy().dropna(subset=["Close"])
    close = f["Close"]
    high = f["High"] if "High" in f else close
    low = f["Low"] if "Low" in f else close
    volume = f["Volume"] if "Volume" in f else pd.Series(index=f.index, dtype=float)

    price = float(close.iloc[-1])
    out["price"] = price
    if len(close) >= 2 and close.iloc[-2] != 0:
        out["change_pct"] = float((close.iloc[-1] / close.iloc[-2] - 1) * 100)

    if len(volume.dropna()) >= 6:
        past = volume.iloc[:-1].tail(20).dropna()
        if not past.empty and float(past.mean()) > 0:
            out["avg_volume20"] = float(past.mean())
            out["vol_ratio"] = float(volume.iloc[-1] / past.mean())

    prev5 = high.iloc[:-1].tail(5)
    if not prev5.empty:
        out["breakout5"] = bool(close.iloc[-1] > prev5.max())

    prev20_low = low.iloc[:-1].tail(20)
    if not prev20_low.empty:
        recent_floor = float(prev20_low.min())
        # Intraday undercut followed by close back above the old 20-day floor.
        out["failed_breakdown"] = bool(float(low.iloc[-1]) < recent_floor and price >= recent_floor)

    # Daily AVWAP proxy anchored to the lowest low in the last 20 sessions.
    tail = f.tail(20).copy()
    if not tail.empty and "Volume" in tail:
        anchor_idx = tail["Low"].idxmin() if "Low" in tail else tail["Close"].idxmin()
        anchored = f.loc[anchor_idx:].copy()
        if not anchored.empty and anchored["Volume"].fillna(0).sum() > 0:
            typical = (
                (anchored.get("High", anchored["Close"]) + anchored.get("Low", anchored["Close"]) + anchored["Close"]) / 3.0
            )
            avwap = float((typical * anchored["Volume"]).sum() / anchored["Volume"].sum())
            out["avwap"] = avwap
            out["above_avwap"] = bool(price > avwap)

    if len(close) >= 21:
        out["ret20"] = float((close.iloc[-1] / close.iloc[-21] - 1) * 100)
    if len(close) >= 61:
        out["ret60"] = float((close.iloc[-1] / close.iloc[-61] - 1) * 100)
    if out["ret20"] is not None and out["ret60"] is not None:
        out["rs_accel"] = bool(out["ret20"] > out["ret60"] / 3.0)

    return out


def _finalize_price_feature_rows(rows: list[dict]) -> pd.DataFrame:
    result = pd.DataFrame(rows)
    if result.empty:
        return result

    # Watchlist/candidate-universe relative strength percentile, not IBD's proprietary RS Rating.
    raw = result["ret20"].fillna(0) * 0.4 + result["ret60"].fillna(0) * 0.6
    if len(result) >= 2:
        pct = raw.rank(pct=True, method="average")
        result["rs_watch"] = (1 + pct * 98).round(0)
    else:
        result["rs_watch"] = 50.0
    return result


def build_price_feature_snapshots(tickers: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build latest and previous-session price features with one yfinance download.

    The previous snapshot intentionally uses the same disclosed-short metrics later in
    score_short_cover(). This makes promotion detection a price/volume state transition
    detector and avoids any need for persistent daily state on Streamlit Cloud.
    """
    tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    frames = _download_prices(sorted(set(tickers)))
    current_rows: list[dict] = []
    previous_rows: list[dict] = []

    for ticker in tickers:
        frame = frames.get(ticker)

        current = _price_features(frame)
        current["ticker"] = ticker
        current["snapshot_date"] = (
            pd.Timestamp(frame.index[-1]).normalize()
            if frame is not None and not frame.empty else pd.NaT
        )
        current_rows.append(current)

        previous_frame = (
            frame.iloc[:-1].copy()
            if frame is not None and len(frame) >= 2
            else pd.DataFrame()
        )
        previous = _price_features(previous_frame)
        previous["ticker"] = ticker
        previous["snapshot_date"] = (
            pd.Timestamp(previous_frame.index[-1]).normalize()
            if not previous_frame.empty else pd.NaT
        )
        previous_rows.append(previous)

    return (
        _finalize_price_feature_rows(current_rows),
        _finalize_price_feature_rows(previous_rows),
    )


def build_price_feature_table(tickers: list[str]) -> pd.DataFrame:
    current, _previous = build_price_feature_snapshots(tickers)
    return current



def _truthy(value) -> bool:
    """NaN/NAをFalseとして扱う安全な真偽変換。"""
    try:
        return bool(value) if pd.notna(value) else False
    except Exception:
        return False

def score_short_cover(short_metrics: pd.DataFrame, price_features: pd.DataFrame) -> pd.DataFrame:
    if short_metrics is None or short_metrics.empty:
        return pd.DataFrame()
    df = short_metrics.copy()
    if price_features is not None and not price_features.empty:
        df = df.merge(price_features, on="ticker", how="left")
    else:
        for col in ["price", "change_pct", "vol_ratio", "avg_volume20", "breakout5", "failed_breakdown", "avwap", "above_avwap", "ret20", "ret60", "rs_accel", "rs_watch"]:
            df[col] = None

    # Days-to-cover from disclosed large-short shares only. It is a lower-bound proxy.
    df["dtc"] = df.apply(
        lambda r: (float(r["short_shares"]) / float(r["avg_volume20"]))
        if pd.notna(r.get("short_shares")) and pd.notna(r.get("avg_volume20")) and float(r.get("avg_volume20")) > 0
        else None,
        axis=1,
    )

    rows = []
    for _, r in df.iterrows():
        pressure = float(r.get("short_pressure_base") or 0)
        dtc = r.get("dtc")
        if pd.notna(dtc):
            pressure = min(100.0, pressure * 0.8 + min(20.0, float(dtc) / 5.0 * 20.0))

        # Absorption: short build-up or still-high short pressure while price refuses to break down.
        absorption = 0.0
        delta = r.get("delta_short")
        chg = r.get("change_pct")
        if pd.notna(delta) and float(delta) >= 0 and pd.notna(chg) and float(chg) >= 0:
            absorption += 35
        if _truthy(r.get("failed_breakdown")):
            absorption += 35
        if _truthy(r.get("above_avwap")):
            absorption += 20
        if pd.notna(r.get("ret20")) and float(r.get("ret20")) > -2:
            absorption += 10
        absorption = min(100.0, absorption)

        vol = float(r.get("vol_ratio")) if pd.notna(r.get("vol_ratio")) else 0.0
        volume_score = min(100.0, max(0.0, (vol - 1.0) / 2.0 * 100.0))
        avwap_score = 100.0 if _truthy(r.get("above_avwap")) else 0.0
        breakout_score = 100.0 if _truthy(r.get("breakout5")) else (55.0 if pd.notna(chg) and float(chg) > 0 else 0.0)
        rs = float(r.get("rs_watch")) if pd.notna(r.get("rs_watch")) else 50.0
        rs_score = max(0.0, min(100.0, rs))
        breadth = float(r.get("cover_breadth")) if pd.notna(r.get("cover_breadth")) else 0.0
        confirm_score = breadth
        if pd.notna(delta) and float(delta) < 0:
            confirm_score = min(100.0, confirm_score + min(35.0, abs(float(delta)) / 0.5 * 35.0))
        if int(r.get("threshold_exit_count") or 0) > 0:
            confirm_score = min(100.0, confirm_score + 20.0)

        early = (
            pressure * 0.25
            + absorption * 0.20
            + volume_score * 0.15
            + avwap_score * 0.15
            + breakout_score * 0.10
            + rs_score * 0.10
            + confirm_score * 0.05
        )
        early = round(min(100.0, early), 1)

        # Long-demand score helps separate pure covering from fresh demand.
        long_demand = (
            volume_score * 0.30
            + avwap_score * 0.25
            + breakout_score * 0.20
            + rs_score * 0.25
        )
        long_demand = round(min(100.0, long_demand), 1)

        # Ignition isolates the actual 'spark': volume expansion + AVWAP recovery
        # + short-term breakout + acceleration. It is explanatory; Cover Score remains
        # the stable v1 composite above.
        ignition = (
            volume_score * 0.35
            + avwap_score * 0.30
            + breakout_score * 0.25
            + (100.0 if _truthy(r.get("rs_accel")) else 0.0) * 0.10
        )
        ignition = round(min(100.0, ignition), 1)

        if early >= 65 and long_demand >= 65:
            regime = "🔥 COVER + NEW MONEY"
        elif early >= 65 and long_demand < 45:
            regime = "⚠️ PURE SHORT COVER"
        elif early < 65 and long_demand >= 70:
            regime = "🟢 NEW MONEY"
        else:
            regime = "⚪ NEUTRAL"

        if pressure < 35:
            phase = "NORMAL"
        elif early >= 85 and vol >= 2.0 and _truthy(r.get("breakout5")):
            phase = "🚀 SQUEEZE"
        elif (pd.notna(delta) and float(delta) < 0) and breadth >= 50 and early >= 60:
            phase = "✅ COVER CONFIRMED"
        elif early >= 65:
            phase = "🔥 COVER EARLY"
        elif pressure >= 55 and (absorption >= 35 or _truthy(r.get("above_avwap"))):
            phase = "👀 COVER WATCH"
        else:
            phase = "🧱 SHORT BUILDUP"

        # Confidence is about data completeness, not trade quality.
        comparable = int(r.get("observed_sellers") or 0)
        events = int(r.get("report_events") or 0)
        age = int(r.get("data_age_days") or 999)
        confidence = 0
        confidence += 35 if comparable >= 2 else (20 if comparable == 1 else 0)
        confidence += 35 if events >= 4 else min(35, events * 8)
        confidence += 30 if age <= 7 else (20 if age <= 21 else 10 if age <= 45 else 0)

        item = r.to_dict()
        item.update({
            "short_pressure": round(pressure, 1),
            "absorption_score": round(absorption, 1),
            "ignition_score": ignition,
            "cover_score": early,
            "long_demand_score": long_demand,
            "regime": regime,
            "confidence": int(min(100, confidence)),
            "phase": phase,
        })
        rows.append(item)

    out = pd.DataFrame(rows)
    return out.sort_values(["cover_score", "confidence", "short_pressure"], ascending=False).reset_index(drop=True)


PHASE_RANK = {
    "NORMAL": 0,
    "🧱 SHORT BUILDUP": 1,
    "👀 COVER WATCH": 2,
    "🔥 COVER EARLY": 3,
    "✅ COVER CONFIRMED": 4,
    "🚀 SQUEEZE": 5,
}


def build_promotion_table(current: pd.DataFrame, previous: pd.DataFrame) -> pd.DataFrame:
    """Compare the latest two price sessions and return fresh Short Cover promotions."""
    columns = [
        "ticker", "name", "prev_phase", "phase", "phase_jump",
        "prev_cover_score", "cover_score", "cover_delta",
        "prev_ignition_score", "ignition_score", "ignition_delta",
        "prev_long_demand_score", "long_demand_score",
        "fresh_breakout", "fresh_avwap_reclaim", "promotion_reason",
        "promotion_score", "regime", "short_pressure", "vol_ratio",
        "rs_watch", "confidence", "snapshot_date", "prev_snapshot_date",
    ]
    if current is None or current.empty or previous is None or previous.empty:
        return pd.DataFrame(columns=columns)

    prev_cols = [
        "ticker", "phase", "cover_score", "ignition_score", "long_demand_score",
        "breakout5", "above_avwap", "snapshot_date",
    ]
    p = previous[[x for x in prev_cols if x in previous.columns]].copy()
    p = p.rename(columns={
        "phase": "prev_phase",
        "cover_score": "prev_cover_score",
        "ignition_score": "prev_ignition_score",
        "long_demand_score": "prev_long_demand_score",
        "breakout5": "prev_breakout5",
        "above_avwap": "prev_above_avwap",
        "snapshot_date": "prev_snapshot_date",
    })

    merged = current.merge(p, on="ticker", how="inner")
    if merged.empty:
        return pd.DataFrame(columns=columns)

    rows = []
    for _, r in merged.iterrows():
        phase = str(r.get("phase", "NORMAL"))
        prev_phase = str(r.get("prev_phase", "NORMAL"))
        cur_rank = PHASE_RANK.get(phase, 0)
        prev_rank = PHASE_RANK.get(prev_phase, 0)
        phase_jump = cur_rank - prev_rank

        cover = float(r.get("cover_score")) if pd.notna(r.get("cover_score")) else 0.0
        prev_cover = float(r.get("prev_cover_score")) if pd.notna(r.get("prev_cover_score")) else 0.0
        ignition = float(r.get("ignition_score")) if pd.notna(r.get("ignition_score")) else 0.0
        prev_ignition = float(r.get("prev_ignition_score")) if pd.notna(r.get("prev_ignition_score")) else 0.0

        cover_delta = cover - prev_cover
        ignition_delta = ignition - prev_ignition
        fresh_breakout = _truthy(r.get("breakout5")) and not _truthy(r.get("prev_breakout5"))
        fresh_avwap = _truthy(r.get("above_avwap")) and not _truthy(r.get("prev_above_avwap"))
        cover_cross = prev_cover < 65 <= cover
        ignition_cross = prev_ignition < 65 <= ignition

        # A fresh candidate needs a meaningful state change; pure score noise is ignored.
        is_promotion = (
            phase_jump > 0
            or cover_cross
            or ignition_cross
            or (fresh_breakout and cover_delta >= 5)
            or (fresh_avwap and ignition_delta >= 10)
        )
        if not is_promotion:
            continue

        reasons = []
        if phase_jump > 0:
            reasons.append(f"{prev_phase} → {phase}")
        if cover_cross:
            reasons.append("Cover 65突破")
        if ignition_cross:
            reasons.append("Ignition 65突破")
        if fresh_breakout:
            reasons.append("5日高値を新規突破")
        if fresh_avwap:
            reasons.append("AVWAPを新規回復")
        if cover_delta >= 10:
            reasons.append(f"Cover +{cover_delta:.0f}")
        if ignition_delta >= 15:
            reasons.append(f"Ignition +{ignition_delta:.0f}")

        promotion_score = (
            max(0.0, phase_jump) * 18.0
            + max(0.0, cover_delta) * 0.8
            + max(0.0, ignition_delta) * 0.5
            + (12.0 if fresh_breakout else 0.0)
            + (8.0 if fresh_avwap else 0.0)
            + (8.0 if str(r.get("regime", "")).startswith("🔥") else 0.0)
        )

        item = r.to_dict()
        item.update({
            "prev_phase": prev_phase,
            "phase_jump": phase_jump,
            "prev_cover_score": round(prev_cover, 1),
            "cover_delta": round(cover_delta, 1),
            "prev_ignition_score": round(prev_ignition, 1),
            "ignition_delta": round(ignition_delta, 1),
            "prev_long_demand_score": r.get("prev_long_demand_score"),
            "fresh_breakout": fresh_breakout,
            "fresh_avwap_reclaim": fresh_avwap,
            "promotion_reason": " / ".join(reasons),
            "promotion_score": round(min(100.0, promotion_score), 1),
        })
        rows.append(item)

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    return out.sort_values(
        ["promotion_score", "phase_jump", "cover_delta", "ignition_delta"],
        ascending=False,
    ).reset_index(drop=True)



def _price_feature_table_asof(
    frames: dict[str, pd.DataFrame],
    tickers: list[str],
    as_of_date: pd.Timestamp,
) -> pd.DataFrame:
    rows: list[dict] = []
    cutoff = pd.Timestamp(as_of_date)

    for ticker in tickers:
        frame = frames.get(ticker)
        if frame is None or frame.empty:
            feat = _price_features(pd.DataFrame())
            feat["ticker"] = ticker
            feat["snapshot_date"] = pd.NaT
            rows.append(feat)
            continue

        idx = pd.to_datetime(frame.index)
        hist = frame.loc[idx <= cutoff].copy()
        feat = _price_features(hist)
        feat["ticker"] = ticker
        feat["snapshot_date"] = (
            pd.Timestamp(hist.index[-1]).normalize() if not hist.empty else pd.NaT
        )
        rows.append(feat)

    return _finalize_price_feature_rows(rows)


def _forward_trade_stats(frame: pd.DataFrame, signal_pos: int) -> dict:
    """Evaluate a close-confirmed signal using the next session open as entry."""
    out = {
        "entry_date": pd.NaT,
        "entry_price": None,
        "ret_1d": None,
        "ret_3d": None,
        "ret_5d": None,
        "ret_10d": None,
        "mfe_10d": None,
        "mae_10d": None,
    }
    if frame is None or frame.empty:
        return out

    # signal_pos points at the session whose close generated the signal.
    entry_pos = signal_pos + 1
    if entry_pos >= len(frame):
        return out

    entry_row = frame.iloc[entry_pos]
    entry = _to_float(entry_row.get("Open"))
    if entry is None or entry <= 0:
        entry = _to_float(entry_row.get("Close"))
    if entry is None or entry <= 0:
        return out

    out["entry_date"] = pd.Timestamp(frame.index[entry_pos]).normalize()
    out["entry_price"] = float(entry)

    for horizon in (1, 3, 5, 10):
        exit_pos = entry_pos + horizon - 1
        if exit_pos >= len(frame):
            continue
        exit_close = _to_float(frame.iloc[exit_pos].get("Close"))
        if exit_close is not None:
            out[f"ret_{horizon}d"] = (float(exit_close) / entry - 1.0) * 100.0

    end_pos = min(len(frame) - 1, entry_pos + 9)
    future = frame.iloc[entry_pos : end_pos + 1]
    if not future.empty:
        highs = future["High"] if "High" in future else future["Close"]
        lows = future["Low"] if "Low" in future else future["Close"]
        if highs.notna().any():
            out["mfe_10d"] = (float(highs.max()) / entry - 1.0) * 100.0
        if lows.notna().any():
            out["mae_10d"] = (float(lows.min()) / entry - 1.0) * 100.0

    return out


def backtest_short_cover(
    events: pd.DataFrame,
    tickers: list[str],
    sessions: int = 40,
    max_tickers: int = 40,
    min_cover_score: float = 55.0,
) -> pd.DataFrame:
    """Point-in-time backtest for recent Short Cover signals.

    - JPX events are filtered by publication_date when available (calc_date fallback).
    - Signals use data available through that session close.
    - Entry is the next session open.
    - Returns are measured to the 1st/3rd/5th/10th session close after entry.
    """
    columns = [
        "signal_date", "ticker", "name", "phase", "regime",
        "cover_score", "short_pressure", "absorption_score",
        "ignition_score", "long_demand_score", "confidence",
        "short_ratio", "delta_short", "cover_breadth", "dtc",
        "vol_ratio", "rs_watch", "breakout5", "above_avwap",
        "entry_date", "entry_price", "ret_1d", "ret_3d",
        "ret_5d", "ret_10d", "mfe_10d", "mae_10d",
    ]
    if events is None or events.empty:
        return pd.DataFrame(columns=columns)

    clean_tickers = list(dict.fromkeys(
        normalize_ticker(t) for t in tickers if normalize_ticker(t)
    ))[:max_tickers]
    if not clean_tickers:
        return pd.DataFrame(columns=columns)

    frames = _download_prices(clean_tickers, period="1y")
    if not frames:
        return pd.DataFrame(columns=columns)

    # Build a common recent trading calendar from downloaded prices.
    all_dates: set[pd.Timestamp] = set()
    for frame in frames.values():
        if frame is None or frame.empty:
            continue
        for idx in frame.index:
            all_dates.add(pd.Timestamp(idx).normalize())

    trade_dates = sorted(all_dates)
    if len(trade_dates) < 12:
        return pd.DataFrame(columns=columns)

    # Leave ten future sessions for outcome measurement.
    signal_dates = trade_dates[-(sessions + 10) : -10]
    if not signal_dates:
        return pd.DataFrame(columns=columns)

    rows: list[dict] = []
    for signal_date in signal_dates:
        short_metrics = build_short_metrics(
            events,
            window_days=75,
            as_of_date=signal_date,
        )
        if short_metrics.empty:
            continue

        short_metrics = short_metrics[
            short_metrics["ticker"].isin(clean_tickers)
        ].copy()
        if short_metrics.empty:
            continue

        price_features = _price_feature_table_asof(
            frames,
            short_metrics["ticker"].astype(str).tolist(),
            signal_date,
        )
        scored = score_short_cover(short_metrics, price_features)
        if scored.empty:
            continue

        scored = scored[scored["cover_score"] >= float(min_cover_score)].copy()
        scored = scored[
            scored["phase"].isin([
                "👀 COVER WATCH",
                "🔥 COVER EARLY",
                "✅ COVER CONFIRMED",
                "🚀 SQUEEZE",
            ])
        ]
        if scored.empty:
            continue

        for _, r in scored.iterrows():
            ticker = str(r["ticker"])
            frame = frames.get(ticker)
            if frame is None or frame.empty:
                continue

            normalized_index = pd.to_datetime(frame.index).normalize()
            matches = [i for i, d in enumerate(normalized_index) if d == signal_date]
            if not matches:
                continue
            signal_pos = matches[-1]

            stats = _forward_trade_stats(frame, signal_pos)
            if stats["entry_price"] is None:
                continue

            item = {
                "signal_date": pd.Timestamp(signal_date),
                "ticker": ticker,
                "name": r.get("name", ""),
                "phase": r.get("phase", ""),
                "regime": r.get("regime", ""),
                "cover_score": r.get("cover_score"),
                "short_pressure": r.get("short_pressure"),
                "absorption_score": r.get("absorption_score"),
                "ignition_score": r.get("ignition_score"),
                "long_demand_score": r.get("long_demand_score"),
                "confidence": r.get("confidence"),
                "short_ratio": r.get("short_ratio"),
                "delta_short": r.get("delta_short"),
                "cover_breadth": r.get("cover_breadth"),
                "dtc": r.get("dtc"),
                "vol_ratio": r.get("vol_ratio"),
                "rs_watch": r.get("rs_watch"),
                "breakout5": r.get("breakout5"),
                "above_avwap": r.get("above_avwap"),
            }
            item.update(stats)
            rows.append(item)

    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    out = out.sort_values(["signal_date", "cover_score"], ascending=[False, False])
    return out.reset_index(drop=True)


def summarize_backtest(backtest: pd.DataFrame, group_col: str = "phase") -> pd.DataFrame:
    """Aggregate signal count, win rates, returns, MFE and MAE by phase/regime."""
    columns = [
        group_col, "signals", "win_1d", "win_3d", "win_5d", "win_10d",
        "avg_1d", "avg_3d", "avg_5d", "avg_10d",
        "median_5d", "avg_mfe_10d", "avg_mae_10d",
    ]
    if backtest is None or backtest.empty or group_col not in backtest.columns:
        return pd.DataFrame(columns=columns)

    rows = []
    for key, g in backtest.groupby(group_col, dropna=False):
        def _win(col: str):
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            return float((s > 0).mean() * 100.0) if not s.empty else None

        def _avg(col: str):
            s = pd.to_numeric(g[col], errors="coerce").dropna()
            return float(s.mean()) if not s.empty else None

        s5 = pd.to_numeric(g["ret_5d"], errors="coerce").dropna()
        rows.append({
            group_col: key,
            "signals": int(len(g)),
            "win_1d": _win("ret_1d"),
            "win_3d": _win("ret_3d"),
            "win_5d": _win("ret_5d"),
            "win_10d": _win("ret_10d"),
            "avg_1d": _avg("ret_1d"),
            "avg_3d": _avg("ret_3d"),
            "avg_5d": _avg("ret_5d"),
            "avg_10d": _avg("ret_10d"),
            "median_5d": float(s5.median()) if not s5.empty else None,
            "avg_mfe_10d": _avg("mfe_10d"),
            "avg_mae_10d": _avg("mae_10d"),
        })

    return pd.DataFrame(rows).sort_values(
        ["signals", "avg_5d"],
        ascending=[False, False],
    ).reset_index(drop=True)



def _condition_stats(df: pd.DataFrame, return_col: str) -> dict:
    """Compact outcome statistics for one threshold condition."""
    s = pd.to_numeric(df.get(return_col), errors="coerce").dropna()
    mfe = pd.to_numeric(df.get("mfe_10d"), errors="coerce").dropna()
    mae = pd.to_numeric(df.get("mae_10d"), errors="coerce").dropna()
    if s.empty:
        return {
            "n": 0, "win": None, "avg": None, "median": None,
            "mfe": None, "mae": None,
        }
    return {
        "n": int(len(s)),
        "win": float((s > 0).mean() * 100.0),
        "avg": float(s.mean()),
        "median": float(s.median()),
        "mfe": float(mfe.mean()) if not mfe.empty else None,
        "mae": float(mae.mean()) if not mae.empty else None,
    }


def optimize_short_cover_thresholds(
    backtest: pd.DataFrame,
    horizon: int = 5,
    train_fraction: float = 0.65,
    min_train_signals: int = 8,
    min_test_signals: int = 4,
    top_train_candidates: int = 30,
) -> pd.DataFrame:
    """Search robust score thresholds with chronological train/holdout validation.

    Workflow:
    1. Split unique signal dates chronologically.
    2. Search threshold combinations on the earlier training period only.
    3. Keep only the best training candidates.
    4. Evaluate those candidates on the later holdout period.

    This deliberately avoids choosing thresholds from the entire dataset at once.
    """
    columns = [
        "cover_min", "ignition_min", "long_min", "pressure_min", "confidence_min",
        "train_signals", "train_win", "train_avg", "train_median",
        "test_signals", "test_win", "test_avg", "test_median",
        "test_mfe", "test_mae", "train_score", "stability_score",
        "robustness", "train_start", "train_end", "test_start", "test_end",
    ]
    if backtest is None or backtest.empty:
        return pd.DataFrame(columns=columns)

    horizon = int(horizon)
    if horizon not in (1, 3, 5, 10):
        horizon = 5
    return_col = f"ret_{horizon}d"
    if return_col not in backtest.columns:
        return pd.DataFrame(columns=columns)

    bt = backtest.copy()
    bt["signal_date"] = pd.to_datetime(bt["signal_date"], errors="coerce").dt.normalize()
    bt = bt.dropna(subset=["signal_date", return_col]).sort_values("signal_date")
    if bt.empty:
        return pd.DataFrame(columns=columns)

    unique_dates = sorted(bt["signal_date"].dropna().unique())
    if len(unique_dates) < 8:
        return pd.DataFrame(columns=columns)

    split_idx = int(len(unique_dates) * float(train_fraction))
    split_idx = max(4, min(len(unique_dates) - 3, split_idx))
    train_dates = set(unique_dates[:split_idx])
    test_dates = set(unique_dates[split_idx:])
    train = bt[bt["signal_date"].isin(train_dates)].copy()
    test = bt[bt["signal_date"].isin(test_dates)].copy()
    if train.empty or test.empty:
        return pd.DataFrame(columns=columns)

    cover_grid = [50, 55, 60, 65, 70, 75, 80]
    ignition_grid = [0, 50, 60, 70, 80]
    long_grid = [0, 50, 60, 70, 80]
    pressure_grid = [0, 40, 50, 60, 70]
    confidence_grid = [0, 40, 60]

    train_candidates = []
    seen_masks: set[tuple[int, ...]] = set()

    for cover_min in cover_grid:
        for ignition_min in ignition_grid:
            for long_min in long_grid:
                for pressure_min in pressure_grid:
                    for confidence_min in confidence_grid:
                        mask = (
                            (pd.to_numeric(train["cover_score"], errors="coerce") >= cover_min)
                            & (pd.to_numeric(train["ignition_score"], errors="coerce") >= ignition_min)
                            & (pd.to_numeric(train["long_demand_score"], errors="coerce") >= long_min)
                            & (pd.to_numeric(train["short_pressure"], errors="coerce") >= pressure_min)
                            & (pd.to_numeric(train["confidence"], errors="coerce") >= confidence_min)
                        )
                        idx_key = tuple(train.index[mask].tolist())
                        if idx_key in seen_masks:
                            continue
                        seen_masks.add(idx_key)

                        subset = train.loc[mask]
                        stats = _condition_stats(subset, return_col)
                        if stats["n"] < int(min_train_signals):
                            continue

                        avg = stats["avg"] or 0.0
                        median = stats["median"] or 0.0
                        win = stats["win"] or 0.0
                        mae = abs(stats["mae"] or 0.0)

                        # Reward positive expectancy and consistency, penalize adverse excursion.
                        train_score = (
                            avg
                            + 0.03 * (win - 50.0)
                            + 0.20 * median
                            - 0.12 * mae
                        )
                        train_candidates.append({
                            "cover_min": cover_min,
                            "ignition_min": ignition_min,
                            "long_min": long_min,
                            "pressure_min": pressure_min,
                            "confidence_min": confidence_min,
                            "train_stats": stats,
                            "train_score": float(train_score),
                        })

    if not train_candidates:
        return pd.DataFrame(columns=columns)

    train_candidates.sort(
        key=lambda x: (x["train_score"], x["train_stats"]["n"]),
        reverse=True,
    )
    train_candidates = train_candidates[: max(1, int(top_train_candidates))]

    rows = []
    for candidate in train_candidates:
        mask = (
            (pd.to_numeric(test["cover_score"], errors="coerce") >= candidate["cover_min"])
            & (pd.to_numeric(test["ignition_score"], errors="coerce") >= candidate["ignition_min"])
            & (pd.to_numeric(test["long_demand_score"], errors="coerce") >= candidate["long_min"])
            & (pd.to_numeric(test["short_pressure"], errors="coerce") >= candidate["pressure_min"])
            & (pd.to_numeric(test["confidence"], errors="coerce") >= candidate["confidence_min"])
        )
        test_stats = _condition_stats(test.loc[mask], return_col)
        train_stats = candidate["train_stats"]

        test_n = test_stats["n"]
        test_avg = test_stats["avg"]
        test_win = test_stats["win"]
        train_avg = train_stats["avg"] or 0.0

        enough_test = test_n >= int(min_test_signals)
        positive_holdout = enough_test and test_avg is not None and test_avg > 0
        win_holdout = enough_test and test_win is not None and test_win >= 50.0

        if not enough_test:
            robustness = "⚪ サンプル不足"
            stability = 0.0
        else:
            avg_component = max(0.0, min(40.0, (test_avg or 0.0) * 8.0))
            win_component = max(0.0, min(30.0, ((test_win or 0.0) - 45.0) * 1.5))
            sample_component = min(20.0, test_n / max(1, min_test_signals) * 10.0)
            degradation = abs((test_avg or 0.0) - train_avg)
            stability_component = max(0.0, 10.0 - degradation * 2.0)
            stability = min(
                100.0,
                avg_component + win_component + sample_component + stability_component,
            )

            if positive_holdout and win_holdout and stability >= 60:
                robustness = "🟢 ROBUST"
            elif positive_holdout and stability >= 35:
                robustness = "🟡 PROMISING"
            else:
                robustness = "🔴 UNSTABLE"

        rows.append({
            "cover_min": candidate["cover_min"],
            "ignition_min": candidate["ignition_min"],
            "long_min": candidate["long_min"],
            "pressure_min": candidate["pressure_min"],
            "confidence_min": candidate["confidence_min"],
            "train_signals": train_stats["n"],
            "train_win": train_stats["win"],
            "train_avg": train_stats["avg"],
            "train_median": train_stats["median"],
            "test_signals": test_stats["n"],
            "test_win": test_stats["win"],
            "test_avg": test_stats["avg"],
            "test_median": test_stats["median"],
            "test_mfe": test_stats["mfe"],
            "test_mae": test_stats["mae"],
            "train_score": candidate["train_score"],
            "stability_score": round(float(stability), 1),
            "robustness": robustness,
            "train_start": pd.Timestamp(min(train_dates)),
            "train_end": pd.Timestamp(max(train_dates)),
            "test_start": pd.Timestamp(min(test_dates)),
            "test_end": pd.Timestamp(max(test_dates)),
        })

    out = pd.DataFrame(rows)
    if out.empty:
        return pd.DataFrame(columns=columns)

    robustness_rank = {
        "🟢 ROBUST": 3,
        "🟡 PROMISING": 2,
        "🔴 UNSTABLE": 1,
        "⚪ サンプル不足": 0,
    }
    out["_robust_rank"] = out["robustness"].map(robustness_rank).fillna(0)
    out = out.sort_values(
        ["_robust_rank", "stability_score", "test_avg", "train_score"],
        ascending=False,
    ).drop(columns="_robust_rank")
    return out.reset_index(drop=True)



def apply_optimizer_condition(
    current: pd.DataFrame,
    condition,
) -> pd.DataFrame:
    """Apply one optimizer threshold row to the current live ranking.

    Returns the original rows plus:
    - optimizer_match: all five thresholds are satisfied
    - optimizer_label: ROBUST/PROMISING match label
    - match_strength: 0-100 score based on how far the row clears thresholds
    - condition_text: compact threshold description
    """
    if current is None or current.empty:
        return pd.DataFrame() if current is None else current.copy()

    out = current.copy()
    out["optimizer_match"] = False
    out["optimizer_label"] = ""
    out["match_strength"] = 0.0
    out["condition_text"] = ""

    if condition is None:
        return out

    try:
        robustness = str(condition.get("robustness", ""))
    except Exception:
        return out

    if robustness not in {"🟢 ROBUST", "🟡 PROMISING"}:
        return out

    thresholds = {
        "cover_score": float(condition.get("cover_min", 0) or 0),
        "ignition_score": float(condition.get("ignition_min", 0) or 0),
        "long_demand_score": float(condition.get("long_min", 0) or 0),
        "short_pressure": float(condition.get("pressure_min", 0) or 0),
        "confidence": float(condition.get("confidence_min", 0) or 0),
    }

    numeric = {}
    for col, threshold in thresholds.items():
        numeric[col] = pd.to_numeric(out.get(col), errors="coerce").fillna(-1e9)

    mask = pd.Series(True, index=out.index)
    for col, threshold in thresholds.items():
        mask &= numeric[col] >= threshold

    out["optimizer_match"] = mask

    label = "⭐ ROBUST MATCH" if robustness == "🟢 ROBUST" else "🟡 PROMISING MATCH"
    out.loc[mask, "optimizer_label"] = label

    # Strength rewards clearance above each threshold, capped at +20 points per factor.
    strength = pd.Series(0.0, index=out.index)
    active_factors = 0
    for col, threshold in thresholds.items():
        # Threshold 0 means the optimizer did not require this factor.
        if threshold <= 0:
            continue
        active_factors += 1
        excess = (numeric[col] - threshold).clip(lower=0, upper=20)
        strength += excess / 20.0 * 100.0

    if active_factors > 0:
        strength = strength / active_factors
    out["match_strength"] = strength.where(mask, 0.0).round(1)

    condition_text = (
        f"C{int(thresholds['cover_score'])}/"
        f"I{int(thresholds['ignition_score'])}/"
        f"L{int(thresholds['long_demand_score'])}/"
        f"P{int(thresholds['short_pressure'])}/"
        f"Q{int(thresholds['confidence'])}"
    )
    out["condition_text"] = condition_text
    return out



def build_priority_alerts(
    current: pd.DataFrame,
    promotions: pd.DataFrame | None = None,
    limit: int = 5,
) -> pd.DataFrame:
    """Rank today's Short Cover candidates by review priority.

    This is an attention-priority score, not a return forecast or buy signal.
    It deliberately emphasizes:
    1) validation against ROBUST/PROMISING historical thresholds,
    2) a fresh day-over-day promotion,
    3) current phase/regime,
    4) data confidence and volume confirmation.
    """
    columns = [
        "ticker", "name", "alert_score", "alert_tier", "alert_reason",
        "phase", "regime", "optimizer_label", "match_strength",
        "cover_score", "ignition_score", "long_demand_score",
        "short_pressure", "confidence", "vol_ratio", "rs_watch",
        "is_promotion", "promotion_reason",
    ]
    if current is None or current.empty:
        return pd.DataFrame(columns=columns)

    out = current.copy()

    promo_map: dict[str, dict] = {}
    if promotions is not None and not promotions.empty and "ticker" in promotions.columns:
        for _, r in promotions.iterrows():
            promo_map[str(r.get("ticker", ""))] = r.to_dict()

    phase_points = {
        "NORMAL": 0.0,
        "🧱 SHORT BUILDUP": 20.0,
        "👀 COVER WATCH": 50.0,
        "🔥 COVER EARLY": 78.0,
        "✅ COVER CONFIRMED": 90.0,
        "🚀 SQUEEZE": 100.0,
    }
    regime_points = {
        "🔥 COVER + NEW MONEY": 100.0,
        "⚠️ PURE SHORT COVER": 60.0,
        "🟢 NEW MONEY": 55.0,
        "⚪ NEUTRAL": 25.0,
    }

    rows = []
    for _, r in out.iterrows():
        ticker = str(r.get("ticker", ""))
        promo = promo_map.get(ticker)

        label = str(r.get("optimizer_label", "") or "")
        match_strength = (
            float(r.get("match_strength"))
            if pd.notna(r.get("match_strength"))
            else 0.0
        )
        if label == "⭐ ROBUST MATCH":
            validation = min(100.0, 70.0 + match_strength * 0.30)
        elif label == "🟡 PROMISING MATCH":
            validation = min(85.0, 45.0 + match_strength * 0.25)
        else:
            validation = 0.0

        freshness = 0.0
        is_promotion = promo is not None
        promotion_reason = ""
        if promo is not None:
            phase_jump = max(0.0, float(promo.get("phase_jump") or 0.0))
            cover_delta = max(0.0, float(promo.get("cover_delta") or 0.0))
            ignition_delta = max(0.0, float(promo.get("ignition_delta") or 0.0))
            freshness = 35.0
            freshness += min(30.0, phase_jump * 18.0)
            freshness += min(15.0, cover_delta * 0.8)
            freshness += min(10.0, ignition_delta * 0.4)
            if _truthy(promo.get("fresh_breakout")):
                freshness += 7.0
            if _truthy(promo.get("fresh_avwap_reclaim")):
                freshness += 5.0
            freshness = min(100.0, freshness)
            promotion_reason = str(promo.get("promotion_reason", "") or "")

        phase = str(r.get("phase", "NORMAL"))
        regime = str(r.get("regime", "⚪ NEUTRAL"))
        phase_component = phase_points.get(phase, 0.0)
        regime_component = regime_points.get(regime, 25.0)

        confidence = (
            max(0.0, min(100.0, float(r.get("confidence"))))
            if pd.notna(r.get("confidence"))
            else 0.0
        )
        vol = float(r.get("vol_ratio")) if pd.notna(r.get("vol_ratio")) else 0.0
        volume_component = max(0.0, min(100.0, (vol - 1.0) / 2.0 * 100.0))

        alert_score = (
            validation * 0.30
            + freshness * 0.25
            + phase_component * 0.20
            + regime_component * 0.10
            + confidence * 0.10
            + volume_component * 0.05
        )
        alert_score = round(min(100.0, max(0.0, alert_score)), 1)

        if alert_score >= 80:
            tier = "🚨 A+ 最優先確認"
        elif alert_score >= 70:
            tier = "🔥 A 優先確認"
        elif alert_score >= 58:
            tier = "🟡 B 監視"
        else:
            tier = "⚪ C 通常"

        reasons = []
        if label:
            reasons.append(label)
        if is_promotion:
            reasons.append("⚡ 今日昇格")
        if phase in {"🔥 COVER EARLY", "✅ COVER CONFIRMED", "🚀 SQUEEZE"}:
            reasons.append(phase)
        if regime == "🔥 COVER + NEW MONEY":
            reasons.append("新規資金併走")
        if vol >= 2.0:
            reasons.append(f"出来高{vol:.1f}x")
        elif vol >= 1.5:
            reasons.append(f"出来高{vol:.1f}x")
        if confidence >= 70:
            reasons.append("信頼度高")

        item = r.to_dict()
        item.update({
            "alert_score": alert_score,
            "alert_tier": tier,
            "alert_reason": " / ".join(reasons[:5]) if reasons else "通常監視",
            "is_promotion": is_promotion,
            "promotion_reason": promotion_reason,
        })
        rows.append(item)

    result = pd.DataFrame(rows)
    result = result.sort_values(
        ["alert_score", "match_strength", "cover_score", "ignition_score"],
        ascending=False,
    ).reset_index(drop=True)
    if limit and int(limit) > 0:
        result = result.head(int(limit)).copy()
    return result



ALERT_HISTORY_COLUMNS = [
    "alert_date", "ticker", "name", "alert_tier", "alert_score",
    "phase", "regime", "optimizer_label", "match_strength",
    "cover_score", "ignition_score", "long_demand_score",
    "short_pressure", "confidence", "vol_ratio", "rs_watch",
    "alert_price", "alert_reason", "promotion_reason", "condition_text",
    "entry_date", "entry_price", "ret_1d", "ret_3d",
    "ret_5d", "ret_10d", "mfe_10d", "mae_10d",
    "outcome_status", "last_updated",
]


def normalize_alert_history(history: pd.DataFrame | None) -> pd.DataFrame:
    """Return a stable alert-history schema suitable for CSV persistence."""
    if history is None or history.empty:
        return pd.DataFrame(columns=ALERT_HISTORY_COLUMNS)

    out = history.copy()
    for col in ALERT_HISTORY_COLUMNS:
        if col not in out.columns:
            out[col] = None

    for col in ["alert_date", "entry_date", "last_updated"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    numeric_cols = [
        "alert_score", "match_strength", "cover_score", "ignition_score",
        "long_demand_score", "short_pressure", "confidence", "vol_ratio",
        "rs_watch", "alert_price", "entry_price", "ret_1d", "ret_3d",
        "ret_5d", "ret_10d", "mfe_10d", "mae_10d",
    ]
    for col in numeric_cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["ticker"] = out["ticker"].map(normalize_ticker)
    out = out[out["ticker"] != ""].copy()

    # If the same alert is merged from local/GitHub copies, prefer the most
    # recently updated record, then the higher alert score. This preserves
    # newly calculated outcomes instead of accidentally keeping an older row.
    out["_last_updated_sort"] = pd.to_datetime(out["last_updated"], errors="coerce")
    out = out.sort_values(
        ["alert_date", "_last_updated_sort", "alert_score"],
        ascending=[False, False, False],
        na_position="last",
    )
    out = out.drop_duplicates(subset=["alert_date", "ticker"], keep="first")
    out = out.drop(columns=["_last_updated_sort"], errors="ignore")
    return out[ALERT_HISTORY_COLUMNS].reset_index(drop=True)


def append_priority_alert_history(
    history: pd.DataFrame | None,
    priority_alerts: pd.DataFrame,
    min_tiers: tuple[str, ...] = ("🚨 A+ 最優先確認", "🔥 A 優先確認"),
) -> tuple[pd.DataFrame, int]:
    """Append today's actionable alerts once per alert-date/ticker.

    ROBUST/PROMISING matches are also stored even if their priority tier is below A.
    """
    base = normalize_alert_history(history)
    if priority_alerts is None or priority_alerts.empty:
        return base, 0

    new_rows = []
    for _, r in priority_alerts.iterrows():
        tier = str(r.get("alert_tier", "") or "")
        optimizer_label = str(r.get("optimizer_label", "") or "")
        should_store = tier in min_tiers or optimizer_label in {
            "⭐ ROBUST MATCH",
            "🟡 PROMISING MATCH",
        }
        if not should_store:
            continue

        alert_date = r.get("snapshot_date")
        if pd.isna(alert_date):
            alert_date = pd.Timestamp(datetime.now().date())
        alert_date = pd.Timestamp(alert_date).normalize()

        price = _to_float(r.get("price"))
        new_rows.append({
            "alert_date": alert_date,
            "ticker": normalize_ticker(r.get("ticker")),
            "name": str(r.get("name", "") or ""),
            "alert_tier": tier,
            "alert_score": r.get("alert_score"),
            "phase": r.get("phase"),
            "regime": r.get("regime"),
            "optimizer_label": optimizer_label,
            "match_strength": r.get("match_strength"),
            "cover_score": r.get("cover_score"),
            "ignition_score": r.get("ignition_score"),
            "long_demand_score": r.get("long_demand_score"),
            "short_pressure": r.get("short_pressure"),
            "confidence": r.get("confidence"),
            "vol_ratio": r.get("vol_ratio"),
            "rs_watch": r.get("rs_watch"),
            "alert_price": price,
            "alert_reason": r.get("alert_reason", ""),
            "promotion_reason": r.get("promotion_reason", ""),
            "condition_text": r.get("condition_text", ""),
            "entry_date": pd.NaT,
            "entry_price": None,
            "ret_1d": None,
            "ret_3d": None,
            "ret_5d": None,
            "ret_10d": None,
            "mfe_10d": None,
            "mae_10d": None,
            "outcome_status": "⏳ 追跡中",
            "last_updated": pd.Timestamp.now(),
        })

    if not new_rows:
        return base, 0

    incoming = pd.DataFrame(new_rows)
    incoming = normalize_alert_history(incoming)

    existing_keys = set(
        zip(
            pd.to_datetime(base["alert_date"], errors="coerce").dt.normalize(),
            base["ticker"].astype(str),
        )
    ) if not base.empty else set()

    fresh = incoming[
        ~incoming.apply(
            lambda r: (pd.Timestamp(r["alert_date"]).normalize(), str(r["ticker"])) in existing_keys,
            axis=1,
        )
    ].copy()

    if fresh.empty:
        return base, 0

    combined = normalize_alert_history(pd.concat([fresh, base], ignore_index=True))
    return combined, int(len(fresh))


def update_alert_history_outcomes(
    history: pd.DataFrame | None,
    period: str = "1y",
) -> pd.DataFrame:
    """Fill forward outcomes for persisted alerts using next-session-open entry.

    The calculation matches the backtest convention:
    signal/alert at session close -> next session open entry.
    """
    out = normalize_alert_history(history)
    if out.empty:
        return out

    tickers = out["ticker"].dropna().astype(str).unique().tolist()
    frames = _download_prices(tickers, period=period)
    if not frames:
        return out

    now_ts = pd.Timestamp.now()
    for idx, row in out.iterrows():
        ticker = str(row["ticker"])
        alert_date = pd.to_datetime(row["alert_date"], errors="coerce")
        frame = frames.get(ticker)
        if pd.isna(alert_date) or frame is None or frame.empty:
            continue

        normalized_index = pd.to_datetime(frame.index).normalize()
        prior_positions = [
            i for i, d in enumerate(normalized_index)
            if d <= pd.Timestamp(alert_date).normalize()
        ]
        if not prior_positions:
            continue

        signal_pos = prior_positions[-1]
        # Require exact trading-date match so a weekend app run doesn't shift the signal.
        if normalized_index[signal_pos] != pd.Timestamp(alert_date).normalize():
            continue

        stats = _forward_trade_stats(frame, signal_pos)
        if stats["entry_price"] is None:
            out.at[idx, "outcome_status"] = "⏳ 翌営業日待ち"
            out.at[idx, "last_updated"] = now_ts
            continue

        out.at[idx, "entry_date"] = stats["entry_date"]
        out.at[idx, "entry_price"] = stats["entry_price"]
        for col in ["ret_1d", "ret_3d", "ret_5d", "ret_10d", "mfe_10d", "mae_10d"]:
            out.at[idx, col] = stats[col]

        if stats["ret_10d"] is not None:
            status = "✅ 10日完了"
        elif stats["ret_5d"] is not None:
            status = "📈 5日経過"
        elif stats["ret_3d"] is not None:
            status = "📊 3日経過"
        elif stats["ret_1d"] is not None:
            status = "🌱 1日経過"
        else:
            status = "⏳ 追跡中"
        out.at[idx, "outcome_status"] = status
        out.at[idx, "last_updated"] = now_ts

    return normalize_alert_history(out)


def summarize_alert_history(history: pd.DataFrame | None) -> dict:
    """Headline performance metrics for the persisted real alert log."""
    h = normalize_alert_history(history)
    if h.empty:
        return {
            "alerts": 0, "tracked": 0, "win_5d": None,
            "avg_5d": None, "avg_10d": None,
        }
    r5 = pd.to_numeric(h["ret_5d"], errors="coerce").dropna()
    r10 = pd.to_numeric(h["ret_10d"], errors="coerce").dropna()
    tracked = int(pd.to_numeric(h["entry_price"], errors="coerce").notna().sum())
    return {
        "alerts": int(len(h)),
        "tracked": tracked,
        "win_5d": float((r5 > 0).mean() * 100.0) if not r5.empty else None,
        "avg_5d": float(r5.mean()) if not r5.empty else None,
        "avg_10d": float(r10.mean()) if not r10.empty else None,
    }



def condition_text_from_row(condition) -> str:
    """Return the compact C/I/L/P/Q identity used by live history."""
    if condition is None:
        return ""
    try:
        return (
            f"C{int(float(condition.get('cover_min', 0) or 0))}/"
            f"I{int(float(condition.get('ignition_min', 0) or 0))}/"
            f"L{int(float(condition.get('long_min', 0) or 0))}/"
            f"P{int(float(condition.get('pressure_min', 0) or 0))}/"
            f"Q{int(float(condition.get('confidence_min', 0) or 0))}"
        )
    except Exception:
        return ""


def _filter_backtest_by_condition(backtest: pd.DataFrame, condition) -> pd.DataFrame:
    if backtest is None or backtest.empty or condition is None:
        return pd.DataFrame() if backtest is None else backtest.copy()

    thresholds = {
        "cover_score": float(condition.get("cover_min", 0) or 0),
        "ignition_score": float(condition.get("ignition_min", 0) or 0),
        "long_demand_score": float(condition.get("long_min", 0) or 0),
        "short_pressure": float(condition.get("pressure_min", 0) or 0),
        "confidence": float(condition.get("confidence_min", 0) or 0),
    }
    mask = pd.Series(True, index=backtest.index)
    for col, threshold in thresholds.items():
        vals = pd.to_numeric(backtest.get(col), errors="coerce")
        mask &= vals >= threshold
    return backtest.loc[mask].copy()


def compare_live_vs_backtest(
    backtest: pd.DataFrame | None,
    history: pd.DataFrame | None,
    condition=None,
    horizon: int = 5,
    recent_live_n: int = 20,
    min_live_signals: int = 5,
) -> dict:
    """Compare live alert performance with the historical benchmark for one condition.

    The result is a drift/health monitor, not a forecast.
    """
    horizon = int(horizon)
    if horizon not in (1, 3, 5, 10):
        horizon = 5
    ret_col = f"ret_{horizon}d"

    result = {
        "status": "⚪ DATA BUILDING",
        "health_score": None,
        "condition_text": condition_text_from_row(condition),
        "backtest_n": 0,
        "live_n": 0,
        "backtest_avg": None,
        "live_avg": None,
        "avg_drift": None,
        "backtest_win": None,
        "live_win": None,
        "win_drift": None,
        "backtest_mfe": None,
        "live_mfe": None,
        "backtest_mae": None,
        "live_mae": None,
        "message": "実運用サンプルを蓄積中です。",
    }

    if backtest is None or backtest.empty:
        result["message"] = "比較できるバックテスト結果がありません。"
        return result

    bt = _filter_backtest_by_condition(backtest, condition)
    if ret_col not in bt.columns:
        result["message"] = f"{horizon}日リターンのバックテスト列がありません。"
        return result

    bt_ret = pd.to_numeric(bt[ret_col], errors="coerce").dropna()
    if bt_ret.empty:
        result["message"] = "現在条件に一致するバックテストサンプルがありません。"
        return result

    hist = normalize_alert_history(history)
    if hist.empty:
        result["backtest_n"] = int(len(bt_ret))
        result["backtest_avg"] = float(bt_ret.mean())
        result["backtest_win"] = float((bt_ret > 0).mean() * 100.0)
        return result

    condition_id = result["condition_text"]
    if condition_id:
        hist = hist[hist["condition_text"].fillna("").astype(str) == condition_id].copy()

    if ret_col not in hist.columns:
        return result

    hist[ret_col] = pd.to_numeric(hist[ret_col], errors="coerce")
    hist = hist.dropna(subset=[ret_col]).sort_values("alert_date")
    if recent_live_n and recent_live_n > 0:
        hist = hist.tail(int(recent_live_n))

    live_ret = hist[ret_col].dropna()

    bt_avg = float(bt_ret.mean())
    bt_win = float((bt_ret > 0).mean() * 100.0)
    bt_mfe_s = pd.to_numeric(bt.get("mfe_10d"), errors="coerce").dropna()
    bt_mae_s = pd.to_numeric(bt.get("mae_10d"), errors="coerce").dropna()

    result.update({
        "backtest_n": int(len(bt_ret)),
        "backtest_avg": bt_avg,
        "backtest_win": bt_win,
        "backtest_mfe": float(bt_mfe_s.mean()) if not bt_mfe_s.empty else None,
        "backtest_mae": float(bt_mae_s.mean()) if not bt_mae_s.empty else None,
        "live_n": int(len(live_ret)),
    })

    if len(live_ret) < int(min_live_signals):
        result["message"] = (
            f"同一条件の実運用{horizon}日結果が{len(live_ret)}件。"
            f"{min_live_signals}件までは劣化判定を保留します。"
        )
        return result

    live_avg = float(live_ret.mean())
    live_win = float((live_ret > 0).mean() * 100.0)
    live_mfe_s = pd.to_numeric(hist.get("mfe_10d"), errors="coerce").dropna()
    live_mae_s = pd.to_numeric(hist.get("mae_10d"), errors="coerce").dropna()
    avg_drift = live_avg - bt_avg
    win_drift = live_win - bt_win

    # Tolerances expand with the historical edge so we do not overreact to normal noise.
    avg_tolerance = max(1.0, abs(bt_avg) * 0.50)
    severe_avg_tolerance = max(2.0, abs(bt_avg) * 1.00)

    avg_penalty = max(0.0, -avg_drift / avg_tolerance) * 35.0
    win_penalty = max(0.0, -win_drift / 10.0) * 20.0
    health = max(0.0, min(100.0, 100.0 - avg_penalty - win_penalty))

    if avg_drift >= -avg_tolerance and win_drift >= -10.0:
        status = "🟢 STABLE"
        message = "実運用成績はバックテストの想定レンジ内です。"
    elif avg_drift >= -severe_avg_tolerance and win_drift >= -20.0:
        status = "🟡 WATCH"
        message = "実運用成績が弱含み。サンプル追加と条件の再検証を優先します。"
    else:
        status = "🔴 DEGRADED"
        message = "実運用成績がバックテストから大きく悪化しています。条件の再最適化候補です。"

    result.update({
        "status": status,
        "health_score": round(health, 1),
        "live_avg": live_avg,
        "avg_drift": avg_drift,
        "live_win": live_win,
        "win_drift": win_drift,
        "live_mfe": float(live_mfe_s.mean()) if not live_mfe_s.empty else None,
        "live_mae": float(live_mae_s.mean()) if not live_mae_s.empty else None,
        "message": message,
    })
    return result



def build_reoptimization_comparison(
    backtest: pd.DataFrame | None,
    active_condition,
    horizon: int = 5,
    recent_fraction: float = 0.65,
    train_fraction: float = 0.65,
    min_train_signals: int = 6,
    min_test_signals: int = 3,
) -> dict:
    """Build an old-vs-reoptimized comparison on a recent chronological window.

    The new candidate is optimized only inside the recent window and both the old
    and new conditions are compared on the *same* recent holdout dates. Nothing
    is adopted automatically.
    """
    result = {
        "status": "⚪ INSUFFICIENT",
        "message": "再最適化に必要なデータが不足しています。",
        "old_condition": condition_text_from_row(active_condition),
        "new_condition": "",
        "old_n": 0, "new_n": 0,
        "old_avg": None, "new_avg": None,
        "old_win": None, "new_win": None,
        "old_mfe": None, "new_mfe": None,
        "old_mae": None, "new_mae": None,
        "avg_improvement": None,
        "win_improvement": None,
        "candidate_robustness": "",
        "candidate_stability": None,
        "holdout_start": None,
        "holdout_end": None,
        "candidate": None,
    }
    if backtest is None or backtest.empty or active_condition is None:
        return result

    horizon = int(horizon)
    if horizon not in (1, 3, 5, 10):
        horizon = 5
    ret_col = f"ret_{horizon}d"

    bt = backtest.copy()
    bt["signal_date"] = pd.to_datetime(bt["signal_date"], errors="coerce").dt.normalize()
    bt = bt.dropna(subset=["signal_date", ret_col]).sort_values("signal_date")
    dates = sorted(bt["signal_date"].unique())
    if len(dates) < 10:
        return result

    keep_n = max(10, int(round(len(dates) * float(recent_fraction))))
    recent_dates = set(dates[-keep_n:])
    recent_bt = bt[bt["signal_date"].isin(recent_dates)].copy()

    candidates = optimize_short_cover_thresholds(
        recent_bt,
        horizon=horizon,
        train_fraction=train_fraction,
        min_train_signals=min_train_signals,
        min_test_signals=min_test_signals,
        top_train_candidates=30,
    )
    if candidates.empty:
        result["message"] = "最近のデータでは再最適化候補を作れませんでした。"
        return result

    preferred = candidates[
        candidates["robustness"].isin(["🟢 ROBUST", "🟡 PROMISING"])
    ]
    candidate = preferred.iloc[0] if not preferred.empty else candidates.iloc[0]

    test_start = pd.Timestamp(candidate["test_start"]).normalize()
    test_end = pd.Timestamp(candidate["test_end"]).normalize()
    holdout = recent_bt[
        (recent_bt["signal_date"] >= test_start)
        & (recent_bt["signal_date"] <= test_end)
    ].copy()
    if holdout.empty:
        result["message"] = "共通ホールドアウト期間を作れませんでした。"
        return result

    old_rows = _filter_backtest_by_condition(holdout, active_condition)
    new_rows = _filter_backtest_by_condition(holdout, candidate)
    old_stats = _condition_stats(old_rows, ret_col)
    new_stats = _condition_stats(new_rows, ret_col)

    result.update({
        "old_n": old_stats["n"],
        "new_n": new_stats["n"],
        "old_avg": old_stats["avg"],
        "new_avg": new_stats["avg"],
        "old_win": old_stats["win"],
        "new_win": new_stats["win"],
        "old_mfe": old_stats["mfe"],
        "new_mfe": new_stats["mfe"],
        "old_mae": old_stats["mae"],
        "new_mae": new_stats["mae"],
        "new_condition": condition_text_from_row(candidate),
        "candidate_robustness": candidate.get("robustness", ""),
        "candidate_stability": candidate.get("stability_score"),
        "holdout_start": test_start,
        "holdout_end": test_end,
        "candidate": candidate.to_dict(),
    })

    if old_stats["avg"] is not None and new_stats["avg"] is not None:
        result["avg_improvement"] = float(new_stats["avg"] - old_stats["avg"])
    if old_stats["win"] is not None and new_stats["win"] is not None:
        result["win_improvement"] = float(new_stats["win"] - old_stats["win"])

    if new_stats["n"] < int(min_test_signals):
        result["status"] = "⚪ INSUFFICIENT"
        result["message"] = "新条件の共通検証サンプルが不足しています。"
        return result

    avg_gain = result["avg_improvement"]
    win_gain = result["win_improvement"]
    robust = str(candidate.get("robustness", "")) in {"🟢 ROBUST", "🟡 PROMISING"}

    if (
        robust
        and avg_gain is not None and avg_gain > 0.75
        and (win_gain is None or win_gain >= -5.0)
    ):
        result["status"] = "🟢 RESEARCH CANDIDATE"
        result["message"] = "最近の共通ホールドアウトでは新条件が旧条件を上回っています。採用前に追加検証対象です。"
    elif avg_gain is not None and avg_gain > 0:
        result["status"] = "🟡 SMALL IMPROVEMENT"
        result["message"] = "改善は見られますが、差はまだ小さいため旧条件を維持して観察します。"
    else:
        result["status"] = "⚪ KEEP CURRENT"
        result["message"] = "最近の共通検証では、新条件へ替える明確な優位性は確認できません。"

    return result



CONDITION_VERSION_COLUMNS = [
    "version_id", "created_at", "activated_at", "is_active", "source",
    "note", "condition_text", "cover_min", "ignition_min", "long_min",
    "pressure_min", "confidence_min", "horizon", "robustness",
    "stability_score", "test_signals", "test_win", "test_avg",
    "test_mfe", "test_mae",
]


def normalize_condition_versions(versions: pd.DataFrame | None) -> pd.DataFrame:
    if versions is None or versions.empty:
        return pd.DataFrame(columns=CONDITION_VERSION_COLUMNS)

    out = versions.copy()
    for col in CONDITION_VERSION_COLUMNS:
        if col not in out.columns:
            out[col] = None

    for col in ["created_at", "activated_at"]:
        out[col] = pd.to_datetime(out[col], errors="coerce")

    for col in [
        "cover_min", "ignition_min", "long_min", "pressure_min",
        "confidence_min", "horizon", "stability_score", "test_signals",
        "test_win", "test_avg", "test_mfe", "test_mae",
    ]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    def _bool(v):
        if isinstance(v, bool):
            return v
        return str(v).strip().lower() in {"1", "true", "yes", "y"}

    out["is_active"] = out["is_active"].map(_bool)
    out["version_id"] = out["version_id"].fillna("").astype(str)
    out = out[out["version_id"] != ""].copy()
    out = out.drop_duplicates(subset=["version_id"], keep="last")

    # At most one active condition; keep the most recently activated.
    active = out[out["is_active"]].sort_values("activated_at")
    if len(active) > 1:
        keep_idx = active.index[-1]
        out.loc[out.index != keep_idx, "is_active"] = False

    return out[CONDITION_VERSION_COLUMNS].sort_values(
        ["created_at", "version_id"],
        ascending=[False, False],
    ).reset_index(drop=True)


def next_condition_version_id(versions: pd.DataFrame | None) -> str:
    v = normalize_condition_versions(versions)
    max_minor = -1
    for value in v["version_id"].astype(str).tolist():
        m = re.fullmatch(r"v1\.(\d+)", value.strip(), flags=re.I)
        if m:
            max_minor = max(max_minor, int(m.group(1)))
    return f"v1.{max_minor + 1}"


def append_condition_version(
    versions: pd.DataFrame | None,
    condition,
    *,
    source: str,
    horizon: int,
    note: str = "",
    activate: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Append one immutable condition snapshot; activation is explicit."""
    base = normalize_condition_versions(versions)
    if condition is None:
        return base, ""

    version_id = next_condition_version_id(base)
    row = {
        "version_id": version_id,
        "created_at": pd.Timestamp.now(),
        "activated_at": pd.Timestamp.now() if activate else pd.NaT,
        "is_active": bool(activate),
        "source": str(source or ""),
        "note": str(note or ""),
        "condition_text": condition_text_from_row(condition),
        "cover_min": condition.get("cover_min"),
        "ignition_min": condition.get("ignition_min"),
        "long_min": condition.get("long_min"),
        "pressure_min": condition.get("pressure_min"),
        "confidence_min": condition.get("confidence_min"),
        "horizon": int(horizon),
        "robustness": condition.get("robustness", ""),
        "stability_score": condition.get("stability_score"),
        "test_signals": condition.get("test_signals"),
        "test_win": condition.get("test_win"),
        "test_avg": condition.get("test_avg"),
        "test_mfe": condition.get("test_mfe"),
        "test_mae": condition.get("test_mae"),
    }

    if activate and not base.empty:
        base["is_active"] = False

    combined = pd.concat([pd.DataFrame([row]), base], ignore_index=True)
    return normalize_condition_versions(combined), version_id


def activate_condition_version(
    versions: pd.DataFrame | None,
    version_id: str,
) -> pd.DataFrame:
    """Activate one saved version and deactivate all others."""
    out = normalize_condition_versions(versions)
    if out.empty or version_id not in set(out["version_id"].astype(str)):
        return out
    out["is_active"] = out["version_id"].astype(str) == str(version_id)
    out.loc[out["is_active"], "activated_at"] = pd.Timestamp.now()
    return normalize_condition_versions(out)


def get_active_condition_version(versions: pd.DataFrame | None):
    """Return the active saved condition as a Series compatible with optimizer rows."""
    out = normalize_condition_versions(versions)
    active = out[out["is_active"]]
    if active.empty:
        return None
    row = active.sort_values("activated_at", ascending=False).iloc[0].copy()
    # Saved version rows use the same threshold column names expected downstream.
    return row


def candidate_tickers(short_metrics: pd.DataFrame, limit: int = 60) -> list[str]:
    if short_metrics is None or short_metrics.empty:
        return []
    s = short_metrics.copy()
    # Prefer evidence of either pressure or recent change/breadth; avoid sending hundreds to yfinance.
    activity = s["delta_short"].abs().fillna(0) * 20 + s["short_pressure_base"].fillna(0)
    s = s.assign(_activity=activity).sort_values(["_activity", "short_ratio"], ascending=False)
    return s["ticker"].dropna().astype(str).head(limit).tolist()

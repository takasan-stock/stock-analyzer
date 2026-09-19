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


def discover_jpx_excel_urls(archive_pages: int = 2, timeout: int = 20) -> tuple[list[str], list[str]]:
    """Discover public JPX short-position Excel files from the current page and archives.

    archive_pages=2 means current month + two archive pages (roughly 2-3 months).
    The JPX page is the source of truth; no hard-coded daily filenames are used.
    """
    pages = [JPX_INDEX] + [_archive_url(i) for i in range(1, archive_pages + 1)]
    urls: list[str] = []
    errors: list[str] = []
    seen: set[str] = set()
    s = _session()

    for page in pages:
        try:
            r = s.get(page, timeout=timeout)
            r.raise_for_status()
            text = r.text
            hrefs = re.findall(r'href=["\']([^"\']+\.(?:xlsx|xls)(?:\?[^"\']*)?)["\']', text, flags=re.I)
            for href in hrefs:
                full = urljoin(page, href)
                if full not in seen:
                    seen.add(full)
                    urls.append(full)
        except Exception as e:
            errors.append(f"{page}: {e}")

    return urls, errors


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


def parse_jpx_excel(content: bytes, source_url: str = "") -> pd.DataFrame:
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
        out = out[(out["ticker"] != "") & out["short_ratio"].notna()]
        if not out.empty:
            frames.append(out)

    if not frames:
        return pd.DataFrame(columns=["ticker", "name", "seller", "calc_date", "short_ratio", "short_shares", "source_url"])

    result = pd.concat(frames, ignore_index=True)
    result = result.drop_duplicates(subset=["ticker", "seller", "calc_date", "short_ratio", "short_shares"], keep="last")
    return result


def load_jpx_events(archive_pages: int = 2, max_files: int = 70, workers: int = 8, timeout: int = 25) -> JpxLoadResult:
    urls, errors = discover_jpx_excel_urls(archive_pages=archive_pages, timeout=timeout)
    urls = urls[:max_files]
    frames: list[pd.DataFrame] = []

    def fetch_one(url: str) -> tuple[str, pd.DataFrame | None, str | None]:
        try:
            s = _session()
            r = s.get(url, timeout=timeout)
            r.raise_for_status()
            return url, parse_jpx_excel(r.content, source_url=url), None
        except Exception as e:
            return url, None, str(e)

    with ThreadPoolExecutor(max_workers=max(1, min(workers, 10))) as ex:
        futures = [ex.submit(fetch_one, u) for u in urls]
        for fut in as_completed(futures):
            url, frame, err = fut.result()
            if err:
                errors.append(f"{url}: {err}")
            elif frame is not None and not frame.empty:
                frames.append(frame)

    events = (
        pd.concat(frames, ignore_index=True)
        if frames
        else pd.DataFrame(columns=["ticker", "name", "seller", "calc_date", "short_ratio", "short_shares", "source_url"])
    )
    if not events.empty:
        events["calc_date"] = pd.to_datetime(events["calc_date"], errors="coerce")
        events = events.sort_values(["ticker", "seller", "calc_date"]).drop_duplicates(
            subset=["ticker", "seller", "calc_date", "short_ratio", "short_shares"], keep="last"
        )

    return JpxLoadResult(
        events=events,
        files_found=len(urls),
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


def build_short_metrics(events: pd.DataFrame, window_days: int = 75) -> pd.DataFrame:
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
    ev = ev.dropna(subset=["calc_date"])
    if ev.empty:
        return pd.DataFrame(columns=columns)

    newest = ev["calc_date"].max()
    cutoff = newest - pd.Timedelta(days=window_days)
    ev = ev[ev["calc_date"] >= cutoff].copy()

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
        age = max(0, (pd.Timestamp(datetime.now().date()) - pd.Timestamp(last_date).normalize()).days)
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


def build_price_feature_table(tickers: list[str]) -> pd.DataFrame:
    tickers = [normalize_ticker(t) for t in tickers if normalize_ticker(t)]
    frames = _download_prices(sorted(set(tickers)))
    rows = []
    for ticker in tickers:
        feat = _price_features(frames.get(ticker))
        feat["ticker"] = ticker
        rows.append(feat)
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
            "cover_score": early,
            "long_demand_score": long_demand,
            "confidence": int(min(100, confidence)),
            "phase": phase,
        })
        rows.append(item)

    out = pd.DataFrame(rows)
    return out.sort_values(["cover_score", "confidence", "short_pressure"], ascending=False).reset_index(drop=True)


def candidate_tickers(short_metrics: pd.DataFrame, limit: int = 60) -> list[str]:
    if short_metrics is None or short_metrics.empty:
        return []
    s = short_metrics.copy()
    # Prefer evidence of either pressure or recent change/breadth; avoid sending hundreds to yfinance.
    activity = s["delta_short"].abs().fillna(0) * 20 + s["short_pressure_base"].fillna(0)
    s = s.assign(_activity=activity).sort_values(["_activity", "short_ratio"], ascending=False)
    return s["ticker"].dropna().astype(str).head(limit).tolist()

from __future__ import annotations

import argparse
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd

from jquants_mex_adapter import JQuantsV2Client


OUT_DIR = Path("data/multiple_expansion")
DEFAULT_PORTFOLIO = "portfolio_data.csv"
BARS_CACHE = OUT_DIR / "me_all_market_bars_cache.csv.gz"



def _load_bar_cache(path: Path = BARS_CACHE) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        cache = pd.read_csv(path, dtype={"Code": str, "code": str})
    except Exception as exc:
        print(
            f"[ME-PREFILTER] cache load failed; rebuilding: "
            f"{type(exc).__name__}: {exc}"
        )
        return pd.DataFrame()

    date_col = "Date" if "Date" in cache.columns else (
        "date" if "date" in cache.columns else None
    )
    if date_col is None:
        return pd.DataFrame()

    cache["_cache_date"] = pd.to_datetime(
        cache[date_col], errors="coerce"
    ).dt.normalize()
    cache = cache.dropna(subset=["_cache_date"]).copy()
    return cache


def _merge_bar_cache(
    cache: pd.DataFrame,
    fresh: pd.DataFrame,
    *,
    to_date: date,
    keep_calendar_days: int = 230,
) -> pd.DataFrame:
    frames = [x for x in [cache, fresh] if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()

    merged = pd.concat(frames, ignore_index=True, sort=False)
    date_col = "Date" if "Date" in merged.columns else (
        "date" if "date" in merged.columns else None
    )
    code_col = "Code" if "Code" in merged.columns else (
        "code" if "code" in merged.columns else None
    )
    if date_col is None or code_col is None:
        return merged

    merged["_cache_date"] = pd.to_datetime(
        merged[date_col], errors="coerce"
    ).dt.normalize()
    merged["_cache_code"] = merged[code_col].map(_code4)
    merged = merged.dropna(subset=["_cache_date"]).copy()
    merged = merged[merged["_cache_code"] != ""].copy()

    floor = pd.Timestamp(to_date) - pd.Timedelta(
        days=max(190, int(keep_calendar_days))
    )
    merged = merged[merged["_cache_date"] >= floor.normalize()].copy()
    merged = merged.sort_values(
        ["_cache_date", "_cache_code"]
    ).drop_duplicates(
        subset=["_cache_date", "_cache_code"],
        keep="last",
    )
    return merged.reset_index(drop=True)


def _next_fetch_start(
    cache: pd.DataFrame,
    *,
    requested_from: date,
    to_date: date,
) -> date:
    if cache is None or cache.empty or "_cache_date" not in cache.columns:
        return requested_from

    latest = pd.to_datetime(
        cache["_cache_date"], errors="coerce"
    ).dropna()
    if latest.empty:
        return requested_from

    next_day = (latest.max() + pd.Timedelta(days=1)).date()
    return min(max(requested_from, next_day), to_date)


def _save_bar_cache(
    frame: pd.DataFrame,
    path: Path = BARS_CACHE,
) -> None:
    if frame is None or frame.empty:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    out = frame.drop(
        columns=["_cache_date", "_cache_code"],
        errors="ignore",
    )
    out.to_csv(
        path,
        index=False,
        compression="gzip",
    )


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _code4(value: Any) -> str:
    text = str(value or "").strip()
    if text.endswith(".T"):
        text = text[:-2]
    if len(text) == 5 and text.endswith("0"):
        text = text[:4]
    return text.zfill(4) if text.isdigit() and len(text) < 4 else text


def _first_existing(df: pd.DataFrame, names: list[str]) -> pd.Series:
    for name in names:
        if name in df.columns:
            return df[name]
    return pd.Series(pd.NA, index=df.index)


def _portfolio_codes(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return set()
    for col in ["ticker", "code", "銘柄コード"]:
        if col in df.columns:
            return {
                _code4(v)
                for v in df[col].dropna().astype(str)
                if _code4(v)
            }
    return set()


def _normalize_master(raw: pd.DataFrame) -> pd.DataFrame:
    if raw is None or raw.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=raw.index)
    out["ticker"] = _first_existing(
        raw,
        ["Code", "code", "LocalCode", "SecurityCode"],
    ).map(_code4)
    out["company_name"] = _first_existing(
        raw,
        ["CoName", "CompanyName", "Name", "IssueName"],
    ).astype("string")
    out["company_name_en"] = _first_existing(
        raw,
        ["CoNameEn", "CompanyNameEnglish", "NameEnglish"],
    ).astype("string")
    out["market_name"] = _first_existing(
        raw,
        ["MktNm", "MarketName", "MarketSegmentName"],
    ).astype("string")
    out["market_code"] = _first_existing(
        raw,
        ["MktCd", "MarketCode", "MarketSegmentCode"],
    ).astype("string")
    out["sector33_name"] = _first_existing(
        raw,
        ["S33Nm", "Sect33Nm", "Sector33Name"],
    ).astype("string")
    out["sector17_name"] = _first_existing(
        raw,
        ["S17Nm", "Sect17Nm", "Sector17Name"],
    ).astype("string")

    out = out[out["ticker"].str.fullmatch(r"\d{4}", na=False)].copy()

    market_text = (
        out["market_name"].fillna("")
        + " "
        + out["market_code"].fillna("")
    ).str.lower()

    common_market = (
        market_text.str.contains("prime")
        | market_text.str.contains("プライム")
        | market_text.str.contains("standard")
        | market_text.str.contains("スタンダード")
        | market_text.str.contains("growth")
        | market_text.str.contains("グロース")
    )

    # Some plans expose codes but not English/Japanese labels. In that case,
    # do not throw away the full universe merely because the label is blank.
    if common_market.any():
        out = out[common_market].copy()

    return out.drop_duplicates("ticker", keep="last")



def _fetch_all_market_bars_by_date(
    client: JQuantsV2Client,
    *,
    from_date: date,
    to_date: date,
    request_delay_seconds: float = 4.0,
    max_retries: int = 3,
) -> pd.DataFrame:
    """Fetch all-market bars one date at a time with rate-limit protection.

    J-Quants v2 rejects blank-code range requests. Date-only all-market
    requests work, but they can hit HTTP 429 when fired too quickly. We pace
    calls, retry 429s with exponential backoff, and fail clearly if coverage
    is too incomplete to build a reliable universe snapshot.
    """
    frames: list[pd.DataFrame] = []
    dates = pd.bdate_range(from_date, to_date)
    successful_dates: list[pd.Timestamp] = []

    for idx, ts in enumerate(dates, start=1):
        ymd = pd.Timestamp(ts).strftime("%Y%m%d")
        frame = pd.DataFrame()

        for attempt in range(max(1, int(max_retries)) + 1):
            try:
                frame = client.daily_bars(
                    code="",
                    date=ymd,
                )
                break
            except Exception as exc:
                status = getattr(
                    getattr(exc, "response", None),
                    "status_code",
                    None,
                )
                if status == 429 and attempt < max_retries:
                    wait_seconds = min(30.0, 3.0 * (2 ** attempt))
                    print(
                        f"[ME-PREFILTER] rate limited on {ymd}; "
                        f"retry {attempt + 1}/{max_retries} "
                        f"after {wait_seconds:.0f}s"
                    )
                    time.sleep(wait_seconds)
                    continue

                print(
                    f"[ME-PREFILTER] bars {ymd} unavailable: "
                    f"{type(exc).__name__}: {exc}"
                )
                frame = pd.DataFrame()
                break

        if frame is not None and not frame.empty:
            frames.append(frame)
            successful_dates.append(pd.Timestamp(ts).normalize())

        if idx == 1 or idx % 20 == 0 or idx == len(dates):
            print(
                f"[ME-PREFILTER] all-market bars progress "
                f"{idx}/{len(dates)} dates; "
                f"success={len(successful_dates)}"
            )

        if request_delay_seconds > 0 and idx < len(dates):
            time.sleep(float(request_delay_seconds))

    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)

    expected = max(1, len(dates))
    coverage = len(successful_dates) / expected
    latest_success = max(successful_dates) if successful_dates else None
    staleness_days = (
        (pd.Timestamp(to_date) - latest_success).days
        if latest_success is not None
        else 9999
    )

    print(
        f"[ME-PREFILTER] all-market coverage "
        f"{len(successful_dates)}/{expected} ({coverage:.1%}); "
        f"latest={latest_success.date() if latest_success is not None else 'n/a'}"
    )

    if coverage < 0.70 or staleness_days > 7:
        raise RuntimeError(
            "All-market bars coverage is too incomplete for a reliable "
            "prefilter. This is usually caused by J-Quants rate limiting. "
            f"coverage={coverage:.1%}, staleness_days={staleness_days}"
        )

    return out

def _normalize_market(
    bars: pd.DataFrame,
    valuation: pd.DataFrame,
) -> pd.DataFrame:
    if bars is None or bars.empty:
        return pd.DataFrame()

    out = pd.DataFrame(index=bars.index)
    out["trade_date"] = pd.to_datetime(
        _first_existing(bars, ["Date", "date"]),
        errors="coerce",
    )
    out["ticker"] = _first_existing(
        bars,
        ["Code", "code"],
    ).map(_code4)
    out["close"] = pd.to_numeric(
        _first_existing(bars, ["AdjC", "C", "Close"]),
        errors="coerce",
    )
    out["volume"] = pd.to_numeric(
        _first_existing(bars, ["AdjVo", "Vo", "Volume"]),
        errors="coerce",
    )
    out = out.dropna(subset=["trade_date", "ticker", "close"])
    out = out[out["ticker"].str.fullmatch(r"\d{4}", na=False)].copy()

    if valuation is not None and not valuation.empty:
        val = pd.DataFrame(index=valuation.index)
        val["trade_date"] = pd.to_datetime(
            _first_existing(valuation, ["Date", "date"]),
            errors="coerce",
        )
        val["ticker"] = _first_existing(
            valuation,
            ["Code", "code"],
        ).map(_code4)
        val["per"] = pd.to_numeric(
            _first_existing(valuation, ["PER", "Per"]),
            errors="coerce",
        )
        val["pbr"] = pd.to_numeric(
            _first_existing(valuation, ["PBR", "Pbr"]),
            errors="coerce",
        )
        val["market_cap"] = pd.to_numeric(
            _first_existing(valuation, ["MktCap", "MarketCap"]),
            errors="coerce",
        )
        val = val.dropna(subset=["trade_date", "ticker"])
        val = val.drop_duplicates(["trade_date", "ticker"], keep="last")
        out = out.merge(
            val,
            on=["trade_date", "ticker"],
            how="left",
        )

    return out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def _safe_percentile(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    if s.notna().sum() <= 1:
        return pd.Series(0.5, index=series.index)
    return s.rank(pct=True, method="average").fillna(0.0)


def _coarse_multiple_features(group: pd.DataFrame) -> pd.DataFrame:
    g = group.copy()
    g = g.sort_values("trade_date").reset_index(drop=True)

    per = pd.to_numeric(
        g["per"] if "per" in g.columns else pd.Series(pd.NA, index=g.index),
        errors="coerce",
    )
    pbr = pd.to_numeric(
        g["pbr"] if "pbr" in g.columns else pd.Series(pd.NA, index=g.index),
        errors="coerce",
    )

    per_valid = per.where(per > 0)
    pbr_valid = pbr.where(pbr > 0)

    per_log = per_valid.map(
        lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan")
    )
    pbr_log = pbr_valid.map(
        lambda x: math.log(x) if pd.notna(x) and x > 0 else float("nan")
    )

    per_base = per_log.rolling(100, min_periods=40).median()
    pbr_base = pbr_log.rolling(100, min_periods=40).median()

    per_mlp = ((per_log - per_base) * 0.55).map(
        lambda x: math.exp(x) if pd.notna(x) else float("nan")
    )
    pbr_mlp = ((pbr_log - pbr_base) * 0.55).map(
        lambda x: math.exp(x) if pd.notna(x) else float("nan")
    )

    # P/E is closer to the final AUTO model; P/B is only a coarse fallback.
    mlp = per_mlp.copy()
    source = pd.Series("P/E", index=g.index, dtype="string")
    fallback = mlp.isna() & pbr_mlp.notna()
    mlp.loc[fallback] = pbr_mlp.loc[fallback]
    source.loc[fallback] = "P/B"
    source.loc[mlp.isna()] = "PRICE ONLY"

    vel20 = (mlp / mlp.shift(20) - 1.0) * 100.0
    accel10 = vel20 - vel20.shift(10)
    slope10 = (mlp / mlp.shift(10) - 1.0) * 100.0
    curvature10 = slope10 - slope10.shift(10)

    g["coarse_mlp"] = mlp
    g["coarse_multiple_source"] = source
    g["coarse_velocity_pct"] = vel20
    g["coarse_acceleration_pct"] = accel10
    g["coarse_curvature_pct"] = curvature10
    return g


def _build_universe_snapshot(
    market: pd.DataFrame,
    master: pd.DataFrame,
    *,
    min_turnover_yen: float,
) -> pd.DataFrame:
    if market.empty:
        return pd.DataFrame()

    featured = (
        market.groupby("ticker", group_keys=False)
        .apply(_coarse_multiple_features)
        .reset_index(drop=True)
    )

    rows = []
    for ticker, g in featured.groupby("ticker", sort=False):
        g = g.sort_values("trade_date").reset_index(drop=True)
        if g.empty:
            continue

        latest = g.iloc[-1]
        close = pd.to_numeric(g["close"], errors="coerce")
        volume = pd.to_numeric(g["volume"], errors="coerce")
        turnover = close * volume

        last20 = turnover.tail(20).dropna()
        last60 = close.tail(60).dropna()

        avg_turnover20 = float(last20.mean()) if len(last20) >= 10 else None
        trading_days = int(close.notna().sum())

        ret20 = None
        ret60 = None
        high120_dist = None
        volume_ratio = None
        realized_vol20_pct = None

        if len(close.dropna()) >= 21:
            p0 = _num(close.iloc[-21])
            p1 = _num(close.iloc[-1])
            if p0 and p1:
                ret20 = p1 / p0 - 1.0

        if len(close.dropna()) >= 61:
            p0 = _num(close.iloc[-61])
            p1 = _num(close.iloc[-1])
            if p0 and p1:
                ret60 = p1 / p0 - 1.0

        daily_ret = close.pct_change().tail(20).dropna()
        if len(daily_ret) >= 10:
            realized_vol20_pct = float(daily_ret.std(ddof=0) * math.sqrt(252.0) * 100.0)

        last120 = close.tail(120).dropna()
        if not last120.empty:
            hi = _num(last120.max())
            px = _num(close.iloc[-1])
            if hi and px:
                high120_dist = px / hi - 1.0

        vol_ma20 = volume.shift(1).tail(20).mean()
        last_vol = _num(volume.iloc[-1])
        if last_vol is not None and pd.notna(vol_ma20) and vol_ma20 > 0:
            volume_ratio = last_vol / float(vol_ma20)

        rows.append(
            {
                "ticker": str(ticker),
                "trade_date": latest.get("trade_date"),
                "close": _num(latest.get("close")),
                "avg_turnover20": avg_turnover20,
                "trading_days": trading_days,
                "ret20": ret20,
                "ret60": ret60,
                "distance_120d_high": high120_dist,
                "volume_ratio20": volume_ratio,
                "realized_vol20_pct": realized_vol20_pct,
                "market_cap": _num(latest.get("market_cap")),
                "per": _num(latest.get("per")),
                "pbr": _num(latest.get("pbr")),
                "coarse_multiple_source": str(
                    latest.get("coarse_multiple_source") or "PRICE ONLY"
                ),
                "coarse_mlp": _num(latest.get("coarse_mlp")),
                "coarse_velocity_pct": _num(
                    latest.get("coarse_velocity_pct")
                ),
                "coarse_acceleration_pct": _num(
                    latest.get("coarse_acceleration_pct")
                ),
                "coarse_curvature_pct": _num(
                    latest.get("coarse_curvature_pct")
                ),
            }
        )

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.merge(master, on="ticker", how="left")

    out["liquidity_ok"] = (
        pd.to_numeric(out["avg_turnover20"], errors="coerce")
        >= float(min_turnover_yen)
    )
    out["history_ok"] = out["trading_days"] >= 60

    # Soft scores: the heavy MEX/FCF engine makes the final decision.
    liq_pct = _safe_percentile(
        pd.to_numeric(out["avg_turnover20"], errors="coerce")
    )
    out["liquidity_score"] = liq_pct * 100.0

    accel = pd.to_numeric(
        out["coarse_acceleration_pct"], errors="coerce"
    )
    curve = pd.to_numeric(
        out["coarse_curvature_pct"], errors="coerce"
    )
    vel = pd.to_numeric(
        out["coarse_velocity_pct"], errors="coerce"
    )
    mlp = pd.to_numeric(out["coarse_mlp"], errors="coerce")

    shape = pd.Series(35.0, index=out.index)
    shape.loc[mlp.notna()] = 45.0

    watch = (
        (mlp >= 1.03)
        & (vel <= 0)
        & (accel > 0)
        & (curve > 0)
    )
    ready = (
        (mlp >= 1.07)
        & (vel >= -3.0)
        & (vel <= 0)
        & (accel >= 0.10)
        & (curve >= 0.10)
    )
    expanding = (
        (mlp >= 0.98)
        & (vel > 0)
        & (accel > 0)
    )

    shape.loc[watch] = 72.0
    shape.loc[ready] = 92.0
    shape.loc[expanding] = 76.0
    out["coarse_shape_score"] = shape

    ret20 = pd.to_numeric(out["ret20"], errors="coerce")
    high_dist = pd.to_numeric(out["distance_120d_high"], errors="coerce")
    volume_ratio = pd.to_numeric(out["volume_ratio20"], errors="coerce")

    price_score = ((ret20.clip(-0.15, 0.20) + 0.15) / 0.35 * 100.0).fillna(45.0)
    high_score = ((high_dist.clip(-0.40, 0.0) + 0.40) / 0.40 * 100.0).fillna(45.0)
    volume_score = ((volume_ratio.clip(0.5, 2.0) - 0.5) / 1.5 * 100.0).fillna(45.0)

    out["coarse_score"] = (
        out["coarse_shape_score"] * 0.45
        + out["liquidity_score"] * 0.20
        + price_score * 0.12
        + high_score * 0.13
        + volume_score * 0.10
    ).clip(0.0, 100.0)

    out["prefilter_state"] = "BASE"
    out.loc[watch, "prefilter_state"] = "RE-ACCEL WATCH"
    out.loc[ready, "prefilter_state"] = "RE-READY PROXY"
    out.loc[expanding, "prefilter_state"] = "EXPANDING PROXY"

    out["eligible"] = out["liquidity_ok"] & out["history_ok"]
    return out.sort_values(
        ["eligible", "coarse_score", "avg_turnover20", "ticker"],
        ascending=[False, False, False, True],
        na_position="last",
    ).reset_index(drop=True)


def _select_candidates(
    universe: pd.DataFrame,
    *,
    max_candidates: int,
    pinned_codes: set[str],
) -> pd.DataFrame:
    if universe.empty:
        return universe.copy()

    eligible = universe[universe["eligible"]].copy()
    eligible = eligible.sort_values(
        ["coarse_score", "avg_turnover20", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    )

    chosen: list[str] = []

    def add_codes(codes):
        for code in codes:
            code = _code4(code)
            if code and code not in chosen:
                chosen.append(code)

    # 1) Always retain portfolio/manual names when they are in today's master.
    add_codes(
        universe.loc[
            universe["ticker"].isin(pinned_codes),
            "ticker",
        ].tolist()
    )

    # 2) Reserve a sleeve for Growth so liquid emerging names are not crowded
    #    out by Prime mega-caps.
    growth_mask = (
        eligible.get("market_name", pd.Series("", index=eligible.index))
        .fillna("")
        .astype(str)
        .str.contains("growth|グロース", case=False, regex=True)
    )
    growth_quota = max(5, int(max_candidates * 0.20))
    add_codes(eligible.loc[growth_mask, "ticker"].head(growth_quota).tolist())

    # 3) Fill the rest by coarse MEX score.
    add_codes(eligible["ticker"].tolist())

    chosen = chosen[:max_candidates]
    out = universe[universe["ticker"].isin(chosen)].copy()
    rank_map = {code: i + 1 for i, code in enumerate(chosen)}
    out["prefilter_rank"] = out["ticker"].map(rank_map)
    out["pinned"] = out["ticker"].isin(pinned_codes)
    return out.sort_values("prefilter_rank").reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build an all-TSE lightweight Multiple Expansion prefilter. "
            "The output is the candidate universe for the heavy PIT/FCF MEX pipeline."
        )
    )
    parser.add_argument(
        "--lookback-days",
        type=int,
        default=110,
        help=(
            "Calendar-day lookback for the lightweight all-market bootstrap. "
            "110 days is enough for the 60-session history gate while keeping "
            "the first J-Quants cache build practical."
        ),
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=60,
        help="Maximum names passed to the heavy MEX/FCF stage.",
    )
    parser.add_argument(
        "--min-turnover-yen",
        type=float,
        default=50_000_000,
        help="Minimum 20-session average traded value in yen.",
    )
    parser.add_argument(
        "--portfolio",
        default=DEFAULT_PORTFOLIO,
        help="Portfolio CSV. Its names are pinned into the candidate set.",
    )
    parser.add_argument(
        "--to-date",
        default=date.today().isoformat(),
        help="End date YYYY-MM-DD.",
    )
    args = parser.parse_args()

    to_date = pd.Timestamp(args.to_date).date()
    from_date = to_date - timedelta(days=max(90, args.lookback_days))

    client = JQuantsV2Client.from_env()

    print("[ME-PREFILTER] fetching listed issue master ...")
    master_raw = pd.DataFrame.from_records(
        client._get_paginated("/equities/master", {})
    )
    master = _normalize_master(master_raw)
    if master.empty:
        raise SystemExit("Listed issue master returned no usable TSE codes.")

    cache = _load_bar_cache()
    fetch_from = _next_fetch_start(
        cache,
        requested_from=from_date,
        to_date=to_date,
    )

    if cache.empty:
        print(
            f"[ME-PREFILTER] first-run bootstrap: all-market bars "
            f"{from_date.isoformat()}..{to_date.isoformat()} "
            f"with rate-limit-safe pacing"
        )
    else:
        cache_latest = pd.to_datetime(
            cache["_cache_date"], errors="coerce"
        ).max()
        print(
            f"[ME-PREFILTER] cache loaded: {len(cache):,} rows; "
            f"latest={cache_latest.date() if pd.notna(cache_latest) else 'n/a'}"
        )

    fresh = pd.DataFrame()
    if fetch_from <= to_date:
        print(
            f"[ME-PREFILTER] fetching incremental all-market bars "
            f"{fetch_from.isoformat()}..{to_date.isoformat()} ..."
        )
        fresh = _fetch_all_market_bars_by_date(
            client,
            from_date=fetch_from,
            to_date=to_date,
        )
    else:
        print("[ME-PREFILTER] cache is already current; no bar fetch needed.")

    bars = _merge_bar_cache(
        cache,
        fresh,
        to_date=to_date,
    )
    if bars.empty:
        raise SystemExit("All-market daily bars returned no rows.")

    _save_bar_cache(bars)
    print(
        f"[ME-PREFILTER] cache saved: {len(bars):,} rows "
        f"to {BARS_CACHE}"
    )

    # The all-market range form is not accepted by J-Quants v2 when code is
    # blank. The heavy PIT/FCF stage performs the real valuation work, so the
    # lightweight prefilter intentionally remains price/liquidity-led here.
    valuation = pd.DataFrame()
    print(
        "[ME-PREFILTER] skipping all-market valuation in lightweight stage; "
        "heavy MEX stage will compute exact valuation for shortlisted names."
    )

    market = _normalize_market(bars, valuation)
    universe = _build_universe_snapshot(
        market,
        master,
        min_turnover_yen=args.min_turnover_yen,
    )
    if universe.empty:
        raise SystemExit("No universe snapshot was produced.")

    pinned_codes = _portfolio_codes(args.portfolio)
    candidates = _select_candidates(
        universe,
        max_candidates=max(1, args.max_candidates),
        pinned_codes=pinned_codes,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    universe_path = OUT_DIR / "me_universe_snapshot.csv"
    candidate_path = OUT_DIR / "me_universe_candidates.csv"
    universe.to_csv(universe_path, index=False)
    candidates.to_csv(candidate_path, index=False)

    print(
        f"[ME-PREFILTER] master={len(master)} "
        f"snapshot={len(universe)} eligible={int(universe['eligible'].sum())} "
        f"heavy_candidates={len(candidates)}"
    )
    print(f"[ME-PREFILTER] wrote {universe_path}")
    print(f"[ME-PREFILTER] wrote {candidate_path}")

    show = [
        c
        for c in [
            "prefilter_rank",
            "ticker",
            "company_name",
            "market_name",
            "prefilter_state",
            "coarse_score",
            "coarse_mlp",
            "coarse_velocity_pct",
            "coarse_acceleration_pct",
            "avg_turnover20",
            "pinned",
        ]
        if c in candidates.columns
    ]
    print(candidates[show].head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

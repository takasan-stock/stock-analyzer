from __future__ import annotations

import math
from typing import Any

import pandas as pd


MULTIPLE_WEIGHTS = {
    "p_fcf": 0.50,
    "ev_ebitda": 0.30,
    "per": 0.20,
}

FCF_ENGINE_WEIGHTS = {
    "fcf_per_share_3y_cagr_score": 25.0,
    "fcf_per_share_yoy_score": 10.0,
    "cfo_yoy_score": 10.0,
    "roic_score": 20.0,
    "fcf_margin_score": 10.0,
    "operating_margin_score": 10.0,
    "cash_conversion_score": 5.0,
    "net_cash_ratio_score": 5.0,
    "dilution_score": 5.0,
}

MEX_WEIGHTS = {
    "fcf_engine_score": 30.0,
    "historical_discount_score": 15.0,
    "velocity_score": 15.0,
    "acceleration_score": 15.0,
    "normalization_gap_score": 10.0,
    "rs_score": 5.0,
    "volume_score": 5.0,
    "fcf_quality_score": 5.0,
}

QUALITY_SCORES = {"A": 100.0, "B": 75.0, "C": 40.0, "D": 10.0}


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        value = float(value)
        if not math.isfinite(value):
            return None
        return value
    except (TypeError, ValueError):
        return None


def _piecewise_score(value: Any, anchors: list[tuple[float, float]]) -> float | None:
    x = _num(value)
    if x is None:
        return None
    anchors = sorted(anchors)
    if x <= anchors[0][0]:
        return float(anchors[0][1])
    if x >= anchors[-1][0]:
        return float(anchors[-1][1])
    for (x0, y0), (x1, y1) in zip(anchors[:-1], anchors[1:]):
        if x0 <= x <= x1:
            if x1 == x0:
                return float(y1)
            ratio = (x - x0) / (x1 - x0)
            return float(y0 + ratio * (y1 - y0))
    return None


def _weighted_score(
    row: pd.Series,
    weights: dict[str, float],
    min_coverage: float,
) -> tuple[float | None, float]:
    total_weight = sum(weights.values())
    used_weight = 0.0
    total = 0.0
    for col, weight in weights.items():
        value = _num(row.get(col))
        if value is None:
            continue
        used_weight += weight
        total += value * weight
    coverage = used_weight / total_weight if total_weight else 0.0
    if coverage < min_coverage or used_weight <= 0:
        return None, coverage
    return total / used_weight, coverage


def build_point_in_time_financials(
    financial_df: pd.DataFrame,
    trading_dates: pd.DataFrame,
) -> pd.DataFrame:
    """Align each trading row to the latest financial record actually available.

    financial_df must contain ticker and available_date.
    trading_dates must contain ticker and trade_date.

    Financial records are never carried backward before available_date.
    """
    required_fin = {"ticker", "available_date"}
    required_market = {"ticker", "trade_date"}
    if not required_fin.issubset(financial_df.columns):
        raise ValueError("financial_df requires ticker and available_date")
    if not required_market.issubset(trading_dates.columns):
        raise ValueError("trading_dates requires ticker and trade_date")

    market = trading_dates.copy()
    fin = financial_df.copy()
    market["trade_date"] = pd.to_datetime(market["trade_date"], errors="coerce")
    fin["available_date"] = pd.to_datetime(fin["available_date"], errors="coerce")
    market = market.dropna(subset=["ticker", "trade_date"])
    fin = fin.dropna(subset=["ticker", "available_date"])

    results = []
    financial_payload_cols = [c for c in fin.columns if c != "ticker"]

    for ticker, mg in market.groupby("ticker", sort=False):
        fg = fin[fin["ticker"] == ticker].sort_values("available_date").copy()
        mg = mg.sort_values("trade_date").copy()

        if fg.empty:
            for col in financial_payload_cols:
                if col not in mg.columns:
                    mg[col] = pd.NA
            results.append(mg)
            continue

        fg = fg.drop_duplicates(subset=["available_date"], keep="last")
        fg = fg.drop(columns=["ticker"])
        aligned = pd.merge_asof(
            mg,
            fg,
            left_on="trade_date",
            right_on="available_date",
            direction="backward",
            allow_exact_matches=True,
        )
        results.append(aligned)

    if not results:
        return market.iloc[0:0].copy()

    out = pd.concat(results, ignore_index=True)
    return out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)


def calculate_valuation_multiples(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate valuation multiples from already point-in-time aligned inputs."""
    out = df.copy()

    market_cap = pd.to_numeric(out.get("market_cap_pti"), errors="coerce")
    fcf = pd.to_numeric(out.get("fcf_ttm"), errors="coerce")
    ebitda = pd.to_numeric(out.get("ebitda_ttm"), errors="coerce")
    net_income = pd.to_numeric(out.get("net_income_ttm"), errors="coerce")
    debt = pd.to_numeric(out.get("total_debt", 0.0), errors="coerce").fillna(0.0)
    cash = pd.to_numeric(out.get("cash_and_equivalents", 0.0), errors="coerce").fillna(0.0)

    out["p_fcf"] = (market_cap / fcf).where((market_cap > 0) & (fcf > 0))
    enterprise_value = market_cap + debt - cash
    out["ev_ebitda"] = (enterprise_value / ebitda).where(
        (enterprise_value > 0) & (ebitda > 0)
    )
    out["per"] = (market_cap / net_income).where(
        (market_cap > 0) & (net_income > 0)
    )

    out["valid_multiple_count"] = out[list(MULTIPLE_WEIGHTS)].notna().sum(axis=1)
    return out


def _rolling_mad(
    series: pd.Series,
    window: int,
    min_periods: int,
) -> pd.Series:
    def _mad(arr) -> float:
        s = pd.Series(arr).dropna()
        if len(s) == 0:
            return float("nan")
        med = float(s.median())
        return float((s - med).abs().median())

    return series.rolling(
        window=window,
        min_periods=min_periods,
    ).apply(_mad, raw=False)


def calculate_multiple_normalization(
    df: pd.DataFrame,
    *,
    lookback_days: int = 1260,
    min_history_days: int = 504,
    core_ema_span: int = 40,
) -> pd.DataFrame:
    """Build MLP_C, robust Z, velocity and acceleration without look-ahead."""
    if df.empty:
        return df.copy()

    out = df.copy()
    if "ticker" not in out.columns or "trade_date" not in out.columns:
        raise ValueError("ticker and trade_date are required")

    out["trade_date"] = pd.to_datetime(out["trade_date"], errors="coerce")
    out = out.sort_values(["ticker", "trade_date"]).reset_index(drop=True)

    def process_group(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        d_cols = []
        z_cols = []

        for col in MULTIPLE_WEIGHTS:
            values = pd.to_numeric(g.get(col), errors="coerce")
            log_values = values.where(values > 0).map(
                lambda x: math.log(x) if pd.notna(x) else float("nan")
            )

            # Reference distribution uses only information through t-1.
            hist = log_values.shift(1)
            med = hist.rolling(
                lookback_days,
                min_periods=min_history_days,
            ).median()
            mad = _rolling_mad(
                hist,
                lookback_days,
                min_history_days,
            )

            d_col = f"_{col}_d"
            z_col = f"_{col}_z"
            g[d_col] = log_values - med
            denom = 1.4826 * mad
            g[z_col] = ((log_values - med) / denom).where(denom > 1e-12)
            d_cols.append(d_col)
            z_cols.append(z_col)

        raw_values = []
        z_values = []
        counts = []

        for _, row in g.iterrows():
            d_sum = z_sum = weight_sum_d = weight_sum_z = 0.0
            valid_d = 0
            for col, weight in MULTIPLE_WEIGHTS.items():
                d = _num(row.get(f"_{col}_d"))
                z = _num(row.get(f"_{col}_z"))
                if d is not None:
                    d_sum += weight * d
                    weight_sum_d += weight
                    valid_d += 1
                if z is not None:
                    z_sum += weight * z
                    weight_sum_z += weight

            if valid_d >= 2 and weight_sum_d > 0:
                raw_values.append(math.exp(d_sum / weight_sum_d))
            else:
                raw_values.append(float("nan"))

            if weight_sum_z > 0:
                z_values.append(z_sum / weight_sum_z)
            else:
                z_values.append(float("nan"))
            counts.append(valid_d)

        g["mlp_c_raw"] = raw_values
        g["mlp_z"] = z_values
        g["valid_multiple_count"] = counts

        g["mlp_c_core"] = pd.to_numeric(
            g["mlp_c_raw"],
            errors="coerce",
        ).ewm(
            span=core_ema_span,
            adjust=False,
            min_periods=1,
        ).mean()

        core = pd.to_numeric(g["mlp_c_core"], errors="coerce")
        g["mlp_velocity_20"] = (core / core.shift(20)).map(
            lambda x: math.log(x)
            if pd.notna(x) and x > 0
            else float("nan")
        )
        g["mlp_velocity_pct_20"] = g["mlp_velocity_20"].map(
            lambda x: math.exp(x) - 1.0
            if pd.notna(x)
            else float("nan")
        )
        g["mlp_acceleration"] = (
            g["mlp_velocity_20"]
            - g["mlp_velocity_20"].shift(20)
        )
        g["normalization_gap"] = (
            1.0 / core - 1.0
        ).where(core > 0)

        return g.drop(columns=d_cols + z_cols)

    return (
        out.groupby("ticker", group_keys=False)
        .apply(process_group)
        .reset_index(drop=True)
    )


def _series(
    out: pd.DataFrame,
    col: str,
    default: float | None = None,
) -> pd.Series:
    if col in out.columns:
        return pd.to_numeric(out[col], errors="coerce")
    fill = float("nan") if default is None else float(default)
    return pd.Series(fill, index=out.index, dtype=float)


def _derive_fundamental_metrics(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    if "fcf_per_share_ttm" not in out.columns:
        fcf = _series(out, "fcf_ttm")
        shares = _series(out, "diluted_shares_ttm")
        out["fcf_per_share_ttm"] = (fcf / shares).where(
            (shares > 0) & fcf.notna()
        )

    if "fcf_margin" not in out.columns:
        fcf = _series(out, "fcf_ttm")
        rev = _series(out, "revenue_ttm")
        out["fcf_margin"] = (fcf / rev).where(rev > 0)

    if "operating_margin" not in out.columns:
        op = _series(out, "operating_income_ttm")
        rev = _series(out, "revenue_ttm")
        out["operating_margin"] = (op / rev).where(rev > 0)

    if "cash_conversion" not in out.columns:
        fcf = _series(out, "fcf_ttm")
        op = _series(out, "operating_income_ttm")
        out["cash_conversion"] = (fcf / op).where(op > 0)

    if "net_cash_ratio" not in out.columns:
        cash = _series(out, "cash_and_equivalents")
        debt = _series(out, "total_debt")
        market_cap = _series(out, "market_cap_pti")
        out["net_cash_ratio"] = (
            (cash - debt) / market_cap
        ).where(market_cap > 0)

    if (
        "fcf_per_share_yoy" not in out.columns
        and {"fcf_per_share_ttm", "fcf_per_share_1y_ago"}.issubset(out.columns)
    ):
        now = pd.to_numeric(out["fcf_per_share_ttm"], errors="coerce")
        prev = pd.to_numeric(
            out["fcf_per_share_1y_ago"],
            errors="coerce",
        )
        out["fcf_per_share_yoy"] = (
            now / prev - 1.0
        ).where(prev > 0)

    if (
        "cfo_yoy" not in out.columns
        and {"cfo_ttm", "cfo_1y_ago"}.issubset(out.columns)
    ):
        now = pd.to_numeric(out["cfo_ttm"], errors="coerce")
        prev = pd.to_numeric(out["cfo_1y_ago"], errors="coerce")
        out["cfo_yoy"] = (now / prev - 1.0).where(prev > 0)

    return out


def calculate_fcf_engine(df: pd.DataFrame) -> pd.DataFrame:
    out = _derive_fundamental_metrics(df)

    out["fcf_per_share_3y_cagr_score"] = out.get(
        "fcf_per_share_3y_cagr",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (-0.10, 0),
                (0.00, 40),
                (0.10, 65),
                (0.20, 85),
                (0.30, 100),
            ],
        )
    )
    out["fcf_per_share_yoy_score"] = out.get(
        "fcf_per_share_yoy",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (-0.20, 0),
                (0.00, 50),
                (0.20, 80),
                (0.40, 100),
            ],
        )
    )
    out["cfo_yoy_score"] = out.get(
        "cfo_yoy",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (-0.20, 0),
                (0.00, 50),
                (0.20, 80),
                (0.40, 100),
            ],
        )
    )
    out["roic_score"] = out.get(
        "roic",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (0.00, 0),
                (0.05, 35),
                (0.10, 60),
                (0.15, 80),
                (0.20, 100),
            ],
        )
    )
    out["fcf_margin_score"] = out.get(
        "fcf_margin",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (0.00, 0),
                (0.05, 40),
                (0.10, 65),
                (0.20, 90),
                (0.25, 100),
            ],
        )
    )
    out["operating_margin_score"] = out.get(
        "operating_margin",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (0.00, 0),
                (0.05, 35),
                (0.10, 60),
                (0.20, 90),
                (0.25, 100),
            ],
        )
    )
    out["cash_conversion_score"] = out.get(
        "cash_conversion",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (0.20, 0),
                (0.50, 40),
                (0.80, 70),
                (1.00, 85),
                (1.20, 100),
            ],
        )
    )
    out["net_cash_ratio_score"] = out.get(
        "net_cash_ratio",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (-0.30, 0),
                (-0.10, 30),
                (0.00, 50),
                (0.10, 75),
                (0.25, 100),
            ],
        )
    )

    def dilution_score(v: Any) -> float | None:
        x = _num(v)
        if x is None:
            return None
        score = 70.0
        if x <= -0.03:
            score += 30.0
        elif x <= 0.00:
            score += 15.0
        elif x <= 0.02:
            score += 0.0
        elif x <= 0.05:
            score -= 30.0
        else:
            score -= 70.0
        return max(0.0, min(100.0, score))

    out["dilution_score"] = out.get(
        "share_count_1y_change",
        pd.Series(index=out.index, dtype=float),
    ).map(dilution_score)

    scores = out.apply(
        lambda row: _weighted_score(
            row,
            FCF_ENGINE_WEIGHTS,
            min_coverage=0.60,
        ),
        axis=1,
    )
    out["fcf_engine_score"] = [x[0] for x in scores]
    out["fcf_engine_coverage"] = [x[1] for x in scores]
    out["fcf_engine_status"] = out["fcf_engine_score"].map(
        lambda x: "OK" if pd.notna(x) else "INSUFFICIENT"
    )
    return out


def classify_fcf_quality(df: pd.DataFrame) -> pd.DataFrame:
    out = _derive_fundamental_metrics(df)

    def classify(row: pd.Series) -> str | None:
        fcf_yoy = _num(row.get("fcf_per_share_yoy"))
        cfo_yoy = _num(row.get("cfo_yoy"))
        revenue_growth = _num(row.get("revenue_yoy"))
        operating_growth = _num(row.get("operating_income_yoy"))
        dilution = _num(row.get("share_count_1y_change"))
        cfo = _num(row.get("cfo_ttm"))
        capex_yoy = _num(row.get("capex_yoy"))

        observed = sum(
            v is not None
            for v in [
                fcf_yoy,
                cfo_yoy,
                revenue_growth,
                operating_growth,
                dilution,
                cfo,
            ]
        )
        if observed < 3:
            return None

        if (
            fcf_yoy is not None
            and fcf_yoy < 0
        ) or (
            cfo is not None
            and cfo <= 0
        ):
            return "D"

        capex_cut_driven = (
            fcf_yoy is not None
            and fcf_yoy > 0
            and capex_yoy is not None
            and capex_yoy <= -0.30
            and cfo_yoy is not None
            and cfo_yoy <= 0
        )
        if capex_cut_driven:
            return "C"

        structural = (
            fcf_yoy is not None
            and fcf_yoy > 0
            and cfo_yoy is not None
            and cfo_yoy > 0
            and revenue_growth is not None
            and revenue_growth > 0
            and operating_growth is not None
            and operating_growth > 0
            and (dilution is None or dilution <= 0.02)
        )
        if structural:
            # Summary-only CFO+CFI is useful as a conservative FCF proxy,
            # but it is not identical to CFO-CAPEX. Never grade proxy-based
            # cash flow as top-quality A.
            if str(row.get("fcf_basis") or "") == "CFO_PLUS_CFI_PROXY":
                return "B"
            return "A"

        if (
            fcf_yoy is not None
            and fcf_yoy >= 0
            and (cfo is None or cfo > 0)
        ):
            return "B"
        return "C"

    out["fcf_quality_grade"] = out.apply(classify, axis=1)
    out["fcf_quality_score"] = out["fcf_quality_grade"].map(
        QUALITY_SCORES
    )
    return out


def calculate_market_confirmation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if out.empty:
        return out

    if {"ticker", "trade_date"}.issubset(out.columns):
        out["trade_date"] = pd.to_datetime(
            out["trade_date"],
            errors="coerce",
        )
        out = out.sort_values(
            ["ticker", "trade_date"]
        ).reset_index(drop=True)
        if (
            "rs_rating_20d_ago" not in out.columns
            and "rs_rating" in out.columns
        ):
            out["rs_rating_20d_ago"] = (
                out.groupby("ticker")["rs_rating"].shift(20)
            )
        if (
            "ema20_5d_ago" not in out.columns
            and "ema20" in out.columns
        ):
            out["ema20_5d_ago"] = (
                out.groupby("ticker")["ema20"].shift(5)
            )

    rs = pd.to_numeric(out.get("rs_rating"), errors="coerce")
    rs_prev = pd.to_numeric(
        out.get("rs_rating_20d_ago"),
        errors="coerce",
    )
    out["confirm_rs"] = (rs >= 65) & (rs > rs_prev)

    volume = pd.to_numeric(out.get("volume"), errors="coerce")
    volume_ma20 = pd.to_numeric(
        out.get("volume_ma20"),
        errors="coerce",
    )
    out["volume_ratio"] = (
        volume / volume_ma20
    ).where(volume_ma20 > 0)
    out["confirm_volume"] = out["volume_ratio"] >= 1.20

    close = pd.to_numeric(out.get("adj_close"), errors="coerce")
    ema20 = pd.to_numeric(out.get("ema20"), errors="coerce")
    ema20_prev = pd.to_numeric(
        out.get("ema20_5d_ago"),
        errors="coerce",
    )
    out["confirm_ema20"] = (
        (close > ema20) & (ema20 > ema20_prev)
    )

    high52 = pd.to_numeric(out.get("high_52w"), errors="coerce")
    out["confirm_52w"] = (
        (close / high52).where(high52 > 0) >= 0.85
    )

    out["market_confirm_count"] = out[
        [
            "confirm_rs",
            "confirm_volume",
            "confirm_ema20",
            "confirm_52w",
        ]
    ].fillna(False).astype(int).sum(axis=1)
    return out


def _historical_discount_score(
    mlp: Any,
    z: Any,
) -> float | None:
    level_score = _piecewise_score(
        mlp,
        [
            (0.80, 100),
            (0.90, 85),
            (1.00, 55),
            (1.10, 20),
            (1.20, 0),
        ],
    )
    z_score = _piecewise_score(
        z,
        [
            (-1.50, 100),
            (-0.50, 80),
            (0.00, 55),
            (1.00, 20),
            (2.00, 0),
        ],
    )
    vals = [v for v in [level_score, z_score] if v is not None]
    return sum(vals) / len(vals) if vals else None


def _rs_score(rs: Any, rs_prev: Any) -> float | None:
    value = _num(rs)
    prev = _num(rs_prev)
    if value is None:
        return None
    rising = prev is not None and value > prev
    if value < 40:
        return 0.0
    if value < 65:
        return 40.0
    if value < 75:
        return 70.0 if rising else 55.0
    if value < 90:
        return 90.0 if rising else 70.0
    return 100.0 if rising else 80.0


def _volume_score(ratio: Any) -> float | None:
    return _piecewise_score(
        ratio,
        [
            (0.80, 20),
            (1.00, 40),
            (1.20, 60),
            (1.50, 80),
            (2.00, 100),
        ],
    )


def calculate_mex_components(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["historical_discount_score"] = out.apply(
        lambda r: _historical_discount_score(
            r.get("mlp_c_core"),
            r.get("mlp_z"),
        ),
        axis=1,
    )
    out["velocity_score"] = out.get(
        "mlp_velocity_pct_20",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (-0.05, 0),
                (0.00, 45),
                (0.05, 75),
                (0.10, 100),
            ],
        )
    )
    out["acceleration_score"] = out.get(
        "mlp_acceleration",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (-0.05, 0),
                (0.00, 50),
                (0.025, 75),
                (0.05, 100),
            ],
        )
    )
    out["normalization_gap_score"] = out.get(
        "normalization_gap",
        pd.Series(index=out.index, dtype=float),
    ).map(
        lambda x: _piecewise_score(
            x,
            [
                (0.00, 0),
                (0.10, 40),
                (0.20, 70),
                (0.35, 100),
            ],
        )
    )
    out["rs_score"] = out.apply(
        lambda r: _rs_score(
            r.get("rs_rating"),
            r.get("rs_rating_20d_ago"),
        ),
        axis=1,
    )
    out["volume_score"] = out.get(
        "volume_ratio",
        pd.Series(index=out.index, dtype=float),
    ).map(_volume_score)

    if "fcf_quality_score" not in out.columns:
        out["fcf_quality_score"] = out.get(
            "fcf_quality_grade",
            pd.Series(index=out.index, dtype=object),
        ).map(QUALITY_SCORES)

    scored = out.apply(
        lambda row: _weighted_score(
            row,
            MEX_WEIGHTS,
            min_coverage=0.60,
        ),
        axis=1,
    )
    out["mex_score"] = [x[0] for x in scored]
    out["mex_coverage"] = [x[1] for x in scored]
    return out


def classify_mex_state(
    row: pd.Series | dict[str, Any],
) -> str:
    r = row if isinstance(row, pd.Series) else pd.Series(row)
    fcf = _num(r.get("fcf_engine_score"))
    mlp = _num(r.get("mlp_c_core"))
    z = _num(r.get("mlp_z"))
    vel = _num(r.get("mlp_velocity_20"))
    acc = _num(r.get("mlp_acceleration"))
    confirm = _num(r.get("market_confirm_count"))
    valid_count = _num(r.get("valid_multiple_count"))

    if valid_count is not None and valid_count < 2:
        return "UNAVAILABLE"
    if fcf is None or mlp is None or z is None:
        return "UNAVAILABLE"

    if z >= 2.0 and acc is not None and acc < 0:
        return "EXHAUSTION"
    if (
        fcf < 50
        and mlp >= 1.15
        and vel is not None
        and vel > 0
    ):
        return "SPECULATIVE"
    if (
        fcf >= 70
        and 0.80 <= mlp <= 1.05
        and vel is not None
        and vel > 0
        and acc is not None
        and acc > 0
        and (confirm or 0) >= 2
    ):
        return "IGNITION"
    if (
        fcf >= 60
        and 1.00 < mlp < 1.15
        and vel is not None
        and vel > 0
        and z < 2.0
    ):
        return "EXPANSION"
    if (
        fcf >= 60
        and mlp >= 1.15
        and vel is not None
        and vel >= 0
    ):
        return "MATURE"
    if (
        fcf >= 70
        and mlp <= 1.05
        and vel is not None
        and vel > 0
    ):
        return "PREP"
    if fcf >= 80 and mlp < 0.90 and z < -0.50:
        return "SPRING"
    if fcf >= 70 and mlp < 1.00:
        return "WATCH"
    if (
        vel is not None
        and vel < 0
        and fcf < 60
    ):
        return "DE_RATING"
    if mlp < 0.90:
        return "DISCOUNT"
    if fcf >= 60:
        return "WATCH"
    return "DE_RATING"


def confirm_state_transitions(
    df: pd.DataFrame,
    *,
    confirmation_days: int = 2,
    cooldown_days: int = 20,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    if "state_raw" not in df.columns:
        raise ValueError("state_raw is required")

    out = df.copy()
    out["trade_date"] = pd.to_datetime(
        out["trade_date"],
        errors="coerce",
    )
    out = out.sort_values(
        ["ticker", "trade_date"]
    ).reset_index(drop=True)

    rows = []
    for _, g in out.groupby("ticker", sort=False):
        g = g.copy()
        confirmed = None
        run_state = None
        run_len = 0
        last_event_idx: int | None = None
        ignition_armed = True

        confirmed_values = []
        changed_values = []
        ignition_events = []
        exhaustion_events = []
        spring_prep_events = []

        for local_i, (_, row) in enumerate(g.iterrows()):
            raw = str(
                row.get("state_raw")
                or "UNAVAILABLE"
            )

            if raw == run_state:
                run_len += 1
            else:
                run_state = raw
                run_len = 1

            previous = confirmed
            if raw in {"UNAVAILABLE", "EXHAUSTION"}:
                confirmed = raw
            elif run_len >= confirmation_days:
                confirmed = raw

            changed = (
                confirmed is not None
                and confirmed != previous
            )
            ignition_event = False
            exhaustion_event = (
                changed
                and confirmed == "EXHAUSTION"
            )
            spring_prep_event = (
                changed
                and previous == "SPRING"
                and confirmed == "PREP"
            )

            if confirmed in {
                "DISCOUNT",
                "WATCH",
                "SPRING",
                "DE_RATING",
                "UNAVAILABLE",
            }:
                ignition_armed = True

            if changed and confirmed == "IGNITION":
                cooldown_ok = (
                    last_event_idx is None
                    or (local_i - last_event_idx) >= cooldown_days
                )
                if ignition_armed or cooldown_ok:
                    ignition_event = True
                    last_event_idx = local_i
                    ignition_armed = False

            confirmed_values.append(
                confirmed or "UNAVAILABLE"
            )
            changed_values.append(bool(changed))
            ignition_events.append(bool(ignition_event))
            exhaustion_events.append(bool(exhaustion_event))
            spring_prep_events.append(bool(spring_prep_event))

        g["state_confirmed"] = confirmed_values
        g["state_changed"] = changed_values
        g["ignition_event"] = ignition_events
        g["exhaustion_event"] = exhaustion_events
        g["spring_to_prep_event"] = spring_prep_events
        rows.append(g)

    return pd.concat(rows, ignore_index=True)


def calculate_price_driver(
    df: pd.DataFrame,
    *,
    horizon_days: int = 252,
) -> pd.DataFrame:
    if df.empty:
        return df.copy()
    out = df.copy()
    out["trade_date"] = pd.to_datetime(
        out["trade_date"],
        errors="coerce",
    )
    out = out.sort_values(
        ["ticker", "trade_date"]
    ).reset_index(drop=True)

    def process(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        price = pd.to_numeric(
            g.get("adj_close"),
            errors="coerce",
        )
        fcfps = pd.to_numeric(
            g.get("fcf_per_share_ttm"),
            errors="coerce",
        )
        price_prev = price.shift(horizon_days)
        fcf_prev = fcfps.shift(horizon_days)

        g["price_return_pct"] = (
            price / price_prev - 1.0
        ).where((price > 0) & (price_prev > 0))
        g["fcf_driver_pct"] = (
            fcfps / fcf_prev - 1.0
        ).where((fcfps > 0) & (fcf_prev > 0))

        price_log = (price / price_prev).map(
            lambda x: math.log(x)
            if pd.notna(x) and x > 0
            else float("nan")
        )
        fcf_log = (fcfps / fcf_prev).map(
            lambda x: math.log(x)
            if pd.notna(x) and x > 0
            else float("nan")
        )
        multiple_log = price_log - fcf_log
        g["multiple_driver_pct"] = multiple_log.map(
            lambda x: math.exp(x) - 1.0
            if pd.notna(x)
            else float("nan")
        )

        def label(row: pd.Series) -> str | None:
            f = _num(row.get("fcf_driver_pct"))
            m = _num(row.get("multiple_driver_pct"))
            if f is None or m is None:
                return None
            fp = max(0.0, f)
            mp = max(0.0, m)
            total = fp + mp
            if total <= 0:
                return "BALANCED"
            f_share = fp / total
            if f_share >= 0.65:
                return "FUNDAMENTAL_DRIVEN"
            if f_share <= 0.35:
                return "MULTIPLE_DRIVEN"
            return "BALANCED"

        g["price_driver_label"] = g.apply(
            label,
            axis=1,
        )
        return g

    return (
        out.groupby("ticker", group_keys=False)
        .apply(process)
        .reset_index(drop=True)
    )


def run_multiple_expansion_pipeline(
    market_df: pd.DataFrame,
    financial_df: pd.DataFrame | None = None,
    config: dict[str, Any] | None = None,
) -> pd.DataFrame:
    """Run MEX on market rows and PIT-aligned financial rows.

    v0.1 deliberately refuses to backfill today's fundamentals over
    historical prices. If financial_df has available_date, this function
    aligns it backward-as-of. Otherwise it must already be aligned by
    trade_date + ticker.
    """
    cfg = config or {}
    if financial_df is not None and not financial_df.empty:
        if "available_date" in financial_df.columns:
            df = build_point_in_time_financials(
                financial_df,
                market_df,
            )
        else:
            keys = [
                k
                for k in ["trade_date", "ticker"]
                if (
                    k in market_df.columns
                    and k in financial_df.columns
                )
            ]
            if len(keys) != 2:
                raise ValueError(
                    "financial_df must contain either available_date "
                    "for PIT alignment or both trade_date and ticker "
                    "for pre-aligned input"
                )
            df = market_df.merge(
                financial_df,
                on=keys,
                how="left",
                suffixes=("", "_fin"),
            )
    else:
        df = market_df.copy()

    df = calculate_valuation_multiples(df)
    df = calculate_multiple_normalization(
        df,
        lookback_days=int(
            cfg.get("lookback_days", 1260)
        ),
        min_history_days=int(
            cfg.get("min_history_days", 504)
        ),
        core_ema_span=int(
            cfg.get("core_ema_span", 40)
        ),
    )
    df = calculate_fcf_engine(df)
    df = classify_fcf_quality(df)
    df = calculate_market_confirmation(df)
    df = calculate_mex_components(df)
    df["state_raw"] = df.apply(
        classify_mex_state,
        axis=1,
    )
    df = confirm_state_transitions(
        df,
        confirmation_days=int(
            cfg.get("confirmation_days", 2)
        ),
        cooldown_days=int(
            cfg.get("cooldown_days", 20)
        ),
    )
    df = calculate_price_driver(
        df,
        horizon_days=int(
            cfg.get("price_driver_horizon_days", 252)
        ),
    )
    return df

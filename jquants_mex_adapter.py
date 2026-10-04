from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import pandas as pd
import requests


JQUANTS_V2_BASE = "https://api.jquants.com/v2"

PERIOD_ORDER = {
    "1Q": 1,
    "Q1": 1,
    "1QFY": 1,
    "2Q": 2,
    "Q2": 2,
    "2QFY": 2,
    "3Q": 3,
    "Q3": 3,
    "3QFY": 3,
    "FY": 4,
    "4Q": 4,
    "Q4": 4,
}

# Conservative label hints for EDINET verbose-English labels returned inside
# J-Quants /fins/details FS dictionaries.  A fuzzy match is accepted only
# when it identifies a single field.  Ambiguous matches remain unavailable.
CAPEX_LABEL_HINTS = (
    "purchase of property plant and equipment",
    "purchase of property, plant and equipment",
    "payments for purchase of property plant and equipment",
    "purchase of tangible fixed assets",
    "purchase of non-current assets",
    "purchase of property plant equipment and intangible assets",
    "purchase of property, plant and equipment and intangible assets",
)

DEPRECIATION_LABEL_HINTS = (
    "depreciation and amortization",
    "depreciation",
)

TOTAL_DEBT_LABEL_HINTS = (
    "interest-bearing debt",
    "interest bearing debt",
    "short-term borrowings",
    "short term borrowings",
    "long-term borrowings",
    "long term borrowings",
    "bonds payable",
)


def _num(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        if isinstance(value, str):
            text = value.strip().replace(",", "")
            if not text:
                return None
            value = text
        out = float(value)
        if not math.isfinite(out):
            return None
        return out
    except (TypeError, ValueError):
        return None


def _date_string(value: Any) -> str:
    if value is None:
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return ts.strftime("%Y%m%d")


def _to_code4(value: Any) -> str:
    text = str(value or "").strip()
    if len(text) == 5 and text.endswith("0"):
        return text[:4]
    return text


def _norm_label(value: str) -> str:
    text = value.casefold()
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _parse_fs(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return {}
        try:
            parsed = json.loads(value)
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}
    return {}


def _find_unique_fs_value(
    fs: dict[str, Any],
    hints: tuple[str, ...],
) -> tuple[float | None, str, str]:
    """Return (value, matched_key, status) with conservative ambiguity rules."""
    if not fs:
        return None, "", "MISSING_FS"

    normalized_hints = tuple(_norm_label(x) for x in hints)
    exact_matches: list[tuple[str, float]] = []
    fuzzy_matches: list[tuple[str, float]] = []

    for key, raw in fs.items():
        value = _num(raw)
        if value is None:
            continue
        nk = _norm_label(str(key))
        if any(nk == hint for hint in normalized_hints):
            exact_matches.append((str(key), value))
            continue
        if any(hint in nk or nk in hint for hint in normalized_hints):
            fuzzy_matches.append((str(key), value))

    if len(exact_matches) == 1:
        key, value = exact_matches[0]
        return value, key, "EXACT"
    if len(exact_matches) > 1:
        return None, "", "AMBIGUOUS_EXACT"
    if len(fuzzy_matches) == 1:
        key, value = fuzzy_matches[0]
        return value, key, "FUZZY_UNIQUE"
    if len(fuzzy_matches) > 1:
        return None, "", "AMBIGUOUS_FUZZY"
    return None, "", "NOT_FOUND"


@dataclass
class JQuantsV2Client:
    api_key: str
    base_url: str = JQUANTS_V2_BASE
    timeout: int = 30

    @classmethod
    def from_env(cls) -> "JQuantsV2Client":
        api_key = os.environ.get("JQUANTS_API_KEY", "").strip()
        if not api_key:
            raise ValueError(
                "JQUANTS_API_KEY is required. "
                "Issue an API key in the J-Quants dashboard and set it as a secret."
            )
        return cls(api_key=api_key)

    def _get_paginated(
        self,
        path: str,
        params: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        url = f"{self.base_url}{path}"
        query = {
            k: v
            for k, v in (params or {}).items()
            if v not in (None, "")
        }
        out: list[dict[str, Any]] = []

        while True:
            response = requests.get(
                url,
                params=query,
                headers={"x-api-key": self.api_key},
                timeout=self.timeout,
            )
            if response.status_code == 403:
                raise PermissionError(
                    f"J-Quants plan does not permit {path}, or the API key lacks access."
                )
            response.raise_for_status()
            payload = response.json()
            batch = payload.get("data", [])
            if isinstance(batch, list):
                out.extend(batch)

            pagination_key = payload.get("pagination_key")
            if not pagination_key:
                break
            query["pagination_key"] = pagination_key

        return out

    def fin_summary(
        self,
        *,
        code: str = "",
        date: str = "",
    ) -> pd.DataFrame:
        rows = self._get_paginated(
            "/fins/summary",
            {"code": code, "date": date},
        )
        return pd.DataFrame.from_records(rows)

    def fin_details(
        self,
        *,
        code: str = "",
        date: str = "",
    ) -> pd.DataFrame:
        rows = self._get_paginated(
            "/fins/details",
            {"code": code, "date": date},
        )
        return pd.DataFrame.from_records(rows)

    def daily_bars(
        self,
        *,
        code: str,
        from_date: str = "",
        to_date: str = "",
        date: str = "",
    ) -> pd.DataFrame:
        rows = self._get_paginated(
            "/equities/bars/daily",
            {
                "code": code,
                "from": from_date,
                "to": to_date,
                "date": date,
            },
        )
        return pd.DataFrame.from_records(rows)

    def valuation(
        self,
        *,
        code: str,
        from_date: str = "",
        to_date: str = "",
        date: str = "",
    ) -> pd.DataFrame:
        rows = self._get_paginated(
            "/equities/valuation",
            {
                "code": code,
                "from": from_date,
                "to": to_date,
                "date": date,
            },
        )
        return pd.DataFrame.from_records(rows)

    def topix(
        self,
        *,
        from_date: str = "",
        to_date: str = "",
        date: str = "",
    ) -> pd.DataFrame:
        rows = self._get_paginated(
            "/indices/bars/daily/topix",
            {
                "from": from_date,
                "to": to_date,
                "date": date,
            },
        )
        return pd.DataFrame.from_records(rows)


def normalize_jquants_summary(raw: pd.DataFrame) -> pd.DataFrame:
    """Normalize /v2/fins/summary without inventing unavailable fields."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    df = raw.copy()
    rename = {
        "DiscDate": "available_date",
        "DiscTime": "available_time",
        "Code": "jquants_code",
        "DiscNo": "disclosure_number",
        "DocType": "document_type",
        "CurPerType": "period_type",
        "CurPerSt": "fiscal_period_start",
        "CurPerEn": "fiscal_period_end",
        "CurFYSt": "fiscal_year_start",
        "CurFYEn": "fiscal_year_end",
        "Sales": "revenue_cumulative",
        "OP": "operating_income_cumulative",
        "NP": "net_income_cumulative",
        "EPS": "eps_cumulative",
        "TA": "total_assets",
        "Eq": "equity",
        "ROE": "roe_reported",
        "CFO": "cfo_cumulative",
        "CFI": "cfi_cumulative",
        "CFF": "cff_cumulative",
        "CashEq": "cash_and_equivalents",
        "ShOutFY": "shares_outstanding_fy",
        "TrShFY": "treasury_shares_fy",
        "AvgSh": "average_shares_cumulative",
    }
    df = df.rename(columns=rename)

    df["ticker"] = df["jquants_code"].map(_to_code4)
    for col in [
        "available_date",
        "fiscal_period_start",
        "fiscal_period_end",
        "fiscal_year_start",
        "fiscal_year_end",
    ]:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df["available_time"] = df.get(
        "available_time",
        pd.Series("", index=df.index),
    ).fillna("").astype(str)
    df["available_datetime"] = pd.to_datetime(
        df["available_date"].dt.strftime("%Y-%m-%d")
        + " "
        + df["available_time"].replace("", "23:59"),
        errors="coerce",
    )

    numeric_cols = [
        "revenue_cumulative",
        "operating_income_cumulative",
        "net_income_cumulative",
        "eps_cumulative",
        "total_assets",
        "equity",
        "roe_reported",
        "cfo_cumulative",
        "cfi_cumulative",
        "cff_cumulative",
        "cash_and_equivalents",
        "shares_outstanding_fy",
        "treasury_shares_fy",
        "average_shares_cumulative",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df["period_order"] = (
        df.get("period_type", pd.Series("", index=df.index))
        .fillna("")
        .astype(str)
        .str.upper()
        .str.replace(" ", "", regex=False)
        .map(PERIOD_ORDER)
    )

    return df.sort_values(
        ["ticker", "available_datetime", "disclosure_number"],
        na_position="last",
    ).reset_index(drop=True)


def normalize_jquants_details(raw: pd.DataFrame) -> pd.DataFrame:
    """Extract only fields we can identify conservatively from /fins/details."""
    if raw is None or raw.empty:
        return pd.DataFrame()

    rows = []
    for _, row in raw.iterrows():
        fs = _parse_fs(row.get("FS"))
        capex, capex_key, capex_status = _find_unique_fs_value(
            fs,
            CAPEX_LABEL_HINTS,
        )
        depreciation, dep_key, dep_status = _find_unique_fs_value(
            fs,
            DEPRECIATION_LABEL_HINTS,
        )

        # Cash-flow purchase values are normally negative in statements.
        # Store CAPEX as a positive cash-outflow magnitude.
        capex_mag = abs(capex) if capex is not None else None

        rows.append({
            "jquants_code": row.get("Code"),
            "ticker": _to_code4(row.get("Code")),
            "available_date": pd.to_datetime(
                row.get("DiscDate"),
                errors="coerce",
            ),
            "available_time": str(row.get("DiscTime") or ""),
            "disclosure_number": row.get("DiscNo"),
            "document_type": row.get("DocType"),
            "capex_cumulative": capex_mag,
            "capex_source_key": capex_key,
            "capex_parse_status": capex_status,
            "depreciation_cumulative": depreciation,
            "depreciation_source_key": dep_key,
            "depreciation_parse_status": dep_status,
            "fs_item_count": len(fs),
        })

    out = pd.DataFrame(rows)
    out["available_datetime"] = pd.to_datetime(
        out["available_date"].dt.strftime("%Y-%m-%d")
        + " "
        + out["available_time"].replace("", "23:59"),
        errors="coerce",
    )
    return out.sort_values(
        ["ticker", "available_datetime", "disclosure_number"],
        na_position="last",
    ).reset_index(drop=True)


def merge_summary_and_details(
    summary: pd.DataFrame,
    details: pd.DataFrame,
) -> pd.DataFrame:
    if summary is None or summary.empty:
        return pd.DataFrame()
    if details is None or details.empty:
        out = summary.copy()
        out["capex_cumulative"] = pd.NA
        out["capex_parse_status"] = "DETAILS_UNAVAILABLE"
        out["depreciation_cumulative"] = pd.NA
        return out

    right_cols = [
        c
        for c in [
            "ticker",
            "disclosure_number",
            "capex_cumulative",
            "capex_source_key",
            "capex_parse_status",
            "depreciation_cumulative",
            "depreciation_source_key",
            "depreciation_parse_status",
            "fs_item_count",
        ]
        if c in details.columns
    ]
    return summary.merge(
        details[right_cols],
        on=["ticker", "disclosure_number"],
        how="left",
    )


def _quarter_from_cumulative(
    state: dict[tuple[pd.Timestamp, int], pd.Series],
    fiscal_year_end: pd.Timestamp,
    order: int,
    field: str,
) -> float | None:
    current = state.get((fiscal_year_end, order))
    if current is None:
        return None
    cur = _num(current.get(field))
    if cur is None:
        return None
    if order == 1:
        return cur
    previous = state.get((fiscal_year_end, order - 1))
    if previous is None:
        return None
    prev = _num(previous.get(field))
    if prev is None:
        return None
    return cur - prev


def _derive_event_snapshot(
    state: dict[tuple[pd.Timestamp, int], pd.Series],
) -> dict[str, Any]:
    if not state:
        return {}

    period_rows = []
    for (fy_end, order), row in state.items():
        period_end = pd.to_datetime(
            row.get("fiscal_period_end"),
            errors="coerce",
        )
        if pd.isna(period_end):
            continue

        q = {
            "fiscal_year_end": fy_end,
            "period_order": order,
            "fiscal_period_end": period_end,
            "revenue_q": _quarter_from_cumulative(
                state, fy_end, order, "revenue_cumulative"
            ),
            "operating_income_q": _quarter_from_cumulative(
                state, fy_end, order, "operating_income_cumulative"
            ),
            "net_income_q": _quarter_from_cumulative(
                state, fy_end, order, "net_income_cumulative"
            ),
            "cfo_q": _quarter_from_cumulative(
                state, fy_end, order, "cfo_cumulative"
            ),
            "capex_q": _quarter_from_cumulative(
                state, fy_end, order, "capex_cumulative"
            ),
            "depreciation_q": _quarter_from_cumulative(
                state, fy_end, order, "depreciation_cumulative"
            ),
        }
        if q["cfo_q"] is not None and q["capex_q"] is not None:
            q["fcf_q"] = q["cfo_q"] - q["capex_q"]
        else:
            q["fcf_q"] = None
        period_rows.append(q)

    if not period_rows:
        return {}

    qdf = pd.DataFrame(period_rows).sort_values(
        ["fiscal_period_end", "period_order"]
    )
    latest = qdf.iloc[-1]
    last4 = qdf.tail(4)

    def rolling4(col: str) -> float | None:
        if len(last4) != 4:
            return None
        vals = pd.to_numeric(last4[col], errors="coerce")
        if vals.isna().any():
            return None
        return float(vals.sum())

    revenue_ttm = rolling4("revenue_q")
    op_ttm = rolling4("operating_income_q")
    net_income_ttm = rolling4("net_income_q")
    cfo_ttm = rolling4("cfo_q")
    capex_ttm = rolling4("capex_q")
    fcf_ttm = rolling4("fcf_q")
    depreciation_ttm = rolling4("depreciation_q")
    ebitda_ttm = (
        op_ttm + depreciation_ttm
        if op_ttm is not None and depreciation_ttm is not None
        else None
    )

    latest_state = state[
        (latest["fiscal_year_end"], int(latest["period_order"]))
    ]

    shares_out = _num(latest_state.get("shares_outstanding_fy"))
    treasury = _num(latest_state.get("treasury_shares_fy"))
    diluted_shares = _num(latest_state.get("average_shares_cumulative"))

    if shares_out is not None and treasury is not None:
        shares_outstanding_pti = shares_out - treasury
    else:
        shares_outstanding_pti = shares_out

    if diluted_shares is None:
        diluted_shares = shares_outstanding_pti

    fcf_per_share_ttm = (
        fcf_ttm / diluted_shares
        if fcf_ttm is not None
        and diluted_shares is not None
        and diluted_shares > 0
        else None
    )

    return {
        "fiscal_period_end": latest["fiscal_period_end"],
        "fiscal_year_end": latest["fiscal_year_end"],
        "revenue_ttm": revenue_ttm,
        "operating_income_ttm": op_ttm,
        "net_income_ttm": net_income_ttm,
        "cfo_ttm": cfo_ttm,
        "capex_ttm": capex_ttm,
        "fcf_ttm": fcf_ttm,
        "depreciation_ttm": depreciation_ttm,
        "ebitda_ttm": ebitda_ttm,
        "fcf_per_share_ttm": fcf_per_share_ttm,
        "cash_and_equivalents": _num(
            latest_state.get("cash_and_equivalents")
        ),
        "total_assets": _num(latest_state.get("total_assets")),
        "equity": _num(latest_state.get("equity")),
        "roe": _num(latest_state.get("roe_reported")),
        "shares_outstanding_pti": shares_outstanding_pti,
        "diluted_shares_ttm": diluted_shares,
        "ttm_quarter_count": int(len(last4)),
        "fcf_ttm_complete": fcf_ttm is not None,
    }


def build_point_in_time_financial_events(
    disclosures: pd.DataFrame,
) -> pd.DataFrame:
    """Event-driven TTM builder that respects disclosure/revision timing.

    Each disclosure updates the known state only from that disclosure onward.
    Corrections are therefore not backfilled into earlier dates.
    """
    if disclosures is None or disclosures.empty:
        return pd.DataFrame()

    required = {
        "ticker",
        "available_date",
        "available_datetime",
        "fiscal_year_end",
        "period_order",
    }
    missing = required - set(disclosures.columns)
    if missing:
        raise ValueError(
            "disclosures missing required columns: "
            + ", ".join(sorted(missing))
        )

    rows = []
    for ticker, group in disclosures.groupby("ticker", sort=False):
        state: dict[tuple[pd.Timestamp, int], pd.Series] = {}
        group = group.sort_values(
            ["available_datetime", "disclosure_number"],
            na_position="last",
        )
        for _, event in group.iterrows():
            fy_end = pd.to_datetime(
                event.get("fiscal_year_end"),
                errors="coerce",
            )
            order = _num(event.get("period_order"))
            if pd.isna(fy_end) or order is None:
                continue

            state[(fy_end, int(order))] = event
            snapshot = _derive_event_snapshot(state)
            if not snapshot:
                continue

            out = {
                "ticker": ticker,
                "available_date": pd.to_datetime(
                    event.get("available_date"),
                    errors="coerce",
                ),
                "available_datetime": pd.to_datetime(
                    event.get("available_datetime"),
                    errors="coerce",
                ),
                "disclosure_number": event.get("disclosure_number"),
                "document_type": event.get("document_type"),
                "capex_parse_status": event.get("capex_parse_status"),
                **snapshot,
            }
            rows.append(out)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(
        ["ticker", "available_datetime"]
    ).reset_index(drop=True)



def add_financial_event_growth_features(
    events: pd.DataFrame,
) -> pd.DataFrame:
    """Add PIT-safe 1Y/3Y growth features using only earlier event snapshots.

    Comparison rows are selected from snapshots that were already available
    around the target anniversary. No future disclosure is pulled backward.
    """
    if events is None or events.empty:
        return pd.DataFrame() if events is None else events.copy()

    out_rows = []
    for _, group in events.groupby("ticker", sort=False):
        g = group.copy().sort_values("available_datetime").reset_index(drop=True)

        # Calculate margins/conversion directly from each as-known snapshot.
        revenue = pd.to_numeric(g.get("revenue_ttm"), errors="coerce")
        op = pd.to_numeric(g.get("operating_income_ttm"), errors="coerce")
        cfo = pd.to_numeric(g.get("cfo_ttm"), errors="coerce")
        fcf = pd.to_numeric(g.get("fcf_ttm"), errors="coerce")
        shares = pd.to_numeric(g.get("shares_outstanding_pti"), errors="coerce")

        g["fcf_margin"] = (fcf / revenue).where(revenue > 0)
        g["operating_margin"] = (op / revenue).where(revenue > 0)
        g["cash_conversion"] = (fcf / op).where(op > 0)

        # Use latest snapshot available on or before the anniversary target.
        times = pd.to_datetime(g["available_datetime"], errors="coerce")

        def prior_index(i: int, years: int) -> int | None:
            now = times.iloc[i]
            if pd.isna(now):
                return None
            target = now - pd.DateOffset(years=years)
            candidates = times.iloc[:i]
            valid = candidates[candidates <= target]
            if valid.empty:
                return None
            j = int(valid.index[-1])
            # Avoid matching a very stale snapshot when event history is sparse.
            gap_days = (target - times.iloc[j]).days
            if gap_days > 220:
                return None
            return j

        fcf_yoy = []
        cfo_yoy = []
        share_yoy = []
        revenue_yoy = []
        op_yoy = []
        fcf_cagr3 = []

        for i in range(len(g)):
            j1 = prior_index(i, 1)
            j3 = prior_index(i, 3)

            def growth(col: str, j: int | None) -> float | None:
                if j is None:
                    return None
                cur = _num(g.iloc[i].get(col))
                prev = _num(g.iloc[j].get(col))
                if cur is None or prev is None or prev <= 0:
                    return None
                return cur / prev - 1.0

            fcf_yoy.append(growth("fcf_per_share_ttm", j1))
            cfo_yoy.append(growth("cfo_ttm", j1))
            share_yoy.append(growth("shares_outstanding_pti", j1))
            revenue_yoy.append(growth("revenue_ttm", j1))
            op_yoy.append(growth("operating_income_ttm", j1))

            if j3 is None:
                fcf_cagr3.append(None)
            else:
                cur = _num(g.iloc[i].get("fcf_per_share_ttm"))
                prev = _num(g.iloc[j3].get("fcf_per_share_ttm"))
                if cur is None or prev is None or cur <= 0 or prev <= 0:
                    fcf_cagr3.append(None)
                else:
                    days = (
                        times.iloc[i] - times.iloc[j3]
                    ).days
                    years = days / 365.25 if days > 0 else 0
                    if years < 2.4:
                        fcf_cagr3.append(None)
                    else:
                        fcf_cagr3.append(
                            (cur / prev) ** (1.0 / years) - 1.0
                        )

        g["fcf_per_share_yoy"] = fcf_yoy
        g["cfo_yoy"] = cfo_yoy
        g["share_count_1y_change"] = share_yoy
        g["revenue_yoy"] = revenue_yoy
        g["operating_income_yoy"] = op_yoy
        g["fcf_per_share_3y_cagr"] = fcf_cagr3
        out_rows.append(g)

    return pd.concat(out_rows, ignore_index=True)


def normalize_jquants_daily_market(
    bars: pd.DataFrame,
    valuation: pd.DataFrame,
) -> pd.DataFrame:
    """Normalize J-Quants daily bars/valuation into MEX market columns."""
    if bars is None or bars.empty:
        return pd.DataFrame()

    market = bars.copy()
    market["trade_date"] = pd.to_datetime(
        market.get("Date"),
        errors="coerce",
    )
    market["ticker"] = market.get(
        "Code",
        pd.Series("", index=market.index),
    ).map(_to_code4)
    market["adj_close"] = pd.to_numeric(
        market.get("AdjC", market.get("C")),
        errors="coerce",
    )
    market["volume"] = pd.to_numeric(
        market.get("AdjVo", market.get("Vo")),
        errors="coerce",
    )

    keep = [
        "trade_date",
        "ticker",
        "adj_close",
        "volume",
    ]
    market = market[keep].dropna(
        subset=["trade_date", "ticker", "adj_close"]
    )

    if valuation is not None and not valuation.empty:
        val = valuation.copy()
        val["trade_date"] = pd.to_datetime(
            val.get("Date"),
            errors="coerce",
        )
        val["ticker"] = val.get(
            "Code",
            pd.Series("", index=val.index),
        ).map(_to_code4)
        # J-Quants V2 MktCap is expressed in million yen. Keep the native
        # unit because /fins/summary monetary values are also consumed in
        # their API-native monetary unit for ratio calculations.
        val["market_cap_pti"] = pd.to_numeric(
            val.get("MktCap"),
            errors="coerce",
        )
        val["jquants_per"] = pd.to_numeric(
            val.get("PER"),
            errors="coerce",
        )
        market = market.merge(
            val[
                [
                    "trade_date",
                    "ticker",
                    "market_cap_pti",
                    "jquants_per",
                ]
            ],
            on=["trade_date", "ticker"],
            how="left",
        )

    market = market.sort_values(
        ["ticker", "trade_date"]
    ).reset_index(drop=True)

    g = market.groupby("ticker", group_keys=False)
    market["ema20"] = g["adj_close"].transform(
        lambda s: s.ewm(span=20, adjust=False).mean()
    )
    market["sma50"] = g["adj_close"].transform(
        lambda s: s.rolling(50).mean()
    )
    market["sma200"] = g["adj_close"].transform(
        lambda s: s.rolling(200).mean()
    )
    market["volume_ma20"] = g["volume"].transform(
        lambda s: s.shift(1).rolling(20).mean()
    )
    market["high_52w"] = g["adj_close"].transform(
        lambda s: s.rolling(252, min_periods=60).max()
    )
    market["low_52w"] = g["adj_close"].transform(
        lambda s: s.rolling(252, min_periods=60).min()
    )
    return market


def add_simple_rs_proxy(
    market: pd.DataFrame,
    topix: pd.DataFrame | None,
) -> pd.DataFrame:
    """Add a bounded 0-100 relative-strength proxy, explicitly not IBD RS."""
    out = market.copy()
    if out.empty:
        return out

    if topix is None or topix.empty:
        out["rs_rating"] = pd.NA
        out["rs_rating_20d_ago"] = pd.NA
        return out

    idx = topix.copy()
    idx["trade_date"] = pd.to_datetime(
        idx.get("Date"),
        errors="coerce",
    )
    idx["topix_close"] = pd.to_numeric(
        idx.get("C"),
        errors="coerce",
    )
    idx = idx[["trade_date", "topix_close"]].dropna()

    out = out.merge(idx, on="trade_date", how="left")

    def process(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        stock = pd.to_numeric(g["adj_close"], errors="coerce")
        bench = pd.to_numeric(g["topix_close"], errors="coerce")

        stock63 = stock / stock.shift(63) - 1.0
        stock126 = stock / stock.shift(126) - 1.0
        bench63 = bench / bench.shift(63) - 1.0
        bench126 = bench / bench.shift(126) - 1.0

        excess = (
            (stock63 - bench63) * 0.60
            + (stock126 - bench126) * 0.40
        )
        g["rs_rating"] = (
            50.0 + excess * 180.0
        ).clip(lower=0.0, upper=100.0)
        g["rs_rating_20d_ago"] = g["rs_rating"].shift(20)
        return g

    return (
        out.groupby("ticker", group_keys=False)
        .apply(process)
        .reset_index(drop=True)
    )


def fetch_ticker_bundle(
    client: JQuantsV2Client,
    *,
    code: str,
    from_date: str,
    to_date: str,
    include_details: bool = True,
    include_topix: bool = True,
) -> dict[str, pd.DataFrame]:
    """Fetch one ticker bundle. No data is written or backfilled here."""
    summary_raw = client.fin_summary(code=code)
    details_raw = pd.DataFrame()
    details_error = ""

    if include_details:
        try:
            details_raw = client.fin_details(code=code)
        except PermissionError as exc:
            details_error = str(exc)

    bars = client.daily_bars(
        code=code,
        from_date=_date_string(from_date),
        to_date=_date_string(to_date),
    )
    valuation = client.valuation(
        code=code,
        from_date=_date_string(from_date),
        to_date=_date_string(to_date),
    )
    topix = (
        client.topix(
            from_date=_date_string(from_date),
            to_date=_date_string(to_date),
        )
        if include_topix
        else pd.DataFrame()
    )

    summary = normalize_jquants_summary(summary_raw)
    details = normalize_jquants_details(details_raw)
    disclosures = merge_summary_and_details(summary, details)
    financial_events = build_point_in_time_financial_events(disclosures)
    financial_events = add_financial_event_growth_features(financial_events)
    market = normalize_jquants_daily_market(bars, valuation)
    market = add_simple_rs_proxy(market, topix)

    meta = pd.DataFrame([{
        "code": code,
        "details_available": not details_raw.empty,
        "details_error": details_error,
        "summary_rows": len(summary_raw),
        "detail_rows": len(details_raw),
        "financial_event_rows": len(financial_events),
        "market_rows": len(market),
        "fcf_ttm_complete_events": (
            int(financial_events["fcf_ttm_complete"].fillna(False).sum())
            if not financial_events.empty
            and "fcf_ttm_complete" in financial_events.columns
            else 0
        ),
    }])

    return {
        "summary_raw": summary_raw,
        "details_raw": details_raw,
        "summary": summary,
        "details": details,
        "disclosures": disclosures,
        "financial_events": financial_events,
        "market": market,
        "topix": topix,
        "meta": meta,
    }

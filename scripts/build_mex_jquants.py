from __future__ import annotations

import argparse
import os
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

from jquants_mex_adapter import (
    JQuantsV2Client,
    add_simple_rs_proxy,
    fetch_ticker_bundle,
)
from multiple_expansion import run_multiple_expansion_pipeline


OUT_DIR = Path("data/multiple_expansion")


def _codes_from_portfolio(path: str) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        df = pd.read_csv(path, dtype=str)
    except Exception:
        return []

    for col in ["ticker", "code", "銘柄コード"]:
        if col in df.columns:
            values = df[col].dropna().astype(str).str.strip()
            return [
                v[:-2] if v.endswith(".T") else v
                for v in values
                if v
            ]
    return []


def _parse_codes(value: str) -> list[str]:
    return [
        x.strip()
        for x in value.split(",")
        if x.strip()
    ]


def _default_from_date() -> str:
    # Request a little more than five years where the plan permits it.
    return (date.today() - timedelta(days=365 * 5 + 45)).isoformat()


def _write_outputs(result: pd.DataFrame) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    history_path = OUT_DIR / "mex_history.csv"
    latest_path = OUT_DIR / "mex_latest.csv"
    events_path = OUT_DIR / "mex_events.csv"

    result = result.sort_values(["ticker", "trade_date"]).reset_index(drop=True)
    result.to_csv(history_path, index=False)

    latest = (
        result.sort_values(["ticker", "trade_date"])
        .groupby("ticker", as_index=False)
        .tail(1)
        .copy()
    )
    latest = latest.sort_values(
        ["mex_score", "mex_coverage", "ticker"],
        ascending=[False, False, True],
        na_position="last",
    )
    latest.to_csv(latest_path, index=False)

    event_mask = pd.Series(False, index=result.index)
    for col in [
        "ignition_event",
        "exhaustion_event",
        "spring_to_prep_event",
        "state_changed",
    ]:
        if col in result.columns:
            event_mask |= result[col].fillna(False).astype(bool)

    event_cols = [
        c
        for c in [
            "trade_date",
            "ticker",
            "state_raw",
            "state_confirmed",
            "state_changed",
            "ignition_event",
            "exhaustion_event",
            "spring_to_prep_event",
            "mex_score",
            "fcf_engine_score",
            "fcf_basis",
            "mlp_c_core",
            "mlp_z",
            "mlp_velocity_20",
            "mlp_acceleration",
            "market_confirm_count",
            "data_quality",
        ]
        if c in result.columns
    ]
    result.loc[event_mask, event_cols].to_csv(events_path, index=False)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build Multiple Expansion Hunter data from J-Quants V2 "
            "without backfilling future financial disclosures."
        )
    )
    parser.add_argument(
        "--codes",
        default="",
        help="Comma-separated 4-digit TSE codes. If omitted, portfolio_data.csv is used.",
    )
    parser.add_argument(
        "--portfolio",
        default="portfolio_data.csv",
        help="Portfolio CSV used when --codes is omitted.",
    )
    parser.add_argument(
        "--from-date",
        default=_default_from_date(),
        help="Market-data start date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--to-date",
        default=date.today().isoformat(),
        help="Market-data end date, YYYY-MM-DD.",
    )
    parser.add_argument(
        "--allow-summary-proxy",
        action="store_true",
        help=(
            "Allow CFO+CFI summary-only cash-flow proxy when exact CAPEX "
            "from /fins/details is unavailable. The output remains labeled as proxy."
        ),
    )
    args = parser.parse_args()

    codes = _parse_codes(args.codes)
    if not codes:
        codes = _codes_from_portfolio(args.portfolio)
    codes = sorted(set(codes))
    if not codes:
        raise SystemExit(
            "No ticker codes. Pass --codes 7203,6758 or provide portfolio_data.csv."
        )

    client = JQuantsV2Client.from_env()

    # TOPIX is fetched once and reused across tickers.
    try:
        topix = client.topix(
            from_date=args.from_date.replace("-", ""),
            to_date=args.to_date.replace("-", ""),
        )
    except PermissionError:
        topix = pd.DataFrame()

    results = []
    diagnostics = []

    for code in codes:
        print(f"[MEX] fetching {code} ...")
        try:
            bundle = fetch_ticker_bundle(
                client,
                code=code,
                from_date=args.from_date,
                to_date=args.to_date,
                include_details=True,
                include_topix=False,
            )
        except Exception as exc:
            diagnostics.append({
                "code": code,
                "status": "ERROR",
                "reason": str(exc),
            })
            print(f"[MEX] {code}: ERROR {exc}")
            continue

        market = add_simple_rs_proxy(bundle["market"], topix)
        financial = bundle["financial_events"].copy()

        if financial.empty:
            diagnostics.append({
                "code": code,
                "status": "NO_FINANCIAL_EVENTS",
                "reason": "No PIT financial events were built.",
            })
            continue

        if not args.allow_summary_proxy:
            if "fcf_exact" in financial.columns:
                financial.loc[
                    ~financial["fcf_exact"].fillna(False).astype(bool),
                    ["fcf_ttm", "fcf_per_share_ttm"],
                ] = pd.NA

        try:
            mex = run_multiple_expansion_pipeline(
                market,
                financial,
            )
        except Exception as exc:
            diagnostics.append({
                "code": code,
                "status": "PIPELINE_ERROR",
                "reason": str(exc),
            })
            print(f"[MEX] {code}: PIPELINE_ERROR {exc}")
            continue

        if "fcf_basis" in mex.columns:
            mex["data_quality"] = mex["fcf_basis"].map(
                lambda x: (
                    "FRESH"
                    if x == "CFO_MINUS_CAPEX"
                    else (
                        "PARTIAL"
                        if x == "CFO_PLUS_CFI_PROXY"
                        else "UNAVAILABLE"
                    )
                )
            )
        else:
            mex["data_quality"] = "PARTIAL"

        mex["data_asof"] = args.to_date
        results.append(mex)

        meta = bundle["meta"].iloc[0].to_dict()
        meta["status"] = "OK"
        diagnostics.append(meta)
        print(
            f"[MEX] {code}: {len(mex)} market rows, "
            f"{len(financial)} financial events"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(
        OUT_DIR / "mex_build_diagnostics.csv",
        index=False,
    )

    if not results:
        print("[MEX] No results were produced.")
        return 2

    result = pd.concat(results, ignore_index=True)
    _write_outputs(result)

    print(f"[MEX] wrote {OUT_DIR / 'mex_latest.csv'}")
    print(f"[MEX] wrote {OUT_DIR / 'mex_history.csv'}")
    print(f"[MEX] wrote {OUT_DIR / 'mex_events.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

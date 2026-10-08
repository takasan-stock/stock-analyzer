import unittest

import pandas as pd

from jquants_mex_adapter import (
    add_financial_event_growth_features,
    build_point_in_time_financial_events,
    merge_summary_and_details,
    normalize_jquants_daily_market,
    normalize_jquants_details,
    normalize_jquants_summary,
)


class JQuantsMexAdapterTests(unittest.TestCase):
    def _summary_row(
        self,
        *,
        date,
        disc_no,
        period,
        fy_end,
        per_end,
        sales,
        op,
        np,
        cfo=None,
        cfi=None,
        cash=100,
        sh_out=100,
        tr_sh=0,
        avg_sh=100,
    ):
        return {
            "DiscDate": date,
            "DiscTime": "15:00",
            "Code": "11110",
            "DiscNo": disc_no,
            "DocType": f"{period}FinancialStatements_Consolidated_JP",
            "CurPerType": period,
            "CurPerSt": str(pd.Timestamp(fy_end) - pd.DateOffset(years=1) + pd.Timedelta(days=1))[:10],
            "CurPerEn": per_end,
            "CurFYSt": str(pd.Timestamp(fy_end) - pd.DateOffset(years=1) + pd.Timedelta(days=1))[:10],
            "CurFYEn": fy_end,
            "Sales": sales,
            "OP": op,
            "NP": np,
            "EPS": np / avg_sh,
            "TA": 1000,
            "Eq": 600,
            "ROE": 0.10,
            "CFO": cfo,
            "CFI": cfi,
            "CFF": -10,
            "CashEq": cash,
            "ShOutFY": sh_out,
            "TrShFY": tr_sh,
            "AvgSh": avg_sh,
        }

    def test_summary_normalizes_disclosure_time_and_code(self):
        raw = pd.DataFrame([
            self._summary_row(
                date="2026-05-10",
                disc_no="D1",
                period="FY",
                fy_end="2026-03-31",
                per_end="2026-03-31",
                sales=1000,
                op=100,
                np=60,
                cfo=120,
                cfi=-50,
            )
        ])
        out = normalize_jquants_summary(raw).iloc[0]
        self.assertEqual(out["ticker"], "1111")
        self.assertEqual(out["period_order"], 4)
        self.assertEqual(
            out["available_datetime"],
            pd.Timestamp("2026-05-10 15:00"),
        )

    def test_details_ambiguous_capex_is_not_guessed(self):
        raw = pd.DataFrame([{
            "DiscDate": "2026-05-10",
            "DiscTime": "15:00",
            "Code": "11110",
            "DiscNo": "D1",
            "DocType": "FYFinancialStatements_Consolidated_JP",
            "FS": {
                "Purchase of property plant and equipment": -30,
                "Payments for purchase of property plant and equipment": -31,
            },
        }])
        out = normalize_jquants_details(raw).iloc[0]
        self.assertTrue(pd.isna(out["capex_cumulative"]))
        self.assertIn(
            out["capex_parse_status"],
            {"AMBIGUOUS_EXACT", "AMBIGUOUS_FUZZY"},
        )

    def test_revision_only_changes_state_from_revision_date_forward(self):
        base = pd.DataFrame([
            self._summary_row(
                date="2026-05-10",
                disc_no="D1",
                period="FY",
                fy_end="2026-03-31",
                per_end="2026-03-31",
                sales=1000,
                op=100,
                np=60,
                cfo=120,
                cfi=-50,
            ),
            self._summary_row(
                date="2026-05-20",
                disc_no="D2",
                period="FY",
                fy_end="2026-03-31",
                per_end="2026-03-31",
                sales=1010,
                op=105,
                np=62,
                cfo=125,
                cfi=-50,
            ),
        ])
        summary = normalize_jquants_summary(base)
        disclosures = merge_summary_and_details(
            summary,
            pd.DataFrame(),
        )
        events = build_point_in_time_financial_events(disclosures)
        self.assertEqual(len(events), 2)
        self.assertEqual(events.iloc[0]["available_date"], pd.Timestamp("2026-05-10"))
        self.assertEqual(events.iloc[1]["available_date"], pd.Timestamp("2026-05-20"))

    def test_summary_only_fcf_proxy_uses_cfo_plus_negative_cfi(self):
        rows = []
        # Previous FY / H1, then current H1: enough to build TTM cash flow.
        rows.append(self._summary_row(
            date="2025-05-10",
            disc_no="P-FY",
            period="FY",
            fy_end="2025-03-31",
            per_end="2025-03-31",
            sales=1000,
            op=100,
            np=60,
            cfo=140,
            cfi=-60,
        ))
        rows.append(self._summary_row(
            date="2024-11-10",
            disc_no="P-H1",
            period="2Q",
            fy_end="2025-03-31",
            per_end="2024-09-30",
            sales=480,
            op=45,
            np=25,
            cfo=60,
            cfi=-25,
        ))
        rows.append(self._summary_row(
            date="2025-11-10",
            disc_no="C-H1",
            period="2Q",
            fy_end="2026-03-31",
            per_end="2025-09-30",
            sales=560,
            op=60,
            np=35,
            cfo=80,
            cfi=-30,
        ))

        summary = normalize_jquants_summary(pd.DataFrame(rows))
        events = build_point_in_time_financial_events(
            merge_summary_and_details(summary, pd.DataFrame())
        )
        latest = events.iloc[-1]

        # CFO TTM = current H1 80 + (previous FY 140 - previous H1 60) = 160
        # CFI TTM = -30 + (-60 - -25) = -65
        # Proxy FCF = 95
        self.assertAlmostEqual(float(latest["cfo_ttm"]), 160.0)
        self.assertAlmostEqual(float(latest["cfi_ttm"]), -65.0)
        self.assertAlmostEqual(float(latest["fcf_ttm"]), 95.0)
        self.assertEqual(latest["fcf_basis"], "CFO_PLUS_CFI_PROXY")
        self.assertFalse(bool(latest["fcf_exact"]))

    def test_exact_capex_overrides_proxy(self):
        rows = [
            self._summary_row(
                date="2025-05-10",
                disc_no="FY",
                period="FY",
                fy_end="2025-03-31",
                per_end="2025-03-31",
                sales=1000,
                op=100,
                np=60,
                cfo=140,
                cfi=-60,
            )
        ]
        summary = normalize_jquants_summary(pd.DataFrame(rows))
        details = pd.DataFrame([{
            "ticker": "1111",
            "disclosure_number": "FY",
            "capex_cumulative": 50.0,
            "capex_source_key": "Purchase of property plant and equipment",
            "capex_parse_status": "EXACT",
            "depreciation_cumulative": 20.0,
            "depreciation_source_key": "Depreciation",
            "depreciation_parse_status": "EXACT",
            "fs_item_count": 2,
        }])
        disclosures = merge_summary_and_details(summary, details)
        events = build_point_in_time_financial_events(disclosures)
        latest = events.iloc[-1]
        self.assertAlmostEqual(float(latest["fcf_ttm"]), 90.0)
        self.assertEqual(latest["fcf_basis"], "CFO_MINUS_CAPEX")
        self.assertTrue(bool(latest["fcf_exact"]))

    def test_growth_features_use_only_earlier_snapshots(self):
        events = pd.DataFrame([
            {
                "ticker": "1111",
                "available_datetime": "2023-05-10 15:00",
                "available_date": "2023-05-10",
                "fcf_per_share_ttm": 1.0,
                "cfo_ttm": 100,
                "shares_outstanding_pti": 100,
                "revenue_ttm": 1000,
                "operating_income_ttm": 100,
                "fcf_ttm": 100,
            },
            {
                "ticker": "1111",
                "available_datetime": "2024-05-10 15:00",
                "available_date": "2024-05-10",
                "fcf_per_share_ttm": 1.2,
                "cfo_ttm": 120,
                "shares_outstanding_pti": 100,
                "revenue_ttm": 1100,
                "operating_income_ttm": 120,
                "fcf_ttm": 120,
            },
            {
                "ticker": "1111",
                "available_datetime": "2026-05-10 15:00",
                "available_date": "2026-05-10",
                "fcf_per_share_ttm": 1.8,
                "cfo_ttm": 180,
                "shares_outstanding_pti": 98,
                "revenue_ttm": 1500,
                "operating_income_ttm": 180,
                "fcf_ttm": 176.4,
            },
        ])
        out = add_financial_event_growth_features(events)
        latest = out.iloc[-1]

        # 1Y anniversary has no suitable snapshot within the staleness guard,
        # so it must stay unavailable instead of guessing.
        self.assertTrue(pd.isna(latest["fcf_per_share_yoy"]))

        # 3Y comparison can use the 2023 snapshot.
        self.assertFalse(pd.isna(latest["fcf_per_share_3y_cagr"]))

    def test_market_cap_keeps_jquants_native_million_yen_unit(self):
        bars = pd.DataFrame([{
            "Date": "2026-01-05",
            "Code": "11110",
            "AdjC": 1000,
            "AdjVo": 5000,
        }])
        valuation = pd.DataFrame([{
            "Date": "2026-01-05",
            "Code": "11110",
            "MktCap": 123456.0,
            "PER": 12.3,
        }])
        out = normalize_jquants_daily_market(bars, valuation)
        self.assertEqual(float(out.iloc[0]["market_cap_pti"]), 123456.0)


if __name__ == "__main__":
    unittest.main()


def test_fetch_bundle_can_skip_market_endpoints():
    from jquants_mex_adapter import fetch_ticker_bundle

    class FakeClient:
        def fin_summary(self, *, code="", date=""):
            return pd.DataFrame()
        def fin_details(self, *, code="", date=""):
            return pd.DataFrame()
        def daily_bars(self, **kwargs):
            raise AssertionError("daily_bars should not be called")
        def valuation(self, **kwargs):
            raise AssertionError("valuation should not be called")
        def topix(self, **kwargs):
            raise AssertionError("topix should not be called")

    out = fetch_ticker_bundle(
        FakeClient(),
        code="1111",
        from_date="2026-01-01",
        to_date="2026-10-08",
        include_details=False,
        include_topix=False,
        include_market=False,
    )

    assert out["market"].empty
    assert out["financial_events"].empty

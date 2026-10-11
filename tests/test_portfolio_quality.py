import unittest

import pandas as pd

from portfolio_quality import (
    build_company_news_query,
    display_portfolio_dataframe,
    find_duplicate_tickers,
    is_relevant_company_news,
    normalize_metric_text,
    normalize_portfolio_dataframe,
    normalize_ticker_code,
)


class PortfolioQualityTests(unittest.TestCase):
    def test_metric_normalization_removes_display_units(self):
        self.assertEqual(normalize_metric_text("38.7%"), "38.7")
        self.assertEqual(normalize_metric_text("7.59％"), "7.59")
        self.assertEqual(normalize_metric_text("15.5倍"), "15.5")
        self.assertEqual(normalize_metric_text("1.33x"), "1.33")
        self.assertEqual(normalize_metric_text(None), "")

    def test_portfolio_dataframe_normalizes_numeric_fields(self):
        frame = pd.DataFrame(
            [{
                "ティッカー": "472a ",
                "売上5y CAGR": "15.0%",
                "ROIC": "7.20%",
                "PER": "12.0倍",
                "DPUP": "0.79x",
            }]
        )

        out = normalize_portfolio_dataframe(frame)

        self.assertEqual(out.iloc[0]["ティッカー"], "472A")
        self.assertEqual(out.iloc[0]["売上5y CAGR"], "15")
        self.assertEqual(out.iloc[0]["ROIC"], "7.2")
        self.assertEqual(out.iloc[0]["PER"], "12")
        self.assertEqual(out.iloc[0]["DPUP"], "0.79")

    def test_display_dataframe_formats_percent_and_missing_values(self):
        frame = pd.DataFrame(
            [{
                "ティッカー": "9560",
                "売上5y CAGR": "36.6",
                "ROIC": "47.80",
                "PER": "13.1",
                "DPUP": "4.24",
                "材料メモ": None,
            }]
        )

        out = display_portfolio_dataframe(frame)

        self.assertEqual(out.iloc[0]["売上5y CAGR"], "36.6%")
        self.assertEqual(out.iloc[0]["ROIC"], "47.8%")
        self.assertEqual(out.iloc[0]["PER"], "13.1")
        self.assertEqual(out.iloc[0]["DPUP"], "4.24")
        self.assertEqual(out.iloc[0]["材料メモ"], "—")

    def test_duplicate_ticker_detection_normalizes_codes(self):
        frame = pd.DataFrame(
            {
                "ティッカー": ["9560", "472a", "472A ", "09560", ""],
            }
        )

        self.assertEqual(find_duplicate_tickers(frame), ["472A"])

    def test_company_news_query_contains_name_ticker_and_finance_context(self):
        query = build_company_news_query("4019", "スタメン")
        self.assertIn('"スタメン"', query)
        self.assertIn("4019", query)
        self.assertIn("決算", query)
        self.assertIn("IR", query)

    def test_irrelevant_sports_story_is_rejected_for_stamen(self):
        self.assertFalse(
            is_relevant_company_news(
                "マンチェスター・ユナイテッド対トッテナム、予想スタメン＆フォーメーション",
                "4019",
                "スタメン",
                "ABEMA",
            )
        )

    def test_relevant_company_story_is_kept_for_stamen(self):
        self.assertTrue(
            is_relevant_company_news(
                "スタメン、通期業績予想を上方修正　営業利益が拡大",
                "4019",
                "スタメン",
                "日本経済新聞",
            )
        )

    def test_ticker_hit_can_disambiguate_generic_company_name(self):
        self.assertTrue(
            is_relevant_company_news(
                "5243 NOTE が新サービスを発表",
                "5243",
                "NOTE",
                "Tech News",
            )
        )

    def test_ticker_normalization_preserves_alphanumeric_tse_codes(self):
        self.assertEqual(normalize_ticker_code("472a"), "472A")
        self.assertEqual(normalize_ticker_code("9560.0"), "9560")


if __name__ == "__main__":
    unittest.main()

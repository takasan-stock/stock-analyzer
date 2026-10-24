import datetime
import unittest

from stock_action_status import evaluate_stock_action_status


class TestStockActionStatus(unittest.TestCase):
    def test_earnings_event_overrides_entry_ready(self):
        result = evaluate_stock_action_status(
            portfolio_row={"売上5y CAGR": "18%", "ステータス": "監視中"},
            me_row={
                "fcf_engine_score": 72,
                "fcf_engine_coverage": 0.9,
                "fcf_quality_grade": "A",
                "mex_score": 82,
                "sw_decision": "READY",
                "second_wave_state": "RE-WATCH READY",
                "rs_rating": 88,
            },
            short_row={"alert_tier": "🚨 A+ 最優先確認", "alert_score": 90},
            live_entry={"status": "ENTRY READY"},
            earnings_date=datetime.date(2026, 10, 25),
            today=datetime.date(2026, 10, 24),
        )
        self.assertEqual(result.stage, "EVENT_CHECK")
        self.assertEqual(result.label, "決算イベント確認")
        self.assertIn("決算まで1日", result.blockers)

    def test_entry_ready_goes_to_pretrade(self):
        result = evaluate_stock_action_status(
            portfolio_row={"売上5y CAGR": "16%", "ステータス": "監視中"},
            me_row={
                "fcf_engine_score": 70,
                "fcf_engine_coverage": 0.8,
                "fcf_quality_grade": "B",
                "mex_score": 78,
                "sw_decision": "READY",
                "second_wave_state": "RE-WATCH READY",
                "rs_rating": 82,
            },
            short_row={"alert_tier": "🔥 A 優先確認", "alert_score": 80},
            live_entry={"status": "ENTRY READY"},
            earnings_date=datetime.date(2026, 11, 20),
            today=datetime.date(2026, 10, 24),
        )
        self.assertEqual(result.stage, "PRETRADE")
        self.assertEqual(result.next_page, "pages/9_Pre_Trade_Check.py")
        self.assertGreaterEqual(result.readiness_score, 60)

    def test_me_ready_goes_to_entry_monitor(self):
        result = evaluate_stock_action_status(
            portfolio_row={"売上5y CAGR": "12%", "ステータス": "監視中"},
            me_row={
                "fcf_engine_score": 60,
                "fcf_engine_coverage": 0.7,
                "fcf_quality_grade": "C",
                "mex_score": 74,
                "sw_decision": "READY",
                "second_wave_state": "RE-WATCH READY",
                "rs_rating": 72,
            },
            short_row={},
            live_entry={},
            earnings_date=datetime.date(2026, 12, 1),
            today=datetime.date(2026, 10, 24),
        )
        self.assertEqual(result.stage, "ENTRY_MONITOR")
        self.assertEqual(result.next_page, "pages/8_Short_Cover_Hunter.py")

    def test_sparse_data_returns_data_check(self):
        result = evaluate_stock_action_status(
            portfolio_row={"ステータス": "監視中"},
            me_row={},
            short_row={},
            live_entry={},
            earnings_date=None,
            today=datetime.date(2026, 10, 24),
        )
        self.assertEqual(result.stage, "DATA_CHECK")
        self.assertEqual(result.confidence, "LOW")
        self.assertIn("売上成長率が未登録", result.blockers)


if __name__ == "__main__":
    unittest.main()

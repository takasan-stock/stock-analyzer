import datetime
import unittest

from stock_priority_board import build_daily_priority_board


class TestStockPriorityBoard(unittest.TestCase):
    def test_orders_by_stage_then_readiness(self):
        rows = build_daily_priority_board(
            portfolio_rows=[
                {"ティッカー": "1001", "銘柄名": "A", "ステータス": "監視中", "売上5y CAGR": "12%"},
                {"ティッカー": "1002", "銘柄名": "B", "ステータス": "監視中", "売上5y CAGR": "18%"},
                {"ティッカー": "1003", "銘柄名": "C", "ステータス": "監視中", "売上5y CAGR": "8%"},
            ],
            me_rows=[
                {"ticker": "1001", "mex_score": 74, "sw_decision": "READY", "fcf_engine_score": 60, "fcf_engine_coverage": 0.7, "fcf_quality_grade": "C"},
                {"ticker": "1002", "mex_score": 82, "sw_decision": "READY", "fcf_engine_score": 75, "fcf_engine_coverage": 0.9, "fcf_quality_grade": "A"},
                {"ticker": "1003", "mex_score": 55, "sw_decision": "WATCH", "fcf_engine_score": 55, "fcf_engine_coverage": 0.6, "fcf_quality_grade": "C"},
            ],
            live_entry_rows=[
                {"ticker": "1002", "status": "ENTRY READY"},
            ],
            today=datetime.date(2026, 10, 24),
        )
        self.assertEqual(rows[0]["ticker"], "1002")
        self.assertEqual(rows[0]["stage"], "PRETRADE")
        self.assertEqual(rows[1]["ticker"], "1001")
        self.assertEqual(rows[1]["stage"], "ENTRY_MONITOR")
        self.assertEqual(rows[2]["stage"], "MONITOR")

    def test_excludes_passed_by_default(self):
        rows = build_daily_priority_board(
            portfolio_rows=[
                {"ティッカー": "2001", "銘柄名": "A", "ステータス": "見送り"},
                {"ティッカー": "2002", "銘柄名": "B", "ステータス": "監視中"},
            ]
        )
        self.assertEqual([row["ticker"] for row in rows], ["2002"])

    def test_uses_latest_short_cover_row(self):
        rows = build_daily_priority_board(
            portfolio_rows=[
                {"ティッカー": "3001", "銘柄名": "A", "ステータス": "監視中", "売上5y CAGR": "10%"},
            ],
            me_rows=[
                {"ticker": "3001", "mex_score": 50, "sw_decision": "WATCH"},
            ],
            short_rows=[
                {"ticker": "3001", "alert_date": "2026-10-20", "alert_tier": "🟡 B 監視", "alert_score": 60},
                {"ticker": "3001", "alert_date": "2026-10-23", "alert_tier": "🚨 A+ 最優先確認", "alert_score": 90},
            ],
        )
        self.assertEqual(rows[0]["entry_status"], "🚨 A+ 最優先確認")


if __name__ == "__main__":
    unittest.main()

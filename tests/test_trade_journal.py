import unittest

import pandas as pd

from trade_journal import (
    JOURNAL_COLUMNS,
    build_group_summary,
    build_summary,
    calc_trade_outcome,
    normalize_journal,
    score_bucket,
    upsert_journal_record,
)


class TradeJournalTests(unittest.TestCase):
    def test_calc_trade_outcome_profit_and_r(self):
        result = calc_trade_outcome(
            planned_entry=1000,
            actual_entry=1010,
            actual_stop=980,
            shares=200,
            actual_exit=1070,
            fees=500,
            entry_date="2026-09-01",
            exit_date="2026-09-05",
        )
        self.assertAlmostEqual(result["entry_slippage_pct"], 1.0)
        self.assertAlmostEqual(result["risk_per_share"], 30.0)
        self.assertAlmostEqual(result["risk_amount"], 6000.0)
        self.assertAlmostEqual(result["pnl"], 11500.0)
        self.assertAlmostEqual(result["r_multiple"], 11500.0 / 6000.0)
        self.assertEqual(result["holding_days"], 4)

    def test_calc_trade_outcome_requires_valid_stop_for_r(self):
        result = calc_trade_outcome(
            planned_entry=1000,
            actual_entry=1000,
            actual_stop=1010,
            shares=100,
            actual_exit=1050,
        )
        self.assertIsNone(result["risk_amount"])
        self.assertIsNone(result["r_multiple"])
        self.assertAlmostEqual(result["pnl"], 5000.0)

    def test_upsert_uses_plan_id(self):
        empty = pd.DataFrame(columns=JOURNAL_COLUMNS)
        first = upsert_journal_record(empty, {
            "journal_id": "J-1",
            "plan_id": "P-1",
            "status": "OPEN",
            "ticker": "4063",
        })
        second = upsert_journal_record(first, {
            "journal_id": "J-1",
            "plan_id": "P-1",
            "status": "CLOSED",
            "ticker": "4063",
            "pnl": 10000,
            "r_multiple": 2.0,
        })
        self.assertEqual(len(second), 1)
        self.assertEqual(second.iloc[0]["status"], "CLOSED")
        self.assertEqual(float(second.iloc[0]["pnl"]), 10000.0)

    def test_summary_and_grouping(self):
        rows = pd.DataFrame([
            {
                "journal_id": "J1", "plan_id": "P1", "status": "CLOSED",
                "ticker": "1111", "pnl": 10000, "r_multiple": 2.0,
                "return_pct": 5.0, "holding_days": 3,
                "total_score": 88, "stage": "🟢 Stage 2",
                "hunter_status": "🟢 ENTRY READY",
                "source": "Entry Hunter",
                "followed_plan": "はい",
            },
            {
                "journal_id": "J2", "plan_id": "P2", "status": "CLOSED",
                "ticker": "2222", "pnl": -5000, "r_multiple": -1.0,
                "return_pct": -2.5, "holding_days": 2,
                "total_score": 78, "stage": "🟢 Stage 2",
                "hunter_status": "🟡 WAIT",
                "source": "今日見るべき銘柄 Aランク",
                "followed_plan": "いいえ",
            },
        ])
        df = normalize_journal(rows)
        summary = build_summary(df)
        self.assertEqual(summary["trades"], 2)
        self.assertEqual(summary["wins"], 1)
        self.assertAlmostEqual(summary["win_rate"], 50.0)
        self.assertAlmostEqual(summary["total_pnl"], 5000.0)
        self.assertAlmostEqual(summary["avg_r"], 0.5)
        self.assertAlmostEqual(summary["plan_follow_rate"], 50.0)

        grouped = build_group_summary(df, "score_bucket")
        self.assertEqual(set(grouped["score_bucket"]), {"85-100", "75-84"})

    def test_score_bucket(self):
        self.assertEqual(score_bucket(90), "85-100")
        self.assertEqual(score_bucket(80), "75-84")
        self.assertEqual(score_bucket(70), "65-74")
        self.assertEqual(score_bucket(55), "50-64")
        self.assertEqual(score_bucket(40), "<50")


if __name__ == "__main__":
    unittest.main()

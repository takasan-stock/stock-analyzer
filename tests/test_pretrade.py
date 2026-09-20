import unittest

import pandas as pd

from pretrade import (
    build_technical_snapshot,
    calc_position_size,
    calc_risk_reward,
    evaluate_pretrade,
)


class PreTradeTests(unittest.TestCase):
    def make_daily(self, start=1000.0, days=260, daily_step=2.0, volume=1_000_000):
        idx = pd.bdate_range("2025-09-01", periods=days)
        close = pd.Series([start + i * daily_step for i in range(days)], index=idx)
        df = pd.DataFrame({
            "Open": close * 0.998,
            "High": close * 1.01,
            "Low": close * 0.99,
            "Close": close,
            "Volume": volume,
        }, index=idx)
        return df

    def test_risk_reward(self):
        out = calc_risk_reward(2000, 1950, 2100)
        self.assertAlmostEqual(out["rr"], 2.0)
        self.assertAlmostEqual(out["risk_per_share"], 50.0)

    def test_position_size_rounds_to_lot(self):
        out = calc_position_size(1_000_000, 1.0, 2000, 1950, lot_size=100)
        self.assertEqual(out["shares"], 200)
        self.assertAlmostEqual(out["actual_max_loss"], 10_000)

    def test_snapshot_detects_stage2(self):
        df = self.make_daily()
        bench = self.make_daily(start=1000, daily_step=0.5)
        snap = build_technical_snapshot(df, bench)
        self.assertTrue(snap["data_ok"])
        self.assertTrue(snap["stage2"])
        self.assertGreater(snap["price"], snap["sma50"])
        self.assertGreaterEqual(snap["rs_proxy"], 50)

    def test_pretrade_blocks_low_rr(self):
        df = self.make_daily()
        snap = build_technical_snapshot(df, self.make_daily(daily_step=0.5))
        price = snap["price"]
        result = evaluate_pretrade(
            snap,
            entry=price,
            stop=price * 0.97,
            target=price * 1.03,
            capital=1_000_000,
            risk_percent=1.0,
            earnings_days=20,
            entry_hunter={"status": "🟢 ENTRY READY", "score": 80},
            pattern_flags={"VCP": True},
        )
        self.assertTrue(result["blocked"])
        self.assertIn("RR 1:1.5未満", result["block_reasons"])

    def test_pretrade_candidate_when_conditions_are_good(self):
        df = self.make_daily()
        snap = build_technical_snapshot(df, self.make_daily(daily_step=0.5))
        price = snap["price"]
        result = evaluate_pretrade(
            snap,
            entry=price,
            stop=price * 0.97,
            target=price * 1.09,
            capital=1_000_000,
            risk_percent=1.0,
            earnings_days=20,
            entry_hunter={"status": "🟢 ENTRY READY", "score": 85},
            pattern_flags={"VCP": True, "PP": True},
        )
        self.assertFalse(result["blocked"])
        self.assertGreaterEqual(result["total_score"], 65)
        self.assertIn(result["verdict"], {
            "🟢 TRADE READY", "🟢 ENTRY CANDIDATE", "🟡 WAIT"
        })


if __name__ == "__main__":
    unittest.main()

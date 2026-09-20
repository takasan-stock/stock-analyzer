import unittest

import pandas as pd

from calibration_backtest import (
    BASE_WEIGHTS,
    build_calibration_backtest,
    rescore_components,
)
from trade_journal import normalize_journal


def make_trade(i, *, setup, entry, risk, r, followed="はい", rule="なし", date=None):
    return {
        "journal_id": f"J{i}",
        "plan_id": f"P{i}",
        "status": "CLOSED",
        "ticker": str(1000 + i),
        "name": f"T{i}",
        "updated_at": str(date or pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)),
        "plan_confirmed_at": str(date or pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)),
        "actual_entry_date": str(date or pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)),
        "setup_score": setup,
        "entry_score": entry,
        "risk_score": risk,
        "total_score": setup + entry + risk,
        "r_multiple": r,
        "pnl": r * 5000,
        "return_pct": r * 2,
        "followed_plan": followed,
        "rule_break": rule,
    }


class CalibrationBacktestTests(unittest.TestCase):
    def test_rescore_components_base_weights(self):
        df = pd.DataFrame([{"setup_score": 40, "entry_score": 35, "risk_score": 25}])
        score = rescore_components(df, BASE_WEIGHTS)
        self.assertAlmostEqual(float(score.iloc[0]), 100.0)

    def test_backtest_requires_minimum_sample(self):
        rows = [
            make_trade(i, setup=20+i, entry=20, risk=20, r=i/10)
            for i in range(10)
        ]
        report = build_calibration_backtest(normalize_journal(pd.DataFrame(rows)))
        self.assertEqual(report["status"], "🟡 DATA BUILDING")
        self.assertEqual(report["total"], 10)

    def test_chronological_split_and_validation(self):
        rows = []
        for i in range(30):
            rows.append(make_trade(
                i,
                setup=10 + i,
                entry=20 + i * 0.2,
                risk=18,
                r=-1 + i * 0.12,
            ))
        report = build_calibration_backtest(
            normalize_journal(pd.DataFrame(rows)),
            train_fraction=0.60,
        )
        self.assertEqual(report["train_n"], 18)
        self.assertEqual(report["validation_n"], 12)
        self.assertFalse(report["comparison"].empty)
        self.assertAlmostEqual(sum(report["weights"].values()), 100.0, places=1)

    def test_validation_scores_are_bounded(self):
        rows = [
            make_trade(i, setup=min(40, 15+i), entry=min(35, 18+i/2), risk=20, r=-1+i*0.1)
            for i in range(24)
        ]
        report = build_calibration_backtest(normalize_journal(pd.DataFrame(rows)))
        validation = report["validation"]
        self.assertTrue((validation["current_score_bt"] <= 100).all())
        self.assertTrue((validation["experimental_score_bt"] <= 100).all())


if __name__ == "__main__":
    unittest.main()

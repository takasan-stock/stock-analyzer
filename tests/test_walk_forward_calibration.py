import unittest

import pandas as pd

from trade_journal import normalize_journal
from walk_forward_calibration import build_walk_forward_report


def make_trade(i, *, setup, entry, risk, r, followed="はい", rule="なし"):
    dt = pd.Timestamp("2026-01-01") + pd.Timedelta(days=i)
    return {
        "journal_id": f"J{i}",
        "plan_id": f"P{i}",
        "status": "CLOSED",
        "ticker": str(1000 + i),
        "name": f"T{i}",
        "updated_at": str(dt),
        "plan_confirmed_at": str(dt),
        "actual_entry_date": str(dt),
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


class WalkForwardCalibrationTests(unittest.TestCase):
    def test_requires_initial_train_plus_validation(self):
        rows = [
            make_trade(i, setup=20+i/2, entry=20, risk=20, r=-1+i*0.1)
            for i in range(24)
        ]
        report = build_walk_forward_report(
            normalize_journal(pd.DataFrame(rows)),
            initial_train=20,
            validation_window=10,
        )
        self.assertEqual(report["status"], "🟡 DATA BUILDING")
        self.assertEqual(len(report["folds"]), 0)

    def test_expanding_windows_do_not_use_future_data(self):
        rows = [
            make_trade(i, setup=10+i, entry=18+i*0.2, risk=20, r=-1+i*0.08)
            for i in range(50)
        ]
        report = build_walk_forward_report(
            normalize_journal(pd.DataFrame(rows)),
            initial_train=20,
            validation_window=10,
            step=10,
        )
        self.assertEqual(len(report["folds"]), 3)
        self.assertEqual(report["folds"][0]["train_n"], 20)
        self.assertEqual(report["folds"][1]["train_n"], 30)
        self.assertEqual(report["folds"][2]["train_n"], 40)
        self.assertEqual(report["validation_total"], 30)

        self.assertLess(
            pd.Timestamp(report["folds"][0]["train_end"]),
            pd.Timestamp(report["folds"][0]["validation_start"]),
        )

    def test_weights_sum_to_100_each_fold(self):
        rows = [
            make_trade(i, setup=min(40, 12+i), entry=min(35, 15+i/2), risk=18, r=-1+i*0.08)
            for i in range(50)
        ]
        report = build_walk_forward_report(
            normalize_journal(pd.DataFrame(rows)),
            initial_train=20,
            validation_window=10,
            step=10,
        )
        for fold in report["folds"]:
            self.assertAlmostEqual(sum(fold["weights"].values()), 100.0, places=1)

    def test_oos_scores_are_bounded(self):
        rows = [
            make_trade(i, setup=min(40, 10+i), entry=min(35, 18+i/3), risk=20, r=-1+i*0.09)
            for i in range(50)
        ]
        report = build_walk_forward_report(
            normalize_journal(pd.DataFrame(rows)),
            initial_train=20,
            validation_window=10,
            step=10,
        )
        oos = report["oos_validation"]
        self.assertTrue((oos["current_score_wf"] <= 100).all())
        self.assertTrue((oos["experimental_score_wf"] <= 100).all())

    def test_clean_sample_preferred_when_enough(self):
        rows = [
            make_trade(i, setup=20+i/2, entry=20+i/4, risk=18, r=-1+i*0.12)
            for i in range(35)
        ]
        rows += [
            make_trade(100+i, setup=40, entry=35, risk=25, r=-3,
                       followed="いいえ", rule="追いかけ買い")
            for i in range(8)
        ]
        report = build_walk_forward_report(
            normalize_journal(pd.DataFrame(rows)),
            initial_train=20,
            validation_window=5,
            step=5,
            prefer_clean=True,
        )
        self.assertEqual(report["sample_mode"], "CLEAN")


if __name__ == "__main__":
    unittest.main()

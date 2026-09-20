import unittest

import pandas as pd

from score_calibration import (
    BASE_WEIGHTS,
    build_calibration_report,
    calibration_sample,
    component_analysis,
)
from trade_journal import normalize_journal


def make_trade(i, *, score, setup, entry, risk, r, followed="はい", rule="なし"):
    return {
        "journal_id": f"J{i}",
        "plan_id": f"P{i}",
        "status": "CLOSED",
        "ticker": f"{1000+i}",
        "name": f"T{i}",
        "total_score": score,
        "setup_score": setup,
        "entry_score": entry,
        "risk_score": risk,
        "r_multiple": r,
        "pnl": r * 5000,
        "return_pct": r * 2,
        "followed_plan": followed,
        "rule_break": rule,
        "stage": "🟢 Stage 2",
        "rs_proxy": 50 + i,
        "volume_ratio": 1.0 + i / 20,
        "hunter_score": min(100, 50 + i),
        "planned_rr": 2.0 + i / 50,
        "ema20_gap_pct": 2.0,
        "entry_slippage_pct": 0.1,
    }


class ScoreCalibrationTests(unittest.TestCase):
    def test_learning_keeps_base_weights_under_20(self):
        df = normalize_journal(pd.DataFrame([
            make_trade(i, score=70+i, setup=28+i/3, entry=24+i/4, risk=18, r=-0.5+i*0.15)
            for i in range(12)
        ]))
        report = build_calibration_report(df)
        self.assertEqual(report["status"], "🟡 LEARNING")
        self.assertEqual(report["proposed_weights"], BASE_WEIGHTS)

    def test_experimental_weights_after_20(self):
        rows = []
        for i in range(25):
            rows.append(make_trade(
                i,
                score=65+i,
                setup=20+i*0.7,
                entry=18+i*0.5,
                risk=20-i*0.05,
                r=-1.0+i*0.15,
            ))
        report = build_calibration_report(normalize_journal(pd.DataFrame(rows)))
        self.assertEqual(report["status"], "🟠 EXPERIMENTAL")
        proposed = report["proposed_weights"]
        self.assertAlmostEqual(sum(proposed.values()), 100.0, places=1)
        self.assertGreater(proposed["Setup Quality"], 34)
        self.assertTrue(report["weights"]["eligible"].all())

    def test_clean_sample_excludes_rule_breaks_when_enough_clean(self):
        rows = [
            make_trade(i, score=70+i, setup=25+i, entry=20+i/2, risk=18, r=i/10)
            for i in range(12)
        ]
        rows += [
            make_trade(100+i, score=95, setup=40, entry=35, risk=25, r=-3,
                       followed="いいえ", rule="追いかけ買い")
            for i in range(5)
        ]
        sample, mode = calibration_sample(normalize_journal(pd.DataFrame(rows)))
        self.assertEqual(mode, "CLEAN")
        self.assertEqual(len(sample), 12)

    def test_component_analysis_detects_positive_setup_relation(self):
        rows = [
            make_trade(i, score=60+i, setup=10+i, entry=20, risk=20, r=-2+i*0.2)
            for i in range(20)
        ]
        out = component_analysis(normalize_journal(pd.DataFrame(rows)))
        setup = out[out["component"] == "Setup Quality"].iloc[0]
        self.assertGreater(float(setup["spearman_r"]), 0.9)


if __name__ == "__main__":
    unittest.main()

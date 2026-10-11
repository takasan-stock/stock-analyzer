import unittest

import pandas as pd

from mex_backtest import (
    build_mex_event_study,
    mex_bucket,
    monotonicity_report,
    summarize_mex_event_study,
)


class MexBacktestTests(unittest.TestCase):
    def test_bucket_boundaries(self):
        self.assertEqual(mex_bucket(95), "90-100")
        self.assertEqual(mex_bucket(85), "80-89")
        self.assertEqual(mex_bucket(75), "70-79")
        self.assertEqual(mex_bucket(65), "60-69")
        self.assertEqual(mex_bucket(59.9), "<60")
        self.assertEqual(mex_bucket(None), "UNSCORED")

    def test_event_study_uses_trading_row_offsets(self):
        dates = pd.bdate_range("2026-01-05", periods=30)
        df = pd.DataFrame({
            "ticker": ["1111"] * len(dates),
            "trade_date": dates,
            "adj_close": [100 + i for i in range(len(dates))],
            "topix_close": [200 + i * 0.5 for i in range(len(dates))],
            "ignition_event": [False] * len(dates),
            "mex_score": [80.0] * len(dates),
            "fcf_engine_score": [85.0] * len(dates),
            "mlp_c_core": [0.9] * len(dates),
            "mlp_z": [-0.5] * len(dates),
            "mlp_velocity_20": [0.02] * len(dates),
            "mlp_acceleration": [0.01] * len(dates),
            "normalization_gap": [0.1] * len(dates),
        })
        df.loc[0, "ignition_event"] = True

        out = build_mex_event_study(df, horizons=(20,))
        self.assertEqual(len(out), 1)
        expected = (120 / 100) - 1
        self.assertAlmostEqual(out.iloc[0]["ret_20d"], expected)

    def test_summary_and_monotonicity(self):
        events = pd.DataFrame([
            {"mex_bucket": "90-100", "ret_120d": 0.30, "excess_120d": 0.20,
             "mfe_120d": 0.4, "mae_120d": -0.05, "max_dd_120d": -0.08},
            {"mex_bucket": "80-89", "ret_120d": 0.20, "excess_120d": 0.12,
             "mfe_120d": 0.3, "mae_120d": -0.08, "max_dd_120d": -0.10},
            {"mex_bucket": "70-79", "ret_120d": 0.10, "excess_120d": 0.05,
             "mfe_120d": 0.2, "mae_120d": -0.10, "max_dd_120d": -0.12},
            {"mex_bucket": "60-69", "ret_120d": 0.05, "excess_120d": 0.01,
             "mfe_120d": 0.1, "mae_120d": -0.12, "max_dd_120d": -0.15},
        ])
        summary = summarize_mex_event_study(events, horizons=(120,))
        report = monotonicity_report(summary, horizon=120)
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(report["ordered_pairs"], report["pairs"])

    def test_incomplete_horizon_stays_nan(self):
        dates = pd.bdate_range("2026-01-05", periods=5)
        df = pd.DataFrame({
            "ticker": ["1111"] * len(dates),
            "trade_date": dates,
            "adj_close": [100, 101, 102, 103, 104],
            "ignition_event": [True, False, False, False, False],
            "mex_score": [90] * len(dates),
        })
        out = build_mex_event_study(df, horizons=(20,))
        self.assertTrue(pd.isna(out.iloc[0]["ret_20d"]))


if __name__ == "__main__":
    unittest.main()

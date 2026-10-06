import unittest

import pandas as pd

from multiple_expansion import (
    build_point_in_time_financials,
    calculate_fcf_engine,
    calculate_mex_components,
    calculate_valuation_multiples,
    classify_mex_state,
    confirm_state_transitions,
)


def state_row(**kwargs):
    base = {
        "fcf_engine_score": 80,
        "mlp_c_core": 1.0,
        "mlp_z": 0.0,
        "mlp_velocity_20": 0.01,
        "mlp_velocity_pct_20": 0.01,
        "mlp_acceleration": 0.01,
        "market_confirm_count": 2,
        "valid_multiple_count": 3,
    }
    base.update(kwargs)
    return base


class MultipleExpansionGoldenTests(unittest.TestCase):
    def test_case_a_healthy_early_rerating(self):
        row = state_row(
            fcf_engine_score=88,
            mlp_c_core=0.91,
            mlp_z=-0.60,
            mlp_velocity_20=0.04,
            mlp_acceleration=0.02,
            market_confirm_count=3,
        )
        self.assertEqual(classify_mex_state(row), "IGNITION")

    def test_case_b_coiled_spring(self):
        row = state_row(
            fcf_engine_score=90,
            mlp_c_core=0.84,
            mlp_z=-1.00,
            mlp_velocity_20=-0.01,
            mlp_acceleration=-0.01,
            market_confirm_count=1,
        )
        self.assertEqual(classify_mex_state(row), "SPRING")

    def test_case_c_prep(self):
        row = state_row(
            fcf_engine_score=82,
            mlp_c_core=0.92,
            mlp_z=-0.40,
            mlp_velocity_20=0.02,
            mlp_acceleration=0.0,
            market_confirm_count=1,
        )
        self.assertEqual(classify_mex_state(row), "PREP")

    def test_case_d_expansion(self):
        row = state_row(
            fcf_engine_score=79,
            mlp_c_core=1.08,
            mlp_z=0.40,
            mlp_velocity_20=0.03,
            mlp_acceleration=0.01,
        )
        self.assertEqual(classify_mex_state(row), "EXPANSION")

    def test_case_e_mature(self):
        row = state_row(
            fcf_engine_score=76,
            mlp_c_core=1.18,
            mlp_z=1.10,
            mlp_velocity_20=0.01,
            mlp_acceleration=0.00,
        )
        self.assertEqual(classify_mex_state(row), "MATURE")

    def test_case_f_exhaustion(self):
        row = state_row(
            fcf_engine_score=80,
            mlp_c_core=1.30,
            mlp_z=2.20,
            mlp_velocity_20=0.01,
            mlp_acceleration=-0.01,
        )
        self.assertEqual(classify_mex_state(row), "EXHAUSTION")

    def test_case_g_speculative_expansion(self):
        row = state_row(
            fcf_engine_score=38,
            mlp_c_core=1.22,
            mlp_z=1.2,
            mlp_velocity_20=0.02,
            mlp_acceleration=0.01,
        )
        self.assertEqual(classify_mex_state(row), "SPECULATIVE")

    def test_case_h_negative_fcf_renormalizes_to_two_valid_multiples(self):
        df = pd.DataFrame([{
            "market_cap_pti": 1000.0,
            "fcf_ttm": -10.0,
            "ebitda_ttm": 100.0,
            "net_income_ttm": 50.0,
            "total_debt": 100.0,
            "cash_and_equivalents": 50.0,
        }])
        out = calculate_valuation_multiples(df).iloc[0]
        self.assertTrue(pd.isna(out["p_fcf"]))
        self.assertAlmostEqual(out["ev_ebitda"], 10.5)
        self.assertAlmostEqual(out["per"], 20.0)
        self.assertEqual(int(out["valid_multiple_count"]), 2)

    def test_case_i_only_one_multiple_is_unavailable(self):
        row = state_row(valid_multiple_count=1)
        self.assertEqual(classify_mex_state(row), "UNAVAILABLE")

    def test_case_j_lookahead_protection(self):
        market = pd.DataFrame({
            "ticker": ["1111", "1111"],
            "trade_date": ["2026-05-09", "2026-05-10"],
            "adj_close": [100.0, 101.0],
        })
        financial = pd.DataFrame([{
            "ticker": "1111",
            "fiscal_period_end": "2026-03-31",
            "announcement_date": "2026-05-10",
            "available_date": "2026-05-10",
            "fcf_ttm": 123.0,
        }])
        out = build_point_in_time_financials(financial, market)
        before = out.loc[
            out["trade_date"] == pd.Timestamp("2026-05-09")
        ].iloc[0]
        on_date = out.loc[
            out["trade_date"] == pd.Timestamp("2026-05-10")
        ].iloc[0]
        self.assertTrue(pd.isna(before["fcf_ttm"]))
        self.assertEqual(float(on_date["fcf_ttm"]), 123.0)

    def test_case_k_state_confirmation(self):
        df = pd.DataFrame([
            {"trade_date": "2026-01-01", "ticker": "1111", "state_raw": "WATCH"},
            {"trade_date": "2026-01-02", "ticker": "1111", "state_raw": "WATCH"},
            {"trade_date": "2026-01-03", "ticker": "1111", "state_raw": "IGNITION"},
            {"trade_date": "2026-01-04", "ticker": "1111", "state_raw": "IGNITION"},
        ])
        out = confirm_state_transitions(
            df,
            confirmation_days=2,
            cooldown_days=20,
        )
        self.assertEqual(
            out.iloc[-1]["state_confirmed"],
            "IGNITION",
        )
        self.assertTrue(bool(out.iloc[-1]["ignition_event"]))
        self.assertFalse(bool(out.iloc[-2]["ignition_event"]))

    def test_case_l_cooldown_blocks_repeat_without_reset(self):
        states = (
            ["WATCH", "WATCH", "IGNITION", "IGNITION"]
            + ["EXPANSION"] * 3
            + ["IGNITION", "IGNITION"]
        )
        df = pd.DataFrame({
            "trade_date": pd.date_range(
                "2026-01-01",
                periods=len(states),
            ),
            "ticker": ["1111"] * len(states),
            "state_raw": states,
        })
        out = confirm_state_transitions(
            df,
            confirmation_days=2,
            cooldown_days=20,
        )
        self.assertEqual(int(out["ignition_event"].sum()), 1)

    def test_fcf_engine_requires_60pct_coverage(self):
        df = pd.DataFrame([{
            "fcf_per_share_3y_cagr": 0.20,
            "roic": 0.15,
        }])
        out = calculate_fcf_engine(df).iloc[0]
        self.assertTrue(pd.isna(out["fcf_engine_score"]))
        self.assertLess(
            out["fcf_engine_coverage"],
            0.60,
        )

    def test_mex_score_available_at_60pct_weight(self):
        df = pd.DataFrame([{
            "fcf_engine_score": 85.0,
            "mlp_c_core": 0.90,
            "mlp_z": -0.7,
            "mlp_velocity_pct_20": 0.05,
            "mlp_acceleration": 0.03,
            "normalization_gap": 0.11,
            "rs_rating": 75,
            "rs_rating_20d_ago": 70,
            "volume_ratio": 1.3,
            "fcf_quality_grade": "A",
        }])
        out = calculate_mex_components(df).iloc[0]
        self.assertFalse(pd.isna(out["mex_score"]))
        self.assertGreaterEqual(
            out["mex_coverage"],
            0.60,
        )
        self.assertGreater(out["mex_score"], 70)


if __name__ == "__main__":
    unittest.main()

import unittest

import pandas as pd

from short_cover import (
    activate_condition_version,
    append_condition_version,
    append_priority_alert_history,
    apply_optimizer_condition,
    build_priority_alerts,
    clean_issue_name,
    compare_live_vs_backtest,
    get_active_condition_version,
    normalize_alert_history,
    normalize_condition_versions,
)


def robust_condition():
    return pd.Series({
        "cover_min": 70,
        "ignition_min": 60,
        "long_min": 60,
        "pressure_min": 50,
        "confidence_min": 40,
        "robustness": "🟢 ROBUST",
        "stability_score": 78,
        "test_signals": 12,
        "test_win": 66.7,
        "test_avg": 3.2,
        "test_mfe": 8.5,
        "test_mae": -2.4,
    })


class ShortCoverCoreTests(unittest.TestCase):
    def test_robust_match_requires_all_thresholds(self):
        current = pd.DataFrame([
            {
                "ticker": "1111", "name": "PASS",
                "cover_score": 80, "ignition_score": 75,
                "long_demand_score": 72, "short_pressure": 65,
                "confidence": 80,
            },
            {
                "ticker": "2222", "name": "FAIL",
                "cover_score": 80, "ignition_score": 75,
                "long_demand_score": 72, "short_pressure": 49,
                "confidence": 80,
            },
        ])
        out = apply_optimizer_condition(current, robust_condition())

        passed = out[out["ticker"] == "1111"].iloc[0]
        failed = out[out["ticker"] == "2222"].iloc[0]

        self.assertTrue(bool(passed["optimizer_match"]))
        self.assertEqual(passed["optimizer_label"], "⭐ ROBUST MATCH")
        self.assertGreater(float(passed["match_strength"]), 0)

        self.assertFalse(bool(failed["optimizer_match"]))
        self.assertEqual(failed["optimizer_label"], "")
        self.assertEqual(float(failed["match_strength"]), 0.0)

    def test_priority_alert_rewards_validation_and_fresh_promotion(self):
        current = pd.DataFrame([{
            "ticker": "1111",
            "name": "TEST",
            "phase": "🔥 COVER EARLY",
            "regime": "🔥 COVER + NEW MONEY",
            "cover_score": 82,
            "ignition_score": 78,
            "long_demand_score": 76,
            "short_pressure": 68,
            "confidence": 82,
            "vol_ratio": 2.2,
            "rs_watch": 90,
            "snapshot_date": pd.Timestamp("2026-09-18"),
        }])
        current = apply_optimizer_condition(current, robust_condition())
        promotions = pd.DataFrame([{
            "ticker": "1111",
            "phase_jump": 1,
            "cover_delta": 12,
            "ignition_delta": 18,
            "fresh_breakout": True,
            "fresh_avwap_reclaim": True,
            "promotion_reason": "COVER WATCH → COVER EARLY",
        }])

        alerts = build_priority_alerts(current, promotions=promotions, limit=5)
        self.assertEqual(len(alerts), 1)
        row = alerts.iloc[0]
        self.assertTrue(bool(row["is_promotion"]))
        self.assertGreaterEqual(float(row["alert_score"]), 70)
        self.assertIn(row["alert_tier"], {"🚨 A+ 最優先確認", "🔥 A 優先確認"})
        self.assertIn("ROBUST MATCH", row["alert_reason"])
        self.assertIn("今日昇格", row["alert_reason"])

    def test_first_run_priority_is_provisional_without_validation(self):
        current = pd.DataFrame([{
            "ticker": "3333",
            "name": "FIRST",
            "phase": "✅ COVER CONFIRMED",
            "regime": "🟢 NEW MONEY",
            "cover_score": 82,
            "ignition_score": 78,
            "long_demand_score": 80,
            "short_pressure": 70,
            "confidence": 85,
            "vol_ratio": 2.0,
            "rs_watch": 90,
            "snapshot_date": pd.Timestamp("2026-09-18"),
            "optimizer_label": "",
            "match_strength": 0,
            "condition_text": "",
        }])
        promotions = pd.DataFrame([{
            "ticker": "3333",
            "phase_jump": 1,
            "cover_delta": 10,
            "ignition_delta": 15,
            "fresh_breakout": True,
            "fresh_avwap_reclaim": True,
            "promotion_reason": "昇格",
        }])

        alerts = build_priority_alerts(
            current,
            promotions=promotions,
            limit=5,
            validation_ready=False,
        )
        row = alerts.iloc[0]
        self.assertLess(float(row["alert_score"]), 80)
        self.assertNotEqual(row["alert_tier"], "🚨 A+ 最優先確認")
        self.assertIn("暫定", row["alert_tier"])
        self.assertIn("未検証", row["alert_reason"])

    def test_clean_issue_name_removes_common_stock_suffix(self):
        self.assertEqual(clean_issue_name("ミナトホールディングス　普通株式"), "ミナトホールディングス")
        self.assertEqual(clean_issue_name("ネクセラファーマ 普通株式"), "ネクセラファーマ")
        self.assertEqual(clean_issue_name("Bitcoin Japan"), "Bitcoin Japan")

    def test_alert_history_dedup_keeps_latest_tracking_result(self):
        older = {
            "alert_date": "2026-09-10",
            "ticker": "1111",
            "name": "TEST",
            "alert_tier": "🔥 A 優先確認",
            "alert_score": 75,
            "ret_5d": None,
            "outcome_status": "⏳ 追跡中",
            "last_updated": "2026-09-10 18:00:00",
        }
        newer = {
            **older,
            "ret_5d": 6.5,
            "outcome_status": "📈 5日経過",
            "last_updated": "2026-09-18 18:00:00",
        }
        out = normalize_alert_history(pd.DataFrame([older, newer]))
        self.assertEqual(len(out), 1)
        self.assertAlmostEqual(float(out.iloc[0]["ret_5d"]), 6.5)
        self.assertEqual(out.iloc[0]["outcome_status"], "📈 5日経過")

    def test_append_priority_history_is_idempotent_same_day_same_ticker(self):
        priority = pd.DataFrame([{
            "snapshot_date": pd.Timestamp("2026-09-18"),
            "ticker": "1111",
            "name": "TEST",
            "alert_tier": "🔥 A 優先確認",
            "alert_score": 76,
            "phase": "🔥 COVER EARLY",
            "regime": "🔥 COVER + NEW MONEY",
            "optimizer_label": "⭐ ROBUST MATCH",
            "match_strength": 80,
            "cover_score": 82,
            "ignition_score": 78,
            "long_demand_score": 76,
            "short_pressure": 68,
            "confidence": 82,
            "vol_ratio": 2.2,
            "rs_watch": 90,
            "price": 1000,
            "alert_reason": "⭐ ROBUST MATCH",
            "promotion_reason": "昇格",
            "condition_text": "C70/I60/L60/P50/Q40",
        }])

        first, added1 = append_priority_alert_history(None, priority)
        second, added2 = append_priority_alert_history(first, priority)

        self.assertEqual(added1, 1)
        self.assertEqual(added2, 0)
        self.assertEqual(len(second), 1)

    def test_condition_versions_require_explicit_activation(self):
        versions, v10 = append_condition_version(
            None,
            robust_condition(),
            source="optimizer",
            horizon=5,
            note="first",
            activate=False,
        )
        self.assertEqual(v10, "v1.0")
        self.assertIsNone(get_active_condition_version(versions))

        versions, v11 = append_condition_version(
            versions,
            pd.Series({**robust_condition().to_dict(), "cover_min": 75}),
            source="reoptimization",
            horizon=5,
            note="second",
            activate=False,
        )
        self.assertEqual(v11, "v1.1")

        versions = activate_condition_version(versions, "v1.0")
        active = get_active_condition_version(versions)
        self.assertIsNotNone(active)
        self.assertEqual(active["version_id"], "v1.0")
        self.assertEqual(int(normalize_condition_versions(versions)["is_active"].sum()), 1)

    def test_health_monitor_detects_degradation_for_same_condition(self):
        dates = pd.date_range("2026-05-01", periods=12, freq="B")
        backtest = pd.DataFrame({
            "signal_date": dates,
            "cover_score": [80] * 12,
            "ignition_score": [75] * 12,
            "long_demand_score": [72] * 12,
            "short_pressure": [65] * 12,
            "confidence": [80] * 12,
            "ret_5d": [4.0, 3.0, 5.0, 2.5, 4.5, 3.5, 5.5, 2.0, 4.0, 3.0, 4.5, 3.5],
            "mfe_10d": [8.0] * 12,
            "mae_10d": [-2.0] * 12,
        })
        live_dates = pd.date_range("2026-08-01", periods=6, freq="B")
        history = pd.DataFrame({
            "alert_date": live_dates,
            "ticker": [f"{1000+i}" for i in range(6)],
            "condition_text": ["C70/I60/L60/P50/Q40"] * 6,
            "ret_5d": [-2.0, -1.5, -3.0, -2.5, -1.0, -2.0],
            "mfe_10d": [1.0] * 6,
            "mae_10d": [-5.0] * 6,
            "last_updated": live_dates,
        })

        health = compare_live_vs_backtest(
            backtest,
            history,
            condition=robust_condition(),
            horizon=5,
            recent_live_n=20,
            min_live_signals=5,
        )
        self.assertEqual(health["status"], "🔴 DEGRADED")
        self.assertLess(float(health["live_avg"]), float(health["backtest_avg"]))
        self.assertLess(float(health["health_score"]), 60)


if __name__ == "__main__":
    unittest.main()

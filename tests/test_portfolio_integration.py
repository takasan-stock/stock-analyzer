import unittest

import pandas as pd

from portfolio_integration import (
    apply_portfolio_priority,
    build_portfolio_signal_overlay,
    portfolio_priority_label,
    split_news_by_age,
)


class PortfolioIntegrationTests(unittest.TestCase):
    def test_signal_overlay_combines_me_deep_fcf_and_entry(self):
        portfolio = pd.DataFrame(
            [
                {"ティッカー": "3774", "銘柄名": "インターネットイニシアティブ"},
                {"ティッカー": "2670", "銘柄名": "エービーシー・マート"},
            ]
        )
        me = pd.DataFrame(
            [
                {
                    "trade_date": "2026-10-08",
                    "ticker": "3774",
                    "company_name": "インターネットイニシアティブ",
                    "sw_decision": "PASS",
                    "second_wave_state": "UNAVAILABLE",
                    "fcf_ttm": 100.0,
                    "fcf_ttm_complete": True,
                    "fcf_per_share_3y_cagr_score": 80.0,
                    "fcf_per_share_yoy_score": 80.0,
                    "cfo_yoy_score": 80.0,
                    "roic_score": 80.0,
                    "fcf_margin_score": 80.0,
                    "cash_conversion_score": 80.0,
                    "net_cash_ratio_score": 80.0,
                    "dilution_score": 80.0,
                    "fcf_quality_grade": "B",
                    "fcf_margin": 0.12,
                    "cash_conversion": 0.9,
                    "p_fcf": 20.0,
                },
                {
                    "trade_date": "2026-10-08",
                    "ticker": "2670",
                    "company_name": "エービーシー・マート",
                    "sw_decision": "WATCH",
                    "second_wave_state": "EXP. DECELERATING",
                    "fcf_ttm": 100.0,
                    "fcf_ttm_complete": True,
                    "fcf_per_share_3y_cagr_score": 20.0,
                    "fcf_per_share_yoy_score": 20.0,
                    "cfo_yoy_score": 20.0,
                    "roic_score": 20.0,
                    "fcf_margin_score": 20.0,
                    "cash_conversion_score": 20.0,
                    "net_cash_ratio_score": 20.0,
                    "dilution_score": 20.0,
                    "fcf_quality_grade": "D",
                    "fcf_margin": 0.05,
                    "cash_conversion": 0.4,
                    "p_fcf": 40.0,
                },
            ]
        )
        validation = {
            "me_watch_tickers": ["2670"],
            "rows": [
                {
                    "ticker": "3774",
                    "entry_status": "🟡 WAIT",
                    "opportunity_score": 62.4,
                    "opportunity_rating": "B",
                }
            ],
        }

        out = build_portfolio_signal_overlay(portfolio, me, validation)

        ii = out[out["ティッカー"] == "3774"].iloc[0]
        abc = out[out["ティッカー"] == "2670"].iloc[0]

        self.assertEqual(ii["ME状態"], "PASS")
        self.assertNotEqual(ii["Deep FCF"], "—")
        self.assertEqual(ii["Entry状態"], "🟡 WAIT")
        self.assertEqual(ii["Opportunity"], "B 62")

        self.assertEqual(abc["ME状態"], "WATCH")
        self.assertEqual(abc["Deep FCF"], "—")
        self.assertEqual(abc["Entry状態"], "🟡 ME WATCH")

    def test_split_news_by_age_uses_30_day_boundary(self):
        items = [
            {"title": "A", "hours_ago": 24},
            {"title": "B", "hours_ago": 30 * 24},
            {"title": "C", "hours_ago": 30 * 24 + 1},
            {"title": "D", "hours_ago": None},
        ]

        recent, older = split_news_by_age(items, recent_days=30)

        self.assertEqual([x["title"] for x in recent], ["A", "B"])
        self.assertEqual([x["title"] for x in older], ["C", "D"])

    def test_priority_labels_do_not_change_trading_scores(self):
        frame = pd.DataFrame(
            [
                {
                    "ティッカー": "1111",
                    "Entry状態": "🟢 ENTRY READY",
                    "ME状態": "READY",
                    "Deep FCF": "80 STRONG",
                    "Opportunity": "A 82",
                },
                {
                    "ティッカー": "2222",
                    "Entry状態": "🟡 WAIT",
                    "ME状態": "WATCH",
                    "Deep FCF": "—",
                    "Opportunity": "B 61",
                },
                {
                    "ティッカー": "3333",
                    "Entry状態": "—",
                    "ME状態": "—",
                    "Deep FCF": "72 GOOD",
                    "Opportunity": "—",
                },
                {
                    "ティッカー": "4444",
                    "Entry状態": "—",
                    "ME状態": "—",
                    "Deep FCF": "—",
                    "Opportunity": "—",
                },
            ]
        )

        out = apply_portfolio_priority(frame)

        self.assertEqual(
            list(out["今日の優先度"]),
            [
                "🔴 P1 今すぐ確認",
                "🟠 P2 優先監視",
                "🟡 P3 通常監視",
                "⚪ 未評価",
            ],
        )
        self.assertEqual(list(out["_priority_rank"]), [1, 2, 3, 5])
        self.assertEqual(list(out["Opportunity"]), ["A 82", "B 61", "—", "—"])

    def test_apply_priority_adds_columns_to_plain_portfolio(self):
        frame = pd.DataFrame(
            [
                {"ティッカー": "4187", "銘柄名": "大阪有機化学工業"},
                {"ティッカー": "9560", "銘柄名": "プログリット"},
            ]
        )

        out = apply_portfolio_priority(frame)

        self.assertIn("今日の優先度", out.columns)
        self.assertIn("_priority_rank", out.columns)
        self.assertEqual(list(out["今日の優先度"]), ["⚪ 未評価", "⚪ 未評価"])
        self.assertEqual(list(out["_priority_rank"]), [5, 5])

    def test_priority_opportunity_thresholds(self):
        label, rank = portfolio_priority_label(
            {
                "Entry状態": "—",
                "ME状態": "—",
                "Deep FCF": "—",
                "Opportunity": "A 72",
            }
        )
        self.assertEqual((label, rank), ("🔴 P1 今すぐ確認", 1))

        label, rank = portfolio_priority_label(
            {
                "Entry状態": "🟡 WAIT",
                "ME状態": "—",
                "Deep FCF": "—",
                "Opportunity": "B 58",
            }
        )
        self.assertEqual((label, rank), ("🟠 P2 優先監視", 2))

    def test_wait_with_low_opportunity_stays_p4(self):
        label, rank = portfolio_priority_label(
            {
                "Entry状態": "🟡 WAIT",
                "ME状態": "—",
                "Deep FCF": "—",
                "Opportunity": "C 29",
            }
        )
        self.assertEqual((label, rank), ("⚪ P4 待機", 4))

    def test_low_signal_but_evaluated_name_is_p4(self):
        label, rank = portfolio_priority_label(
            {
                "Entry状態": "🟡 WAIT",
                "ME状態": "—",
                "Deep FCF": "—",
                "Opportunity": "C 29",
            }
        )
        self.assertEqual((label, rank), ("⚪ P4 待機", 4))

    def test_me_watch_and_mid_opportunity_promote_to_p3(self):
        label, rank = portfolio_priority_label(
            {
                "Entry状態": "🟡 ME WATCH",
                "ME状態": "WATCH",
                "Deep FCF": "—",
                "Opportunity": "—",
            }
        )
        self.assertEqual((label, rank), ("🟡 P3 通常監視", 3))

        label, rank = portfolio_priority_label(
            {
                "Entry状態": "—",
                "ME状態": "—",
                "Deep FCF": "—",
                "Opportunity": "C 45",
            }
        )
        self.assertEqual((label, rank), ("🟡 P3 通常監視", 3))

    def test_baseline_me_reduces_unassessed_without_opening_high_priority(self):
        portfolio = pd.DataFrame(
            [
                {"ティッカー": "1111", "銘柄名": "Alpha"},
                {"ティッカー": "2222", "銘柄名": "Beta"},
                {"ティッカー": "3333", "銘柄名": "Gamma"},
            ]
        )
        universe = pd.DataFrame(
            [
                {
                    "ticker": "1111",
                    "trade_date": "2026-10-09",
                    "prefilter_state": "RE-READY PROXY",
                    "coarse_score": 72.0,
                    "eligible": True,
                },
                {
                    "ticker": "2222",
                    "trade_date": "2026-10-09",
                    "prefilter_state": "BASE",
                    "coarse_score": 58.0,
                    "eligible": True,
                },
            ]
        )

        out = build_portfolio_signal_overlay(
            portfolio,
            pd.DataFrame(),
            {},
            universe,
        )

        alpha = out[out["ティッカー"] == "1111"].iloc[0]
        beta = out[out["ティッカー"] == "2222"].iloc[0]
        gamma = out[out["ティッカー"] == "3333"].iloc[0]

        self.assertEqual(alpha["Baseline ME"], "RE-READY PROXY")
        self.assertEqual(alpha["今日の優先度"], "🟡 P3 通常監視")
        self.assertEqual(alpha["_priority_rank"], 3)

        self.assertEqual(beta["Baseline ME"], "BASE")
        self.assertEqual(beta["今日の優先度"], "⚪ P4 待機")
        self.assertEqual(beta["_priority_rank"], 4)

        self.assertEqual(gamma["Baseline ME"], "—")
        self.assertEqual(gamma["今日の優先度"], "⚪ 未評価")
        self.assertEqual(gamma["_priority_rank"], 5)

    def test_baseline_score_alone_never_creates_p1_or_p2(self):
        label, rank = portfolio_priority_label(
            {
                "Entry状態": "—",
                "ME状態": "—",
                "Deep FCF": "—",
                "Opportunity": "—",
                "Baseline ME": "BASE",
                "Baseline Score": "88",
            }
        )
        self.assertEqual((label, rank), ("🟡 P3 通常監視", 3))

    def test_me_ready_is_p2_without_opportunity_score(self):
        label, rank = portfolio_priority_label(
            {
                "Entry状態": "—",
                "ME状態": "READY",
                "Deep FCF": "—",
                "Opportunity": "—",
            }
        )
        self.assertEqual((label, rank), ("🟠 P2 優先監視", 2))


if __name__ == "__main__":
    unittest.main()

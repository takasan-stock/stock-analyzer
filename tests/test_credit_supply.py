import io
import unittest

import pandas as pd

from credit_supply import analyze_credit_supply, build_signal_confluence, parse_jpx_margin_workbook


class CreditSupplyTests(unittest.TestCase):
    def test_parse_two_row_header(self):
        raw = pd.DataFrame(
            [
                ["銘柄別信用取引残高", None, None, None, None, None, None, None],
                ["2026年9月29日申込分", None, None, None, None, None, None, None],
                ["銘柄コード", "銘柄名", "売残高", "売残高", "売残高", "買残高", "買残高", "買残高"],
                ["銘柄コード", "銘柄名", "制度", "一般", "合計", "制度", "一般", "合計"],
                ["4063", "信越化学工業", 100, 50, 150, 600, 200, 800],
            ]
        )
        bio = io.BytesIO()
        with pd.ExcelWriter(bio, engine="openpyxl") as writer:
            raw.to_excel(writer, index=False, header=False, sheet_name="data")

        parsed = parse_jpx_margin_workbook(
            bio.getvalue(),
            default_date=pd.Timestamp("2026-09-29"),
        )
        self.assertEqual(len(parsed), 1)
        row = parsed.iloc[0]
        self.assertEqual(row["ticker"], "4063")
        self.assertEqual(float(row["long_balance"]), 800)
        self.assertEqual(float(row["short_balance"]), 150)
        self.assertEqual(float(row["long_general"]), 200)
        self.assertEqual(float(row["long_standard"]), 600)

    def test_supply_improves_when_price_up_and_long_balance_down(self):
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-09-28", "2026-09-29"]),
                "ticker": ["4063", "4063"],
                "name": ["信越化学工業", "信越化学工業"],
                "long_balance": [1000, 800],
                "short_balance": [100, 150],
            }
        )
        result = analyze_credit_supply(
            history,
            "4063",
            latest_close=5000,
            previous_close=4900,
            avg_volume_5=2000,
            avg_volume_25=2000,
            shares_outstanding=1_000_000,
        )
        self.assertIn(result["status"], {"🟢 改善優位", "🟢 やや改善"})
        self.assertEqual(result["long_delta"], -200)
        self.assertEqual(result["short_delta"], 50)
        self.assertLess(result["credit_ratio"], result["credit_ratio_prev"])
        self.assertTrue(any("株価上昇" in x for x in result["reasons"]))

    def test_supply_warns_when_price_down_and_long_balance_up(self):
        history = pd.DataFrame(
            {
                "date": pd.to_datetime(["2026-09-28", "2026-09-29"]),
                "ticker": ["472A", "472A"],
                "name": ["ミラティブ", "ミラティブ"],
                "long_balance": [1000, 1500],
                "short_balance": [200, 150],
            }
        )
        result = analyze_credit_supply(
            history,
            "472A",
            latest_close=900,
            previous_close=1000,
            avg_volume_5=200,
            avg_volume_25=200,
        )
        self.assertEqual(result["status"], "🟠 悪化注意")
        self.assertGreater(result["long_delta"], 0)
        self.assertTrue(any("株価下落中" in x for x in result["risks"]))


    def test_three_way_confluence(self):
        result = build_signal_confluence(
            cover_score=82,
            credit_score=78,
            entry_score=76,
        )
        self.assertEqual(result["status"], "🔥 3点一致")
        self.assertEqual(result["available_count"], 3)
        self.assertGreaterEqual(result["score"], 75)

    def test_confluence_renormalizes_missing_credit(self):
        result = build_signal_confluence(
            cover_score=80,
            credit_score=None,
            entry_score=70,
        )
        self.assertEqual(result["available_count"], 2)
        self.assertEqual(result["coverage"], 70.0)
        self.assertIsNotNone(result["score"])



if __name__ == "__main__":
    unittest.main()

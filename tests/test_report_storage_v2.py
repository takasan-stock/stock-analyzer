import datetime as dt
import unittest

from report_storage_v2 import (
    display_entry_key,
    make_unique_entry_key,
    merge_report_entries,
    unwrap_report_content,
    wrap_report_content,
)


class ReportStorageV2Tests(unittest.TestCase):
    def test_unique_key_keeps_first_and_suffixes_duplicate(self):
        existing = {"2026-10-11"}
        now = dt.datetime(2026, 10, 11, 19, 30, 45)

        first = make_unique_entry_key("2026-10-12", existing, now=now)
        duplicate = make_unique_entry_key("2026-10-11", existing, now=now)

        self.assertEqual(first, "2026-10-12")
        self.assertEqual(duplicate, "2026-10-11_20261011_193045")

    def test_unique_key_increments_when_timestamp_already_exists(self):
        existing = {
            "2026-4Q",
            "2026-4Q_20261011_193045",
        }
        now = dt.datetime(2026, 10, 11, 19, 30, 45)

        key = make_unique_entry_key("2026-4Q", existing, now=now)

        self.assertEqual(key, "2026-4Q_20261011_193045_2")

    def test_metadata_round_trip_is_invisible_to_body(self):
        wrapped = wrap_report_content(
            "# Report\nBody",
            kind="analysis",
            key="2026-10-11",
            created_at="2026-10-11T10:00:00+09:00",
            updated_at="2026-10-11T11:00:00+09:00",
        )

        body, meta = unwrap_report_content(wrapped)

        self.assertEqual(body, "# Report\nBody")
        self.assertEqual(meta["kind"], "analysis")
        self.assertEqual(meta["key"], "2026-10-11")
        self.assertEqual(meta["created_at"], "2026-10-11T10:00:00+09:00")
        self.assertEqual(meta["updated_at"], "2026-10-11T11:00:00+09:00")

    def test_legacy_markdown_without_metadata_is_compatible(self):
        body, meta = unwrap_report_content("# Legacy\nBody")

        self.assertEqual(body, "# Legacy\nBody")
        self.assertEqual(meta, {})

    def test_merge_keeps_local_only_when_remote_missing(self):
        local = [
            {
                "key": "2026-10-11_20261011_193045",
                "content": "local-only",
                "updated_at": "2026-10-11 19:30",
            }
        ]

        out = merge_report_entries(local, [])

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["content"], "local-only")
        self.assertEqual(out[0]["_sync_status"], "LOCAL ONLY")

    def test_remote_wins_same_key_but_inherits_local_timestamp(self):
        local = [
            {
                "key": "2026-10-11",
                "content": "local",
                "updated_at": "2026-10-11 19:30",
            }
        ]
        remote = [
            {
                "key": "2026-10-11",
                "content": "remote",
                "updated_at": "",
                "_sha": "abc",
            }
        ]

        out = merge_report_entries(local, remote)

        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["content"], "remote")
        self.assertEqual(out[0]["updated_at"], "2026-10-11 19:30")
        self.assertEqual(out[0]["_sync_status"], "GITHUB")

    def test_duplicate_display_label_includes_save_time(self):
        self.assertEqual(
            display_entry_key(
                "2026-10-11_20261011_193045",
                "analysis",
            ),
            "2026-10-11｜追加保存 2026-10-11 19:30:45",
        )
        self.assertEqual(
            display_entry_key(
                "2026-4Q_20261011_193045",
                "earnings",
            ),
            "2026-4Q｜追加保存 2026-10-11 19:30:45",
        )


if __name__ == "__main__":
    unittest.main()

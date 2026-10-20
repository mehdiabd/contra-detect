from datetime import date
from unittest import TestCase

from pydantic import ValidationError

from src.api.main import AnalyzeRequest


class AnalyzeRequestTests(TestCase):
    def test_normalizes_platform_and_user_ids(self):
        request = AnalyzeRequest(
            user_ids=["@alice", " alice ", "", "@bob"],
            platform_name="X",
        )

        self.assertEqual(request.user_ids, ["alice", "bob"])
        self.assertEqual(request.platform_name, "twitter")

    def test_accepts_empty_user_ids_and_optional_date_range(self):
        request = AnalyzeRequest(
            user_ids=[],
            platform_name="telegram",
            date_range={"start_date": "2026-01-01", "end_date": "2026-01-31"},
        )

        self.assertEqual(request.user_ids, [])
        self.assertEqual(request.date_range.start_date, date(2026, 1, 1))
        self.assertEqual(request.date_range.end_date, date(2026, 1, 31))

    def test_rejects_reversed_date_range(self):
        with self.assertRaises(ValidationError):
            AnalyzeRequest(
                user_ids=[],
                platform_name="instagram",
                date_range={"start_date": "2026-02-01", "end_date": "2026-01-01"},
            )


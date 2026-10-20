from datetime import date
from unittest import TestCase
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from src.api.main import app
from src.data.errors import DataSourceError
from src.data.schema import Post
from src.detection.engine import ContradictionEngine


ENDPOINTS = ("contradiction", "contradictory-posts", "contradiction-score")


class StubPredictor:
    def __init__(self, score=.1):
        self.score = score

    def predict_pairs(self, pairs):
        return [{"pair": pair, "contradiction_score": self.score,
                 "label": "contradiction" if self.score >= .55 else "neutral"} for pair in pairs]


def make_posts(user_id="audit-user"):
    return [
        Post(user_id, "من از تولید داخلی حمایت می‌کنم.", timestamp="2026-01-01"),
        Post(user_id, "من از تولید داخلی حمایت نمی‌کنم.", timestamp="2026-01-02"),
    ]


class AnalysisStatusTests(TestCase):
    def test_no_usable_pair_is_unknown_without_loading_or_calling_model(self):
        cases = [[], make_posts()[:1], [Post("audit-user", "کوتاه"), Post("audit-user", "متن")],
                 [Post("audit-user", "https://example.com/a"), Post("audit-user", "https://example.com/b")]]
        for posts in cases:
            with self.subTest(posts=posts), patch("src.detection.engine.ContradictionPredictor") as load_model:
                row = ContradictionEngine().analyze_users(["audit-user"], posts=posts)[0]
                self.assertEqual(row["analysis_status"], "insufficient_data")
                self.assertIsNone(row["has_contradiction"])
                self.assertIsNone(row["score"])
                self.assertEqual(row["pair_count"], 0)
                load_model.assert_not_called()

    def test_date_filter_can_leave_insufficient_data(self):
        row = ContradictionEngine(predictor=Mock()).analyze_users(
            ["audit-user"], posts=make_posts(), start_date=date(2026, 2, 1),
        )[0]
        self.assertEqual(row["analysis_status"], "insufficient_data")
        self.assertEqual(row["post_count"], 0)
        self.assertIsNone(row["has_contradiction"])

    def test_mixed_users_keep_individual_analysis_status(self):
        engine = ContradictionEngine(predictor=StubPredictor())
        rows = engine.analyze_users(["audit-user", "missing-user"], posts=make_posts())
        self.assertEqual([r["analysis_status"] for r in rows], ["completed", "insufficient_data"])
        self.assertIs(rows[0]["has_contradiction"], False)
        self.assertIsInstance(rows[0]["score"], float)
        self.assertIsNone(rows[1]["has_contradiction"])


class AnalysisApiTests(TestCase):
    def setUp(self):
        self.engine = ContradictionEngine(predictor=StubPredictor())
        engine_patch = patch("src.api.main.get_engine", return_value=self.engine)
        engine_patch.start()
        self.addCleanup(engine_patch.stop)
        self.client = TestClient(app)
        self.addCleanup(self.client.close)

    def request(self, endpoint, posts):
        body = {"user_ids": ["audit-user"], "platform_name": "twitter"}
        if posts is not None:
            body["posts"] = [post.to_dict() for post in posts]
        return self.client.post("/api/v1/users/" + endpoint, json=body)

    def test_all_endpoints_identify_insufficient_data(self):
        for endpoint in ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                response = self.request(endpoint, [])
                self.assertEqual(response.status_code, 200)
                row = response.json()["results"][0]
                self.assertEqual(row["analysis_status"], "insufficient_data")
                self.assertEqual(row["post_count"], 0)
                self.assertEqual(row["pair_count"], 0)
                for key in ("has_contradiction", "score"):
                    if key in row:
                        self.assertIsNone(row[key])

    def test_completed_negative_result_remains_distinct_from_unknown(self):
        for endpoint in ENDPOINTS:
            with self.subTest(endpoint=endpoint):
                response = self.request(endpoint, make_posts())
                self.assertEqual(response.status_code, 200)
                row = response.json()["results"][0]
                self.assertEqual(row["analysis_status"], "completed")
                self.assertEqual(row["pair_count"], 1)
                if "has_contradiction" in row:
                    self.assertIs(row["has_contradiction"], False)
                if "score" in row:
                    self.assertEqual(row["score"], .025)

    def test_completed_positive_result_is_preserved(self):
        self.engine.predictor = StubPredictor(.9)
        row = self.request("contradiction", make_posts()).json()["results"][0]
        self.assertEqual(row["analysis_status"], "completed")
        self.assertIs(row["has_contradiction"], True)
        self.assertEqual(row["score"], .9)

    def test_source_failure_returns_503_on_all_endpoints_without_internal_details(self):
        with patch("src.detection.engine.elastic.is_configured", return_value=True), \
             patch("src.detection.engine.elastic.fetch_posts", side_effect=DataSourceError("private source details")):
            for endpoint in ENDPOINTS:
                with self.subTest(endpoint=endpoint):
                    response = self.request(endpoint, None)
                    self.assertEqual(response.status_code, 503)
                    self.assertEqual(response.json()["detail"]["code"], "data_source_unavailable")
                    self.assertNotIn("results", response.json())
                    self.assertNotIn("private source details", response.text)

    def test_empty_successful_source_is_insufficient_data_not_source_failure(self):
        with patch("src.detection.engine.elastic.is_configured", return_value=True), \
             patch("src.detection.engine.elastic.fetch_posts", return_value=[]):
            response = self.request("contradiction", None)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["analysis_status"], "insufficient_data")

    def test_supplied_posts_do_not_require_source_availability(self):
        with patch("src.detection.engine.elastic.fetch_posts", side_effect=DataSourceError("offline")) as fetch:
            response = self.request("contradiction", make_posts())
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["results"][0]["analysis_status"], "completed")
        fetch.assert_not_called()

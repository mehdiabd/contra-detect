import copy
import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from scripts.acceptance_review import binary_metrics, fingerprint, review_pairs, score
from scripts.run_acceptance import AcceptanceRun, GatewayTimeoutError, check_rows, compare_endpoints
from src.data.schema import Post


class AcceptanceEvaluationTests(TestCase):
    def setUp(self):
        self.cases = [{"case_id": f"{platform}-01", "platform": platform, "user_id": "u", "review_posts": [
            {"post_id": "a", "text": "متن نخست برای بررسی تناقض", "platform": platform, "user_id": "u"},
            {"post_id": "b", "text": "متن دیگر برای بررسی تناقض", "platform": platform, "user_id": "u"},
        ]} for platform in ("twitter", "telegram", "instagram")]
        self.labels = {"cases_sha256": fingerprint(self.cases), "reviewer": "human-reviewer", "labels": [
            {"pair_id": row["pair_id"], "label": "not_contradiction"} for row in review_pairs(self.cases)
        ]}
        self.predictions = [{"case_id": c["case_id"], "results": {
            "contradiction": {"analysis_status": "completed", "post_count": 2, "has_contradiction": False},
            "contradictory-posts": {"analysis_status": "completed", "post_count": 2, "contradictory_posts": []},
        }} for c in self.cases]

    def test_confusion_matrix_counts_false_positives_and_missed_contradictions(self):
        metrics = binary_metrics([True, True, False, False], [True, False, True, False])
        self.assertEqual([metrics[k] for k in ("tp", "fn", "fp", "tn")], [1, 1, 1, 1])
        self.assertEqual(metrics["accuracy"], .5)
        self.assertLess(metrics["wilson_95_percent"][0], .5)
        self.assertGreater(metrics["wilson_95_percent"][1], .5)

    def test_all_negative_perfect_accuracy_does_not_pass_acceptance(self):
        result = score(self.cases, self.predictions, self.labels)
        self.assertEqual(result["pair_metrics"]["all"]["accuracy"], 1)
        self.assertEqual(result["status"], "awaiting_complete_adjudicated_labels_or_class_coverage")

    def test_unknown_or_duplicate_labels_and_wrong_snapshot_are_rejected(self):
        for mutate in (lambda x: x.update(cases_sha256="wrong"),
                       lambda x: x["labels"].append(x["labels"][0]),
                       lambda x: x["labels"][0].update(pair_id="unknown")):
            labels = copy.deepcopy(self.labels)
            mutate(labels)
            with self.assertRaises(ValueError):
                score(self.cases, self.predictions, labels)

    def test_unselected_gold_contradiction_counts_as_a_false_negative(self):
        self.labels["labels"][0]["label"] = "contradiction"
        with patch("scripts.acceptance_review.CandidateBuilder.build", return_value=[]):
            result = score(self.cases, self.predictions, self.labels)
        self.assertEqual(result["pair_metrics"]["all"]["fn"], 1)
        self.assertEqual(result["user_metrics"]["all"]["fn"], 1)
        self.assertEqual(result["gold_contradictions_not_selected"], 1)

    def test_empty_endpoint_results_cannot_silently_pass_user_coverage(self):
        errors = check_rows({"results": []}, "contradiction", self.cases[:1], frozen=True)
        self.assertIn("requested_user_order_or_coverage", errors)
        self.assertIn("not_all_endpoints_available", compare_endpoints({}))

    def test_malformed_upstream_response_is_recorded_without_crashing_the_run(self):
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun("https://example.invalid", Path(directory)/"public", Path(directory)/"private")
            run.session = Mock()
            run.session.post.return_value.status_code = 200
            run.session.post.return_value.json.return_value = []
            self.assertIsNone(run.request("malformed", "contradiction", {"platform_name": "twitter"}, []))
            self.assertFalse(run.checks[0]["passed"])

    def test_gateway_timeout_preserves_evidence_and_an_incomplete_summary(self):
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun("https://example.invalid", Path(directory)/"public", Path(directory)/"private")
            run.session = Mock()
            run.session.post.return_value.status_code = 504
            run.session.post.return_value.json.return_value = {"message": "The upstream server is timing out"}
            (run.output/"health.json").write_text(json.dumps({"model": {"release": {"verified": True}}}))
            (run.output/"openapi.json").write_text(json.dumps({"info": {"version": "1.1.2"}}))
            with self.assertRaises(GatewayTimeoutError):
                run.request("gateway", "contradiction", {"platform_name": "twitter"}, [])
            summary = run.summarize([], completed=False, stopped_reason="gateway_http_504")
            self.assertFalse(summary["completed"])
            self.assertEqual(summary["failed"], 1)
            self.assertEqual(len(json.loads((run.private/"api_receipts.json").read_text())), 1)

    def test_client_timeout_is_configurable_and_must_be_positive(self):
        with tempfile.TemporaryDirectory() as directory:
            output, private = Path(directory)/"public", Path(directory)/"private"
            with self.assertRaises(ValueError):
                AcceptanceRun("https://example.invalid", output, private, request_timeout=0)
            run = AcceptanceRun("https://example.invalid", output, private, request_timeout=330)
            run.session = Mock()
            run.session.post.return_value.status_code = 422
            run.session.post.return_value.json.return_value = {"detail": []}
            run.request("invalid_dates", "contradiction", {"platform_name": "twitter"}, [], expected_status=422)
            self.assertEqual(run.session.post.call_args.kwargs["timeout"], 330)

    def test_live_reference_refresh_preserves_the_blind_review_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            run = AcceptanceRun("https://example.invalid", Path(directory)/"public", Path(directory)/"private",
                                refresh_source_reference=True)
            case = {**copy.deepcopy(self.cases[0]), "source_posts": [],
                    "date_range": {"start_date": "2026-10-01", "end_date": "2026-10-26"}}
            original_hash = fingerprint(case)
            post = Post(user_id="u", text="نوشتهٔ مرجع فعلی", post_id="new")
            with patch("src.data.elastic.fetch_posts", return_value=[post]) as fetch:
                current = run.current_source_case(case)
            self.assertEqual(fingerprint(case), original_hash)
            self.assertEqual(current["review_posts"], case["review_posts"])
            self.assertEqual(current["source_posts"], [post.to_dict()])
            self.assertEqual(fetch.call_args.args[0], ["u"])
            self.assertTrue((run.private/"source_reference.json").is_file())

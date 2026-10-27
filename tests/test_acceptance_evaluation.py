import copy
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import Mock, patch

from scripts.acceptance_review import binary_metrics, fingerprint, review_pairs, score
from scripts.run_acceptance import AcceptanceRun, check_rows, compare_endpoints


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

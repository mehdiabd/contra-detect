import copy
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch

from src.models.feedback import (FeedbackPair, fingerprint, isolated_output, partial_label_loss,
                                 read_feedback, split_feedback, validation_eligible)
from scripts.train_human_feedback import replay_records, trainable_modules
from src.data.schema import Pair


class HumanFeedbackTests(TestCase):
    def case_and_review(self):
        cases = [{"case_id": "twitter-01", "platform": "twitter", "review_posts": [
            {"text": "نوشتهٔ اول"}, {"text": "نوشتهٔ دوم"}, {"text": "نوشتهٔ سوم"},
        ]}]
        review = {"cases_sha256": fingerprint(cases), "reviewer": "reviewer-1", "labels": [
            {"pair_id": "twitter-01-p01", "label": "not_contradiction"},
            {"pair_id": "twitter-01-p02", "label": "uncertain"},
            {"pair_id": "twitter-01-p03", "label": "contradiction"},
        ]}
        return cases, review

    def test_uncertainty_is_not_a_negative_label(self):
        cases, review = self.case_and_review()
        examples, metadata = read_feedback(cases, review)
        self.assertEqual([p.label for p in examples], [-1, 1])
        self.assertEqual(metadata["uncertain_excluded"], 1)
        self.assertFalse(metadata["adjudicated"])

    def test_wrong_snapshot_duplicate_or_missing_annotations_are_rejected(self):
        cases, review = self.case_and_review()
        for change in (lambda r: r.update(cases_sha256="wrong"),
                       lambda r: r["labels"].append(r["labels"][0]),
                       lambda r: r["labels"].pop()):
            changed = copy.deepcopy(review)
            change(changed)
            with self.assertRaises(ValueError):
                read_feedback(cases, changed)

    def test_negative_feedback_accepts_entailment_without_forcing_neutral(self):
        logits = torch.tensor([[10., -5., 0.]])
        self.assertLess(float(partial_label_loss(logits, torch.tensor([-1]))), .001)
        self.assertGreater(float(partial_label_loss(logits, torch.tensor([2]))), 9.)

    def test_negative_feedback_reduces_contradiction_and_preserves_both_alternatives(self):
        logits = torch.tensor([[0., 5., 0.]], requires_grad=True)
        partial_label_loss(logits, torch.tensor([-1])).backward()
        self.assertGreater(float(logits.grad[0, 1]), 0)
        self.assertLess(float(logits.grad[0, 0]), 0)
        self.assertLess(float(logits.grad[0, 2]), 0)

    def test_positive_replay_still_learns_the_contradiction_class(self):
        logits = torch.tensor([[0., 0., 0.]], requires_grad=True)
        partial_label_loss(logits, torch.tensor([1])).backward()
        self.assertLess(float(logits.grad[0, 1]), 0)
        with self.assertRaises(ValueError):
            partial_label_loss(logits, torch.tensor([3]))

    def test_users_and_identical_posts_do_not_cross_splits(self):
        examples = [FeedbackPair(f"{platform}-{i}-pair", f"{platform}-{i}", platform,
                                 f"{platform}-{i}-a", f"{platform}-{i}-b", -1)
                    for platform in ("twitter", "telegram", "instagram") for i in range(4)]
        first = split_feedback(examples, 42)
        self.assertEqual(first, split_feedback(examples, 42))
        groups = {name: {p.case_id for p in pairs} for name, pairs in first.items()}
        self.assertFalse(groups["train"] & groups["validation"])
        self.assertFalse(groups["train"] & groups["test"])
        self.assertFalse(groups["validation"] & groups["test"])
        for name in ("validation", "test"):
            self.assertEqual({p.platform for p in first[name]}, {"twitter", "telegram", "instagram"})
        duplicates = [FeedbackPair(p.pair_id, p.case_id, p.platform, "same copied post", p.text_b, p.label)
                      for p in examples]
        with self.assertRaises(ValueError):
            split_feedback(duplicates, 42)

    def test_training_cannot_overwrite_the_production_directory_or_an_existing_run(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            production = root/"saved"
            production.mkdir()
            with self.assertRaises(ValueError):
                isolated_output(production/"member", production)
            existing = root/"experiment"
            existing.mkdir()
            with self.assertRaises(ValueError):
                isolated_output(existing, production)
            self.assertEqual(isolated_output(root/"new-run", production), (root/"new-run").resolve())

    def test_lower_negative_loss_cannot_override_nli_quality_gates(self):
        baseline = {"human_negative_loss": 1., "farstail": {
            "accuracy": .87, "per_class": {"contradiction": {"f1": .80}}}}
        candidate = copy.deepcopy(baseline)
        candidate["human_negative_loss"] = .5
        self.assertTrue(validation_eligible(baseline, candidate))
        candidate["farstail"]["accuracy"] = .84
        self.assertFalse(validation_eligible(baseline, candidate))
        candidate["farstail"]["accuracy"] = .87
        candidate["farstail"]["per_class"]["contradiction"]["f1"] = .70
        self.assertFalse(validation_eligible(baseline, candidate))

    def test_only_the_last_encoder_layer_and_classifier_heads_are_trainable(self):
        model = torch.nn.Module()
        model.base_model = torch.nn.Module()
        model.base_model.embeddings = torch.nn.Linear(2, 2)
        model.base_model.encoder = torch.nn.Module()
        model.base_model.encoder.layer = torch.nn.ModuleList([torch.nn.Linear(2, 2), torch.nn.Linear(2, 2)])
        model.pooler = torch.nn.Linear(2, 2)
        model.classifier = torch.nn.Linear(2, 3)
        trainable_modules(model)
        self.assertFalse(any(p.requires_grad for p in model.base_model.embeddings.parameters()))
        self.assertFalse(any(p.requires_grad for p in model.base_model.encoder.layer[0].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.base_model.encoder.layer[-1].parameters()))
        self.assertTrue(all(p.requires_grad for p in model.pooler.parameters()))
        self.assertTrue(all(p.requires_grad for p in model.classifier.parameters()))

    def test_replay_excludes_human_texts_and_exact_heldout_pairs(self):
        human = [FeedbackPair("p", "u", "twitter", "human-a", "human-b", -1)]
        train = [Pair(f"train-{i}-a", f"train-{i}-b", label=i) for i in range(3)]
        duplicate = Pair("validation-a", "validation-b", label=1)
        contaminated = Pair("human-a", "different", label=0)
        splits = {"train": train + [duplicate, contaminated], "val": [duplicate], "test": []}
        with patch("scripts.train_human_feedback.load_farstail_split", side_effect=lambda name: splits[name]):
            records = replay_records(human, seed=42, per_class=1)
        self.assertEqual(len(records), 3)
        self.assertEqual({r["label"] for r in records}, {0, 1, 2})
        self.assertFalse(any(r["text_a"] in {"human-a", "validation-a"} for r in records))

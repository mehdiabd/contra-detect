import json
import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch
from safetensors.torch import save_file

from scripts.promote_human_feedback import atomic_json, promote
from src.models.release import file_sha256, verify_model_release


class FeedbackPromotionTests(TestCase):
    def fixture(self, root):
        models, experiment = root/"models", root/"experiment"
        bundle, member = models/"saved/best_model", models/"saved/mdeberta_nli"
        bundle.mkdir(parents=True)
        member.mkdir()
        (experiment/"candidate").mkdir(parents=True)
        config = {"model_type": "deberta-v2", "hidden_size": 1, "vocab_size": 2,
                  "max_position_embeddings": 4, "id2label": {"0": "entailment", "1": "neutral", "2": "contradiction"}}
        atomic_json(member/"config.json", config)
        atomic_json(experiment/"candidate/config.json", config)
        for name in ("tokenizer.json", "tokenizer_config.json"):
            (member/name).write_text("{}")
        save_file({"weight": torch.tensor([1.], dtype=torch.float16),
                   "deberta.embeddings.position_ids": torch.arange(4).reshape(1, -1)}, str(member/"model.safetensors"))
        save_file({"weight": torch.tensor([2.])}, str(experiment/"candidate/model.safetensors"))
        atomic_json(bundle/"inference_config.json", {"version": 1, "members": [
            {"name": "mdeberta", "path": "../mdeberta_nli", "weight": 1.}]})
        paths = [bundle/"inference_config.json", *member.iterdir()]
        files = {str(p.relative_to(models)): {"bytes": p.stat().st_size, "sha256": file_sha256(p)} for p in paths}
        atomic_json(models/"release.json", {"version": 1, "release_id": "old", "bundle": "saved/best_model", "files": files,
                    "evaluation": {"dataset": "FarsTail", "split": "test", "n": 100, "correct": 86, "accuracy": .86}})
        report = {"eligible_for_promotion": True, "promotion_gates": {"quality": True},
                  "epochs_completed": 1, "epochs_requested": 1, "selected_epoch": 1,
                  "history": [{"epoch": 1, "eligible": True}],
                  "base_manifest_sha256": file_sha256(models/"release.json"),
                  "candidate_model_sha256": file_sha256(experiment/"candidate/model.safetensors"),
                  "feedback": {"reviewer_count": 1, "adjudicated": False},
                  "split_pairs": {"train": 36, "validation": 18, "test": 18},
                  "test": {"n": 100, "correct": 87, "runtime_seconds": 1, "data_sha256": "dataset-hash",
                           "farstail": {"accuracy": .87, "per_class": {}, "confusion_matrix": [], "matrix_order": []}}}
        atomic_json(experiment/"training_report.json", report)
        return models, experiment, report

    def test_promotes_verified_weights_and_keeps_the_old_release(self):
        with tempfile.TemporaryDirectory() as directory:
            models, experiment, report = self.fixture(Path(directory))
            old_hash = file_sha256(models/"saved/mdeberta_nli/model.safetensors")
            result = promote(experiment, models, "new")
            self.assertFalse(result["server_deployed"])
            self.assertEqual(file_sha256(models/"saved/mdeberta_nli/model.safetensors"), report["candidate_model_sha256"])
            self.assertEqual(file_sha256(experiment/"production_before_promotion.safetensors"), old_hash)
            self.assertEqual(verify_model_release(models/"saved/best_model")["release_id"], "new")

    def test_failed_quality_gate_cannot_modify_the_release(self):
        with tempfile.TemporaryDirectory() as directory:
            models, experiment, report = self.fixture(Path(directory))
            before = file_sha256(models/"release.json")
            report["promotion_gates"]["quality"] = False
            atomic_json(experiment/"training_report.json", report)
            with self.assertRaises(ValueError):
                promote(experiment, models, "new")
            self.assertEqual(file_sha256(models/"release.json"), before)

    def test_changed_checkpoint_cannot_be_published_with_old_metrics(self):
        with tempfile.TemporaryDirectory() as directory:
            models, experiment, report = self.fixture(Path(directory))
            before = file_sha256(models/"release.json")
            report["candidate_model_sha256"] = "wrong"
            atomic_json(experiment/"training_report.json", report)
            with self.assertRaisesRegex(ValueError, "checkpoint"):
                promote(experiment, models, "new")
            self.assertEqual(file_sha256(models/"release.json"), before)

    def test_failed_final_verification_restores_weights_and_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            models, experiment, report = self.fixture(Path(directory))
            old_weights = file_sha256(models/"saved/mdeberta_nli/model.safetensors")
            info = verify_model_release(models/"saved/best_model")
            with patch("scripts.promote_human_feedback.verify_model_release", side_effect=[info, ValueError("verification failed")]):
                with self.assertRaises(ValueError):
                    promote(experiment, models, "new")
            self.assertEqual(file_sha256(models/"saved/mdeberta_nli/model.safetensors"), old_weights)
            self.assertEqual(file_sha256(models/"release.json"), report["base_manifest_sha256"])
            self.assertTrue(verify_model_release(models/"saved/best_model")["verified"])

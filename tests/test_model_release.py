import json
import os
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

from src.api.main import health
from src.models.predictor import ContradictionPredictor
from src.models.release import file_sha256, verify_model_release


class ModelReleaseTests(TestCase):
    def make_release(self, root):
        model = root / "saved/best_model"
        member = root / "saved/member"
        model.mkdir(parents=True)
        member.mkdir()
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            (member / name).write_text("original model bytes")
        (model / "inference_config.json").write_text(json.dumps({"members": [{"name": "member", "path": "../member", "weight": 1}]}))
        files = {str(p.relative_to(root)): {"bytes": p.stat().st_size, "sha256": file_sha256(p)}
                 for p in [model / "inference_config.json", *member.iterdir()]}
        (root / "release.json").write_text(json.dumps({"version": 1, "release_id": "test", "bundle": "saved/best_model",
            "files": files, "evaluation": {"dataset": "FarsTail", "split": "test", "n": 100, "correct": 86, "accuracy": .86}}))
        return model, member

    def test_verifies_release_and_reports_exact_model_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, member = self.make_release(Path(tmp))
            info = verify_model_release(model)
            self.assertTrue(info["verified"])
            self.assertEqual(info["members"][0]["model_sha256"], file_sha256(member / "model.safetensors"))

    def test_rejects_same_size_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, member = self.make_release(Path(tmp))
            path = member / "model.safetensors"
            path.write_bytes(b"x" * path.stat().st_size)
            with self.assertRaisesRegex(ValueError, "checksum"):
                verify_model_release(model)

    def test_rejects_lfs_pointer_in_place_of_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            model, member = self.make_release(Path(tmp))
            (member / "model.safetensors").write_text("version https://git-lfs.github.com/spec/v1\noid sha256:abc\nsize 123\n")
            with self.assertRaisesRegex(ValueError, "git lfs pull"):
                verify_model_release(model)

    def test_production_does_not_fall_back_to_base_model(self):
        with patch.dict(os.environ, {"REQUIRE_MODEL_RELEASE": "true", "ENSEMBLE_MODELS": ""}), \
             patch("src.models.predictor.resolve_model_source", return_value="missing-release"), \
             patch("src.models.predictor.AutoTokenizer.from_pretrained") as download:
            with self.assertRaisesRegex(ValueError, "fallback"):
                ContradictionPredictor()
            download.assert_not_called()

    def test_health_uses_loaded_predictor_metadata(self):
        predictor = SimpleNamespace(bundle_dir="/opt/release/best_model", model_dir="member-a", model_sources=["member-a", "member-b"],
                                    member_weights=[.25, .75], max_length=512, release_info={"verified": True})
        with patch("src.api.main.get_engine", return_value=SimpleNamespace(predictor=predictor)), \
             patch("src.api.main.elastic.is_configured", return_value=False):
            response = health()
        self.assertEqual(response["model_source"], predictor.bundle_dir)
        self.assertEqual(response["model"]["sources"], predictor.model_sources)
        self.assertTrue(response["model"]["release"]["verified"])

    def test_health_not_ready_without_loaded_model(self):
        with patch("src.api.main.get_engine", return_value=SimpleNamespace(predictor=None)):
            self.assertEqual(health().status_code, 503)

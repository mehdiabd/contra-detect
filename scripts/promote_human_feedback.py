"""Promote a completed, gated feedback experiment while retaining the previous model."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

from safetensors import safe_open
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.release import file_sha256, verify_model_release


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".pending")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    os.replace(temporary, path)


def promote(experiment: Path, models: Path, release_id: str) -> dict:
    report = json.loads((experiment / "training_report.json").read_text())
    gates = report.get("promotion_gates") or {}
    if (not report.get("eligible_for_promotion") or not gates or not all(gates.values())
            or report["epochs_completed"] != report["epochs_requested"]
            or not any(h["epoch"] == report["selected_epoch"] and h["eligible"] for h in report["history"])):
        raise ValueError("Only a completed, validation-selected experiment that passed every gate can be promoted.")
    test = report["test"]
    score = test["farstail"]["accuracy"]
    if test["n"] <= 0 or score < .85 or abs(test["correct"] / test["n"] - score) > 1e-9:
        raise ValueError("The evaluated candidate does not meet the FarsTail release target.")
    bundle = models / "saved/best_model"
    previous = verify_model_release(bundle)
    if previous["manifest_sha256"] != report["base_manifest_sha256"]:
        raise ValueError("The production baseline changed after this experiment started.")
    if release_id == previous["release_id"]:
        raise ValueError("A changed model needs a new release identifier.")
    candidate = experiment / "candidate/model.safetensors"
    destination = models / "saved/mdeberta_nli/model.safetensors"
    if file_sha256(candidate) != report["candidate_model_sha256"]:
        raise ValueError("The candidate weights do not match the evaluated checkpoint.")
    old_config = json.loads((destination.parent / "config.json").read_text())
    new_config = json.loads((candidate.parent / "config.json").read_text())
    if any(old_config.get(k) != new_config.get(k) for k in ("model_type", "id2label", "hidden_size", "vocab_size")):
        raise ValueError("The candidate architecture or label order changed.")
    with safe_open(candidate, framework="pt") as new, safe_open(destination, framework="pt") as old:
        missing = set(old.keys()) - set(new.keys())
        # New Transformers versions omit this deterministic, non-persistent buffer.
        position_ids = "deberta.embeddings.position_ids"
        if missing == {position_ids}:
            actual = old.get_tensor(position_ids)
            expected = torch.arange(old_config["max_position_embeddings"], dtype=actual.dtype).reshape(1, -1)
            if not torch.equal(actual, expected):
                raise ValueError("The omitted position buffer is not the standard deterministic sequence.")
        elif missing:
            raise ValueError("The candidate omits learned parameters.")
        if set(new.keys()) - set(old.keys()) or any(new.get_slice(k).get_shape() != old.get_slice(k).get_shape() for k in new.keys()):
            raise ValueError("The candidate parameter structure changed.")
    manifest_path = models / "release.json"
    old_manifest = json.loads(manifest_path.read_text())
    backup = experiment / "production_before_promotion.safetensors"
    os.link(destination, backup)
    atomic_json(experiment / "release_before_promotion.json", old_manifest)
    temporary = destination.with_suffix(".pending")
    # The experiment is immutable after completion; a hard link avoids a second large copy.
    try:
        os.link(candidate, temporary)
    except OSError:
        shutil.copyfile(candidate, temporary)
    os.replace(temporary, destination)
    try:
        files = {relative: {"bytes": (models / relative).stat().st_size, "sha256": file_sha256(models / relative)}
                 for relative in old_manifest["files"]}
        evaluation = {"dataset": "FarsTail", "split": "test", "scope": "three_class_nli_on_labeled_pairs",
                      "n": test["n"], "correct": test["correct"], "accuracy": score,
                      "per_class": test["farstail"]["per_class"],
                      "confusion_matrix": test["farstail"]["confusion_matrix"],
                      "matrix_order": test["farstail"]["matrix_order"],
                      "runtime_seconds": test["runtime_seconds"], "device": "cpu",
                      "test_data_sha256": test["data_sha256"],
                      "limitation": "Previously inspected FarsTail regression benchmark; provisional single-reviewer negative feedback is not independent social-media acceptance."}
        manifest = {"version": 1, "release_id": release_id, "bundle": old_manifest["bundle"],
                    "created_at": datetime.now(timezone.utc).isoformat(), "files": files, "evaluation": evaluation,
                    "feedback": {**report["feedback"], "split_pairs": report["split_pairs"],
                                 "selected_epoch": report["selected_epoch"], "provisional": True,
                                 "social_media_acceptance_complete": False},
                    "previous_release_id": previous["release_id"]}
        atomic_json(manifest_path, manifest)
        verified = verify_model_release(bundle)
    except Exception:
        rollback = destination.with_suffix(".rollback")
        os.link(backup, rollback)
        os.replace(rollback, destination)
        atomic_json(manifest_path, old_manifest)
        raise
    result = {"release_id": verified["release_id"], "verified": True,
              "manifest_sha256": verified["manifest_sha256"], "local_release_updated": True,
              "server_deployed": False, "candidate_model_sha256": report["candidate_model_sha256"]}
    atomic_json(experiment / "promotion_report.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--release-id", required=True)
    args = parser.parse_args()
    print(json.dumps(promote(args.experiment, ROOT / "models", args.release_id), indent=2))


if __name__ == "__main__":
    main()

"""Cache verified baseline predictions before adapting to human feedback."""

from __future__ import annotations

import argparse
import gc
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acceptance_review import review_pairs
from scripts.train_human_feedback import MEMBERS, load_member, predict, save_json
from src.data.loader import load_farstail_split
from src.models.feedback import fingerprint
from src.models.release import file_sha256, verify_model_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(4)
    cases = json.loads(args.cases.read_text())
    release = verify_model_release(ROOT / "models/saved/best_model")
    metadata = {"base_manifest_sha256": release["manifest_sha256"], "max_length": 512,
                "cases_sha256": fingerprint(cases),
                "validation_data_sha256": file_sha256(ROOT / "data/raw/farstail/val.csv"), "members": []}
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = args.output / "manifest.json"
    previous = json.loads(manifest.read_text()) if manifest.is_file() else {}
    if previous and any(previous[k] != v for k, v in metadata.items() if k != "members"):
        raise ValueError("Existing cache belongs to another baseline; use a new directory.")
    by_case = {c["case_id"]: c for c in cases}
    human = [(by_case[r["case_id"]]["review_posts"][r["a"]]["text"],
              by_case[r["case_id"]]["review_posts"][r["b"]]["text"]) for r in review_pairs(cases)]
    validation = [(p.text_a, p.text_b) for p in load_farstail_split("val")]
    for name in MEMBERS:
        source = ROOT / "models/saved" / name
        checksum = file_sha256(source / "model.safetensors")
        archived = next((m for m in previous.get("members", []) if m["name"] == name), None)
        destination = args.output / f"{name}.npz"
        if archived is None or archived["model_sha256"] != checksum or not destination.is_file():
            tokenizer, model = load_member(source)
            np.savez(destination, validation=predict(tokenizer, model, validation), human=predict(tokenizer, model, human))
            del tokenizer, model
            gc.collect()
        metadata["members"].append({"name": name, "model_sha256": checksum, "cache_sha256": file_sha256(destination)})
        save_json(manifest, metadata)
        print(json.dumps({"completed_member": name}), flush=True)


if __name__ == "__main__":
    main()

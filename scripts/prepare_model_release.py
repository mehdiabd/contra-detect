"""Evaluate and fingerprint the local ensemble, or verify a checked-out release."""
import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.release import file_sha256, verify_model_release


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evaluate", action="store_true", help="Run the full FarsTail test before writing release.json")
    args = parser.parse_args()
    model_root = ROOT / "models"
    bundle_dir = model_root / "saved/best_model"
    if not args.evaluate:
        print(json.dumps(verify_model_release(bundle_dir), indent=2))
        return

    import numpy as np
    import torch
    from src.data.loader import load_farstail_split
    from src.models.predictor import ContradictionPredictor
    from scripts.analyze_errors import metrics

    torch.set_num_threads(4)
    bundle_path = bundle_dir / "inference_config.json"
    bundle = json.loads(bundle_path.read_text())
    paths = [bundle_path]
    for member in bundle["members"]:
        folder = (bundle_dir / member["path"]).resolve()
        paths.extend(folder / name for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"))
        paths.extend(folder / name for name in ("spm.model", "special_tokens_map.json") if (folder / name).is_file())
    files = {str(path.relative_to(model_root)): {"bytes": path.stat().st_size, "sha256": file_sha256(path)} for path in paths}
    predictor = ContradictionPredictor(bundle_dir)
    pairs = load_farstail_split("test")
    gold = np.asarray([pair.label for pair in pairs])
    chunks = []
    started = time.monotonic()
    for start in range(0, len(pairs), 16):
        batch = pairs[start:start + 16]
        chunks.append(predictor._scores([p.text_a for p in batch], [p.text_b for p in batch]).numpy())
        if start % 160 == 0:
            print(f"test {min(start + 16, len(pairs))}/{len(pairs)}", flush=True)
    pred = np.concatenate(chunks).argmax(axis=1)
    result = metrics(gold, pred)
    if result["accuracy"] < .85:
        raise RuntimeError("Refusing to publish a release below the 85% FarsTail target")
    payload = {
        "version": 1, "release_id": "farstail-ensemble-8593-v1", "bundle": "saved/best_model",
        "created_at": datetime.now(timezone.utc).isoformat(), "files": files,
        "evaluation": {"dataset": "FarsTail", "split": "test", "scope": "three_class_nli_on_labeled_pairs",
                       "n": len(pairs), "correct": int((gold == pred).sum()),
                       "accuracy": result["accuracy"], "per_class": result["per_class"],
                       "confusion_matrix": result["confusion_matrix"], "matrix_order": result["matrix_order"],
                       "runtime_seconds": round(time.monotonic() - started, 3), "device": str(predictor.device),
                       "test_data_sha256": file_sha256(ROOT / "data/raw/farstail/test.csv"),
                       "limitation": "The test was inspected in earlier project work; this result is not an independent social-media acceptance evaluation."},
    }
    (model_root / "release.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(verify_model_release(bundle_dir), indent=2), flush=True)


if __name__ == "__main__":
    main()

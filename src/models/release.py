"""Verify the exact, evaluated inference files shipped with a model release."""
import hashlib
import json
from pathlib import Path


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_model_release(model_dir: Path) -> dict:
    model_dir = Path(model_dir).resolve()
    root = model_dir.parent.parent
    manifest_path = root / "release.json"
    if not manifest_path.is_file():
        raise ValueError("Missing model release.json; fetch the evaluated release with git lfs pull.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or not manifest.get("release_id"):
        raise ValueError("Unsupported model release manifest")
    if (root / manifest["bundle"]).resolve() != model_dir:
        raise ValueError("Configured model does not match the evaluated release bundle")
    files = manifest.get("files") or {}
    required = {str((model_dir / "inference_config.json").relative_to(root))}
    bundle = json.loads((model_dir / "inference_config.json").read_text(encoding="utf-8"))
    members = []
    for member in bundle["members"]:
        folder = (model_dir / member["path"]).resolve()
        for name in ("config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json"):
            required.add(str((folder / name).relative_to(root)))
        members.append({"name": member["name"], "weight": member["weight"],
                        "model_sha256": files.get(str((folder / "model.safetensors").relative_to(root)), {}).get("sha256")})
    if not required.issubset(files):
        raise ValueError("Model release manifest omits required inference files")
    for relative, expected in files.items():
        path = (root / relative).resolve()
        if not path.is_relative_to(root):
            raise ValueError("Release file lies outside the model directory")
        if not path.is_file() or path.stat().st_size != expected["bytes"]:
            raise ValueError(f"Missing or incomplete release file: {relative}; run git lfs pull.")
        if file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Release checksum mismatch: {relative}")
    evaluation = manifest["evaluation"]
    if (evaluation.get("dataset") != "FarsTail" or evaluation.get("split") != "test"
            or evaluation.get("n", 0) <= 0
            or not 0.85 <= evaluation.get("accuracy", 0) <= 1.0
            or abs(evaluation["correct"] / evaluation["n"] - evaluation["accuracy"]) > 1e-9):
        raise ValueError("Release does not describe the recorded >=85% FarsTail result")
    return {"release_id": manifest["release_id"], "verified": True,
            "manifest_sha256": file_sha256(manifest_path), "verified_files": len(files),
            "members": members, "evaluation": evaluation}

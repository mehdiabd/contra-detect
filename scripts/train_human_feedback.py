"""Adapt one ensemble member to provisional binary feedback without training on test users."""

from __future__ import annotations

import argparse
import gc
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.acceptance_review import binary_metrics, pair_key, review_pairs
from scripts.analyze_errors import metrics
from src.data.loader import load_farstail_split
from src.data.schema import Post
from src.features.candidates import CandidateBuilder
from src.models.feedback import (NOT_CONTRADICTION, fingerprint, isolated_output, partial_label_loss,
                                 read_feedback, split_feedback, validation_eligible)
from src.models.predictor import canonical_label_order
from src.models.release import file_sha256, verify_model_release
from src.preprocess.text_processor import nli_text

MEMBERS = ("targeted_v1", "parsnlu_farstail", "mdeberta_nli")
WEIGHTS = np.asarray([.25, .25, .5], dtype=np.float32)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def predict(tokenizer, model, pairs: list[tuple[str, str]], batch_size: int = 16) -> np.ndarray:
    order = canonical_label_order(model.config)
    chunks = []
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(pairs), batch_size):
            batch = pairs[start:start + batch_size]
            encoded = tokenizer([nli_text(a) for a, _ in batch], [nli_text(b) for _, b in batch],
                                truncation=True, padding=True, max_length=512, return_tensors="pt")
            chunks.append(torch.softmax(model(**encoded).logits, dim=-1)[:, order].numpy())
    return np.concatenate(chunks)


def load_member(path: Path):
    tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)
    model = AutoModelForSequenceClassification.from_pretrained(path, local_files_only=True).float()
    return tokenizer, model


def human_context(cases: list[dict]) -> tuple[dict, np.ndarray]:
    rows = review_pairs(cases)
    selected = []
    builder = CandidateBuilder()
    for case in cases:
        candidates = builder.build([Post.from_dict(p) for p in case["review_posts"]])
        keys = {frozenset(((str(p.post_id_a), p.text_a), (str(p.post_id_b), p.text_b))) for p in candidates}
        for row in (r for r in rows if r["case_id"] == case["case_id"]):
            posts = case["review_posts"]
            selected.append(pair_key(posts[row["a"]], posts[row["b"]]) in keys)
    return {row["pair_id"]: i for i, row in enumerate(rows)}, np.asarray(selected, dtype=bool)


def human_metrics(probs: np.ndarray, examples: list, indexes: dict, selected: np.ndarray) -> dict:
    positions = np.asarray([indexes[p.pair_id] for p in examples])
    values = probs[positions, 1]
    truth = np.asarray([p.label == 1 for p in examples])
    decisions = selected[positions] & (values >= .55)
    negative = values[~truth]
    negative_loss = float(-np.log(np.clip(1 - negative, 1e-9, 1)).mean()) if len(negative) else None
    result = binary_metrics(truth.tolist(), decisions.tolist())
    result["negative_loss"] = negative_loss
    result["negative_only"] = not bool(truth.any())
    return result


def validation_record(val_probs, human_probs, val_gold, examples, indexes, selected) -> dict:
    human = human_metrics(human_probs, examples, indexes, selected)
    return {"farstail": metrics(val_gold, val_probs.argmax(axis=1)), "human": human,
            "human_negative_loss": human["negative_loss"]}


def load_baseline(cache: Path, cases: list[dict], release: dict, val_gold: np.ndarray):
    manifest = json.loads((cache / "manifest.json").read_text())
    if (manifest["base_manifest_sha256"] != release["manifest_sha256"] or manifest["cases_sha256"] != fingerprint(cases)
            or manifest["validation_data_sha256"] != file_sha256(ROOT / "data/raw/farstail/val.csv")
            or manifest["max_length"] != 512):
        raise ValueError("The baseline cache does not match the verified model and data.")
    recorded = {m["name"]: m["model_sha256"] for m in manifest["members"]}
    validation, human = [], []
    n_human = len(review_pairs(cases))
    for name in MEMBERS:
        if recorded.get(name) != file_sha256(ROOT / "models/saved" / name / "model.safetensors"):
            raise ValueError("Baseline cache model checksum mismatch.")
        entry = next(m for m in manifest["members"] if m["name"] == name)
        if entry.get("cache_sha256") and entry["cache_sha256"] != file_sha256(cache / f"{name}.npz"):
            raise ValueError("Baseline cache contents changed.")
        with np.load(cache / f"{name}.npz", allow_pickle=False) as archive:
            v, h = archive["validation"], archive["human"]
        if v.shape != (len(val_gold), 3) or h.shape != (n_human, 3):
            raise ValueError("Baseline cache dimensions do not match the data.")
        if not all(np.isfinite(a).all() and np.allclose(a.sum(axis=1), 1, atol=1e-5) for a in (v, h)):
            raise ValueError("Invalid cached probabilities.")
        validation.append(v)
        human.append(h)
    return np.asarray(validation), np.asarray(human)


def replay_records(examples: list, seed: int, per_class: int) -> list[dict]:
    human_texts = {text for p in examples for text in (p.text_a, p.text_b)}
    heldout = {(nli_text(p.text_a), nli_text(p.text_b))
               for split in ("val", "test") for p in load_farstail_split(split)}
    pool = load_farstail_split("train")
    rng = np.random.default_rng(seed)
    records = []
    for label in range(3):
        choices = [p for p in pool if p.label == label and nli_text(p.text_a) not in human_texts
                   and nli_text(p.text_b) not in human_texts and (nli_text(p.text_a), nli_text(p.text_b)) not in heldout]
        if len(choices) < per_class:
            raise ValueError("Insufficient replay examples.")
        for i in rng.choice(len(choices), size=per_class, replace=False):
            p = choices[i]
            records.append({"text_a": nli_text(p.text_a), "text_b": nli_text(p.text_b), "label": label})
    return records


def trainable_modules(model) -> list:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    modules = [model.base_model.encoder.layer[-1], model.classifier]
    pooler = getattr(model, "pooler", getattr(model.base_model, "pooler", None))
    if pooler is not None:
        modules.append(pooler)
    for module in modules:
        for parameter in module.parameters():
            parameter.requires_grad_(True)
    return modules


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--baseline-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20261103)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--replay-per-class", type=int, default=200)
    parser.add_argument("--feedback-repeat", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()
    if min(args.epochs, args.replay_per_class, args.feedback_repeat, args.batch_size) < 1:
        parser.error("Training counts must be positive.")
    output = isolated_output(args.output, ROOT / "models/saved")
    cases = json.loads(args.cases.read_text())
    examples, metadata = read_feedback(cases, json.loads(args.review.read_text()))
    splits = split_feedback(examples, args.seed)
    if not all(any(p.label == NOT_CONTRADICTION for p in rows) for rows in splits.values()):
        raise ValueError("Each feedback split needs negative examples for the adaptation objective.")
    release = verify_model_release(ROOT / "models/saved/best_model")
    bundle = json.loads((ROOT / "models/saved/best_model/inference_config.json").read_text())
    if [Path(m["path"]).name for m in bundle["members"]] != list(MEMBERS) or not np.allclose(
            [m["weight"] for m in bundle["members"]], WEIGHTS):
        raise ValueError("Unsupported production ensemble.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    val_pairs = load_farstail_split("val")
    val_gold = np.asarray([p.label for p in val_pairs])
    baseline_val, baseline_human = load_baseline(args.baseline_cache, cases, release, val_gold)
    ensemble_val = np.einsum("m,mnc->nc", WEIGHTS, baseline_val)
    ensemble_human = np.einsum("m,mnc->nc", WEIGHTS, baseline_human)
    indexes, selected = human_context(cases)
    baseline = validation_record(ensemble_val, ensemble_human, val_gold, splits["validation"], indexes, selected)
    records = replay_records(examples, args.seed, args.replay_per_class)
    feedback_records = [{"text_a": p.text_a, "text_b": p.text_b, "label": p.label} for p in splits["train"]]
    records.extend(feedback_records * args.feedback_repeat)
    output.mkdir(parents=True)
    save_json(output / "private_training_records.json", {"records": records})
    save_json(output / "private_feedback_splits.json", {name: [p.to_dict() for p in rows] for name, rows in splits.items()})
    protocol = {"seed": args.seed, "epochs_requested": args.epochs, "epochs_completed": 0,
                "learning_rate": args.learning_rate, "batch_size": args.batch_size, "max_length": 512,
                "replay_per_class": args.replay_per_class, "feedback_repeat": args.feedback_repeat,
                "heldout_pairs_excluded_from_replay": True,
                "training_records_sha256": fingerprint(records), "base_release": release["release_id"],
                "base_manifest_sha256": release["manifest_sha256"], "feedback": metadata,
                "split_cases": {name: sorted({p.case_id for p in rows}) for name, rows in splits.items()},
                "split_pairs": {name: len(rows) for name, rows in splits.items()},
                "baseline_validation": baseline, "history": [], "selected_epoch": 0,
                "eligible_for_promotion": False, "production_model_replaced": False,
                "limitations": ["One provisional reviewer, negative human examples only; uncertain pairs excluded.",
                                "This feedback set is no longer an untouched organizational acceptance set.",
                                "Human test users are held out of gradient updates and epoch selection.",
                                "FarsTail test was inspected in earlier project work; this is a regression benchmark.",
                                "Positive controls come from FarsTail, not independently labeled social-media contradictions."]}
    save_json(output / "training_report.json", protocol)
    tokenizer, model = load_member(ROOT / "models/saved/mdeberta_nli")
    modules = trainable_modules(model)
    order = canonical_label_order(model.config)
    parameters = [p for p in model.parameters() if p.requires_grad]
    protocol["trainable_parameters"] = sum(p.numel() for p in parameters)
    encoded = tokenizer([p["text_a"] for p in records], [p["text_b"] for p in records],
                        truncation=True, padding=False, max_length=512)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=.01)
    steps = args.epochs * math.ceil(len(records) / args.batch_size)
    warmup = max(1, int(.1 * steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: min((step + 1) / warmup, max(0, (steps - step) / max(1, steps - warmup))))
    human_rows = review_pairs(cases)
    by_case = {c["case_id"]: c for c in cases}
    human_texts = [(by_case[r["case_id"]]["review_posts"][r["a"]]["text"],
                    by_case[r["case_id"]]["review_posts"][r["b"]]["text"]) for r in human_rows]
    val_texts = [(p.text_a, p.text_b) for p in val_pairs]
    best_loss = baseline["human_negative_loss"]
    started, global_step = time.monotonic(), 0
    print(json.dumps({"stage": "training_started", "records": len(records), "steps": steps,
                      "baseline_val_accuracy": baseline["farstail"]["accuracy"],
                      "baseline_human_val_loss": best_loss}), flush=True)
    for epoch in range(args.epochs):
        model.eval()
        for module in modules:
            module.train()
        permutation = np.random.default_rng(args.seed + epoch).permutation(len(records)).tolist()
        batches = []
        for start in range(0, len(permutation), 64):
            bucket = sorted(permutation[start:start + 64], key=lambda i: len(encoded["input_ids"][i]))
            batches.extend(bucket[i:i + args.batch_size] for i in range(0, len(bucket), args.batch_size))
        random.Random(args.seed + epoch).shuffle(batches)
        losses = []
        for indices in batches:
            inputs = tokenizer.pad([{key: value[i] for key, value in encoded.items()} for i in indices],
                                   padding=True, return_tensors="pt")
            labels = torch.tensor([records[i]["label"] for i in indices], dtype=torch.long)
            optimizer.zero_grad(set_to_none=True)
            loss = partial_label_loss(model(**inputs).logits[:, order], labels)
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite feedback training loss.")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1.)
            optimizer.step()
            scheduler.step()
            losses.append(float(loss.detach()))
            global_step += 1
            if global_step % 10 == 0:
                print(json.dumps({"epoch": epoch + 1, "step": global_step, "steps": steps,
                                  "loss": losses[-1], "elapsed_seconds": round(time.monotonic() - started)}), flush=True)
        candidate_val = predict(tokenizer, model, val_texts)
        candidate_human = predict(tokenizer, model, human_texts)
        val_probs = .25 * baseline_val[0] + .25 * baseline_val[1] + .5 * candidate_val
        human_probs = .25 * baseline_human[0] + .25 * baseline_human[1] + .5 * candidate_human
        result = validation_record(val_probs, human_probs, val_gold, splits["validation"], indexes, selected)
        eligible = validation_eligible(baseline, result)
        row = {"epoch": epoch + 1, "mean_loss": float(np.mean(losses)), "validation": result, "eligible": eligible}
        protocol["history"].append(row)
        protocol["epochs_completed"] = epoch + 1
        if eligible and result["human_negative_loss"] < best_loss:
            best_loss = result["human_negative_loss"]
            model.save_pretrained(output / "candidate")
            tokenizer.save_pretrained(output / "candidate")
            np.savez(output / "selected_validation.npz", validation=val_probs, human=human_probs)
            protocol["selected_epoch"] = epoch + 1
        save_json(output / "training_report.json", protocol)
        print(json.dumps({"epoch": epoch + 1, "validation_accuracy": result["farstail"]["accuracy"],
                          "validation_contradiction_f1": result["farstail"]["per_class"]["contradiction"]["f1"],
                          "human_validation_loss": result["human_negative_loss"], "eligible": eligible,
                          "selected_epoch": protocol["selected_epoch"]}), flush=True)
    if protocol["selected_epoch"] == 0:
        model.save_pretrained(output / "rejected_candidate")
        tokenizer.save_pretrained(output / "rejected_candidate")
        protocol["decision"] = "validation_rejected_keep_verified_production"
        save_json(output / "training_report.json", protocol)
        print(json.dumps({"decision": protocol["decision"]}), flush=True)
        return
    del model, tokenizer, optimizer
    gc.collect()
    test_pairs = load_farstail_split("test")
    test_gold = np.asarray([p.label for p in test_pairs])
    test_texts = [(p.text_a, p.text_b) for p in test_pairs]
    test_chunks = []
    test_started = time.monotonic()
    for name in MEMBERS:
        path = output / "candidate" if name == "mdeberta_nli" else ROOT / "models/saved" / name
        tokenizer, model = load_member(path)
        test_chunks.append(predict(tokenizer, model, test_texts))
        del tokenizer, model
        gc.collect()
        print(json.dumps({"stage": "test_member_complete", "member": name}), flush=True)
    test_probs = np.einsum("m,mnc->nc", WEIGHTS, np.asarray(test_chunks))
    test_result = metrics(test_gold, test_probs.argmax(axis=1))
    with np.load(output / "selected_validation.npz", allow_pickle=False) as archive:
        human_probs = archive["human"]
    human_baseline = human_metrics(ensemble_human, splits["test"], indexes, selected)
    human_candidate = human_metrics(human_probs, splits["test"], indexes, selected)
    gates = {"farstail_accuracy_85": test_result["accuracy"] >= .85,
             "contradiction_f1_preserved": test_result["per_class"]["contradiction"]["f1"]
             >= release["evaluation"]["per_class"]["contradiction"]["f1"] - .01,
             "human_test_false_positives_not_increased": human_candidate["fp"] <= human_baseline["fp"],
             "human_test_loss_improved": human_candidate["negative_loss"] < human_baseline["negative_loss"]}
    protocol.update({"test": {"farstail": test_result, "n": len(test_pairs),
                              "correct": int((test_gold == test_probs.argmax(axis=1)).sum()),
                              "data_sha256": file_sha256(ROOT / "data/raw/farstail/test.csv"),
                              "runtime_seconds": round(time.monotonic() - test_started, 3),
                              "human_baseline": human_baseline, "human_candidate": human_candidate},
                     "promotion_gates": gates, "eligible_for_promotion": all(gates.values()),
                     "candidate_model_sha256": file_sha256(output / "candidate/model.safetensors"),
                     "human_all_baseline": human_metrics(ensemble_human, examples, indexes, selected),
                     "human_all_candidate": human_metrics(human_probs, examples, indexes, selected)})
    np.savez(output / "selected_test.npz", probabilities=test_probs, gold=test_gold)
    save_json(output / "training_report.json", protocol)
    print(json.dumps({"stage": "completed", "selected_epoch": protocol["selected_epoch"],
                      "test_accuracy": test_result["accuracy"], "promotion_gates": gates,
                      "eligible_for_promotion": protocol["eligible_for_promotion"]}), flush=True)


if __name__ == "__main__":
    main()

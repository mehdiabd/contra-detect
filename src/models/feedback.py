"""Validate human feedback and keep its training, validation and test users separate."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path

import torch

from src.preprocess.text_processor import nli_text

NOT_CONTRADICTION = -1
LABELS = {"contradiction": 1, "not_contradiction": NOT_CONTRADICTION}


def fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class FeedbackPair:
    pair_id: str
    case_id: str
    platform: str
    text_a: str
    text_b: str
    label: int

    def to_dict(self) -> dict:
        return asdict(self)


def read_feedback(cases: list[dict], review: dict) -> tuple[list[FeedbackPair], dict]:
    if review.get("cases_sha256") != fingerprint(cases):
        raise ValueError("Feedback belongs to a different case snapshot.")
    if not str(review.get("reviewer") or "").strip():
        raise ValueError("The human reviewer must be identified.")
    expected = {}
    for case in cases:
        for number, (a, b) in enumerate(combinations(case["review_posts"], 2), 1):
            pair_id = f'{case["case_id"]}-p{number:02}'
            expected[pair_id] = (case, a, b)
    annotations = {}
    for row in review.get("labels", []):
        pair_id, label = row.get("pair_id"), row.get("label")
        if pair_id not in expected or pair_id in annotations or label not in {*LABELS, "uncertain"}:
            raise ValueError("Feedback contains an unknown, duplicate or invalid annotation.")
        annotations[pair_id] = label
    if set(annotations) != set(expected):
        raise ValueError("The review must account for every requested pair.")
    examples = []
    for pair_id, (case, a, b) in expected.items():
        label = annotations[pair_id]
        if label == "uncertain":
            continue
        examples.append(FeedbackPair(pair_id, case["case_id"], case["platform"],
                                     nli_text(a["text"]), nli_text(b["text"]), LABELS[label]))
    metadata = {"cases_sha256": fingerprint(cases), "review_sha256": fingerprint(review),
                "annotated": len(annotations), "usable": len(examples),
                "uncertain_excluded": sum(value == "uncertain" for value in annotations.values()),
                "positive": sum(pair.label == 1 for pair in examples),
                "negative": sum(pair.label == NOT_CONTRADICTION for pair in examples),
                "reviewer_count": 1, "adjudicated": False}
    return examples, metadata


def split_feedback(examples: list[FeedbackPair], seed: int) -> dict[str, list[FeedbackPair]]:
    """Reserve one validation and one test user per platform before any adaptation."""
    groups = defaultdict(set)
    for pair in examples:
        groups[pair.platform].add(pair.case_id)
    assignments = {}
    rng = random.Random(seed)
    for platform, users in sorted(groups.items()):
        users = sorted(users)
        if len(users) < 3:
            raise ValueError(f"Need at least three annotated users for {platform}.")
        rng.shuffle(users)
        assignments.update({user: "train" for user in users[2:]})
        assignments[users[0]], assignments[users[1]] = "validation", "test"
    splits = {name: [] for name in ("train", "validation", "test")}
    for pair in examples:
        splits[assignments[pair.case_id]].append(pair)
    text_sets = {name: {text for p in pairs for text in (p.text_a, p.text_b)} for name, pairs in splits.items()}
    for a, b in combinations(splits, 2):
        if text_sets[a] & text_sets[b]:
            raise ValueError("Identical posts cross feedback splits; group shared texts before training.")
    return splits


def partial_label_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Binary non-contradiction means entailment OR neutral, never a forced neutral label."""
    if logits.ndim != 2 or logits.shape[1] != 3 or labels.shape != (logits.shape[0],):
        raise ValueError("Expected canonical three-class logits and one label per row.")
    if not torch.isin(labels, labels.new_tensor([-1, 0, 1, 2])).all():
        raise ValueError("Unsupported training label.")
    log_probs = torch.log_softmax(logits, dim=-1)
    ordinary = -log_probs.gather(1, labels.clamp_min(0).unsqueeze(1)).squeeze(1)
    not_contradiction = -torch.logsumexp(log_probs[:, [0, 2]], dim=-1)
    return torch.where(labels == NOT_CONTRADICTION, not_contradiction, ordinary).mean()


def isolated_output(path: Path, saved_models: Path) -> Path:
    resolved = path.resolve()
    if resolved.is_relative_to(saved_models.resolve()):
        raise ValueError("Feedback training cannot write into the production model directory.")
    if resolved.exists():
        raise ValueError("The feedback experiment directory already exists; preserve previous runs.")
    return resolved


def validation_eligible(baseline: dict, candidate: dict) -> bool:
    """Choose without test labels; preserve NLI quality while lowering negative-review loss."""
    return (candidate["farstail"]["accuracy"] >= max(.85, baseline["farstail"]["accuracy"] - .005)
            and candidate["farstail"]["per_class"]["contradiction"]["f1"]
            >= baseline["farstail"]["per_class"]["contradiction"]["f1"] - .005
            and candidate["human_negative_loss"] < baseline["human_negative_loss"])

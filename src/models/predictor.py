import os
import json
import math
from pathlib import Path
from typing import Sequence, Union

import config  # noqa: F401
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import BEST_MODEL_DIR, CONTRADICTION_LABEL_ID, FALLBACK_MODEL, MODEL_CONFIG, NLI_ID2LABEL
from src.data.schema import Pair
from src.preprocess.text_processor import nli_text

_CONTRA_ALIASES = {"contradiction", "contradicts", "contradict", "c"}
_LABEL_ALIASES = {
    "entails": "entailment", "entail": "entailment", "e": "entailment",
    "contradicts": "contradiction", "contradict": "contradiction", "c": "contradiction",
    "n": "neutral",
}


def canonical_label_order(model_config) -> list[int]:
    """Align every model's columns before averaging: pretrained NLI label orders vary."""
    raw = getattr(model_config, "id2label", None) or NLI_ID2LABEL
    mapped = {}
    for key, value in raw.items():
        idx = int(key)
        name = str(value).strip().lower()
        if name.startswith("label_"):
            name = NLI_ID2LABEL.get(idx, name)
        mapped[idx] = _LABEL_ALIASES.get(name, name)
    if len(mapped) != 3 or set(mapped.values()) != set(NLI_ID2LABEL.values()):
        raise ValueError(f"Unsupported NLI label mapping: {raw}")
    by_name = {name: idx for idx, name in mapped.items()}
    return [by_name[NLI_ID2LABEL[i]] for i in range(3)]


def resolve_model_source() -> str:
    local = Path(BEST_MODEL_DIR)
    if (local / "config.json").exists() or (local / "inference_config.json").exists():
        return str(local)
    return FALLBACK_MODEL


def resolve_ensemble_sources() -> list[str]:
    raw = (os.getenv("ENSEMBLE_MODELS") or "").strip()
    if not raw:
        return []
    models = []
    for item in raw.split(","):
        name = item.strip()
        if not name:
            continue
        path = Path(name)
        models.append(str(path) if (path / "config.json").exists() else name)
    return models


def load_inference_bundle(path: Union[str, Path]) -> tuple[list[str], list[float], int]:
    root = Path(path)
    payload = json.loads((root / "inference_config.json").read_text(encoding="utf-8"))
    if payload.get("version") != 1 or not payload.get("members"):
        raise ValueError("Unsupported or empty inference bundle")
    sources, weights = [], []
    for member in payload["members"]:
        source = (root / member["path"]).resolve()
        if not (source / "config.json").exists():
            raise ValueError(f"Missing local ensemble member: {source}")
        weight = float(member["weight"])
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Ensemble weights must be finite and nonnegative")
        sources.append(str(source))
        weights.append(weight)
    total = sum(weights)
    if total <= 0:
        raise ValueError("At least one ensemble weight must be positive")
    max_length = int(payload.get("max_length", MODEL_CONFIG["max_length"]))
    if not 1 <= max_length <= 512:
        raise ValueError("Unsupported ensemble token limit")
    return sources, [w / total for w in weights], max_length


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ContradictionPredictor:
    def __init__(self, model_dir: Union[str, Path, None] = None):
        self.device = _pick_device()
        self.max_length = MODEL_CONFIG["max_length"]
        sources = [str(model_dir)] if model_dir else resolve_ensemble_sources()
        if not sources:
            sources = [resolve_model_source()]
        self.member_weights = [1 / len(sources)] * len(sources)
        if len(sources) == 1 and (Path(sources[0]) / "inference_config.json").exists():
            sources, self.member_weights, self.max_length = load_inference_bundle(sources[0])
        self.model_dir = sources[0]
        self.members = []
        self.member_label_orders = []
        for source in sources:
            tokenizer = AutoTokenizer.from_pretrained(source)
            model = AutoModelForSequenceClassification.from_pretrained(source)
            model.float()
            self.member_label_orders.append(canonical_label_order(model.config))
            model.eval()
            model.to(self.device)
            self.members.append((tokenizer, model))
        self.tokenizer, self.model = self.members[0]
        self.id2label = dict(NLI_ID2LABEL)
        self.contradiction_index = CONTRADICTION_LABEL_ID

    def _resolve_id2label(self) -> dict[int, str]:
        raw = getattr(self.model.config, "id2label", None) or NLI_ID2LABEL
        mapped = {}
        for key, value in raw.items():
            idx = int(key)
            name = str(value).lower()
            if name.startswith("label_"):
                name = NLI_ID2LABEL.get(idx, name)
            mapped[idx] = name
        return mapped or NLI_ID2LABEL

    def _resolve_contradiction_index(self) -> int:
        for idx, name in self.id2label.items():
            if name in _CONTRA_ALIASES:
                return idx
        return CONTRADICTION_LABEL_ID

    @classmethod
    def from_pretrained(cls, model_dir: Union[str, Path, None] = None) -> "ContradictionPredictor":
        return cls(model_dir)

    def _scores(self, text_a: list[str], text_b: list[str]) -> torch.Tensor:
        probs = []
        with torch.no_grad():
            for (tokenizer, model), order in zip(self.members, self.member_label_orders):
                encoded = tokenizer(
                    text_a,
                    text_b,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                encoded = {k: v.to(self.device) for k, v in encoded.items()}
                probs.append(torch.softmax(model(**encoded).logits, dim=1)[:, order])
        stacked = torch.stack(probs, dim=0)
        weights = torch.tensor(self.member_weights, dtype=stacked.dtype, device=stacked.device)
        return (stacked * weights[:, None, None]).sum(dim=0).cpu()

    def predict_pairs(self, pairs: Sequence[Pair], batch_size: int = 16) -> list[dict]:
        if not pairs:
            return []
        results = []
        for start in range(0, len(pairs), batch_size):
            chunk = list(pairs[start : start + batch_size])
            left = [nli_text(p.text_a) for p in chunk]
            right = [nli_text(p.text_b) for p in chunk]
            forward = self._scores(left, right)
            reverse = self._scores(right, left)
            fwd_c = forward[:, self.contradiction_index]
            rev_c = reverse[:, self.contradiction_index]
            # Entailment is directional: keep the supplied premise/hypothesis order.
            # Retain the symmetric score as a separate diagnostic, not the API decision.
            bidirectional = torch.maximum(fwd_c, rev_c)
            labels = torch.argmax(forward, dim=1).tolist()
            for pair, label, prob_row, contra_score, symmetric_score in zip(
                chunk, labels, forward, fwd_c, bidirectional
            ):
                prob_list = [float(x) for x in prob_row]
                results.append(
                    {
                        "pair": pair,
                        "label_id": int(label),
                        "label": self.id2label.get(int(label), str(label)),
                        "probs": {self.id2label.get(i, str(i)): prob_list[i] for i in range(len(prob_list))},
                        "contradiction_score": float(contra_score),
                        "bidirectional_contradiction_score": float(symmetric_score),
                    }
                )
        return results

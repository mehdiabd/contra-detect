from pathlib import Path
from typing import Sequence, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import BEST_MODEL_DIR, CONTRADICTION_LABEL_ID, FALLBACK_MODEL, MODEL_CONFIG, NLI_ID2LABEL
from src.data.schema import Pair
from src.preprocess.text_processor import TextProcessor

_CONTRA_ALIASES = {"contradiction", "contradicts", "contradict", "c"}


def resolve_model_source() -> str:
    local = Path(BEST_MODEL_DIR)
    if (local / "config.json").exists():
        return str(local)
    return FALLBACK_MODEL


def _pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ContradictionPredictor:
    def __init__(self, model_dir: Union[str, Path, None] = None):
        self.model_dir = str(model_dir or resolve_model_source())
        self.processor = TextProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_dir)
        self.model.eval()
        self.device = _pick_device()
        self.model.to(self.device)
        self.id2label = self._resolve_id2label()
        self.contradiction_index = self._resolve_contradiction_index()

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
        encoded = self.tokenizer(
            text_a,
            text_b,
            truncation=True,
            padding=True,
            max_length=MODEL_CONFIG["max_length"],
            return_tensors="pt",
        )
        encoded = {k: v.to(self.device) for k, v in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
            return torch.softmax(logits, dim=1).cpu()

    def predict_pairs(self, pairs: Sequence[Pair], batch_size: int = 16) -> list[dict]:
        if not pairs:
            return []
        results = []
        for start in range(0, len(pairs), batch_size):
            chunk = list(pairs[start : start + batch_size])
            left = [self.processor.normalize(p.text_a) for p in chunk]
            right = [self.processor.normalize(p.text_b) for p in chunk]
            forward = self._scores(left, right)
            reverse = self._scores(right, left)
            fwd_conf = forward.max(dim=1).values
            rev_conf = reverse.max(dim=1).values
            chosen = torch.where(rev_conf.unsqueeze(1) > fwd_conf.unsqueeze(1), reverse, forward)
            contra = chosen[:, self.contradiction_index]
            labels = torch.argmax(chosen, dim=1).tolist()
            for pair, label, prob_row, contra_score in zip(chunk, labels, chosen, contra):
                prob_list = [float(x) for x in prob_row]
                results.append(
                    {
                        "pair": pair,
                        "label_id": int(label),
                        "label": self.id2label.get(int(label), str(label)),
                        "probs": {self.id2label.get(i, str(i)): prob_list[i] for i in range(len(prob_list))},
                        "contradiction_score": float(contra_score),
                    }
                )
        return results

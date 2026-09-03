from pathlib import Path
from typing import Sequence, Union

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from config import BEST_MODEL_DIR, CONTRADICTION_LABEL_ID, MODEL_CONFIG, NLI_ID2LABEL
from src.data.schema import Pair
from src.preprocess.text_processor import TextProcessor


class ContradictionPredictor:
    def __init__(self, model_dir: Union[str, Path] = BEST_MODEL_DIR):
        self.model_dir = Path(model_dir)
        self.processor = TextProcessor()
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_dir))
        self.model = AutoModelForSequenceClassification.from_pretrained(str(self.model_dir))
        self.model.eval()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    @classmethod
    def from_pretrained(cls, model_dir: Union[str, Path] = BEST_MODEL_DIR) -> "ContradictionPredictor":
        return cls(model_dir)

    def predict_pairs(self, pairs: Sequence[Pair], batch_size: int = 16) -> list[dict]:
        if not pairs:
            return []
        results = []
        for start in range(0, len(pairs), batch_size):
            chunk = list(pairs[start : start + batch_size])
            text_a = [self.processor.normalize(p.text_a) for p in chunk]
            text_b = [self.processor.normalize(p.text_b) for p in chunk]
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
                probs = torch.softmax(logits, dim=1).cpu()
            labels = torch.argmax(probs, dim=1).tolist()
            for pair, label, prob_row in zip(chunk, labels, probs):
                prob_list = [float(x) for x in prob_row]
                results.append(
                    {
                        "pair": pair,
                        "label_id": int(label),
                        "label": NLI_ID2LABEL.get(int(label), str(label)),
                        "probs": {NLI_ID2LABEL[i]: prob_list[i] for i in range(len(prob_list))},
                        "contradiction_score": float(prob_row[CONTRADICTION_LABEL_ID]),
                    }
                )
        return results

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BEST_MODEL_DIR
from src.data.loader import load_farstail, load_pairs_csv
from src.models.predictor import ContradictionPredictor
from src.models.trainer import compute_classification_metrics
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained contradiction model.")
    parser.add_argument("--source", choices=["farstail", "csv"], default="farstail")
    parser.add_argument("--pairs", type=Path, help="Labeled pairs CSV when --source csv")
    parser.add_argument("--model", type=Path, default=BEST_MODEL_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.source == "farstail":
        _, _, pairs = load_farstail()
    else:
        if not args.pairs:
            raise SystemExit("--pairs is required when --source csv")
        pairs = [p for p in load_pairs_csv(args.pairs) if p.label is not None]
        if not pairs:
            raise SystemExit("No labeled pairs found in CSV")

    predictor = ContradictionPredictor(args.model)
    preds = predictor.predict_pairs(pairs)
    labels = np.array([p.label for p in pairs], dtype=int)
    pred_ids = np.array([p["label_id"] for p in preds], dtype=int)
    logits = np.zeros((len(preds), 3), dtype=np.float32)
    logits[np.arange(len(pred_ids)), pred_ids] = 1.0
    metrics = compute_classification_metrics((logits, labels))
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

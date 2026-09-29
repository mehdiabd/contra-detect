import json
import sys
from pathlib import Path

import numpy as np
from scipy.special import softmax

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BEST_MODEL_DIR, FALLBACK_MODEL, MODEL_CONFIG
from src.data.loader import load_farstail
from src.models.trainer import ContradictionTrainer, compute_classification_metrics

OUT = ROOT / "reports" / "farstail_ensemble_metrics.json"
DEFAULT_MODELS = [str(BEST_MODEL_DIR), FALLBACK_MODEL]


def _round(metrics: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def main() -> None:
    models = sys.argv[1:] or DEFAULT_MODELS
    _, _, test = load_farstail()
    labels = None
    stacked = []
    per_model = []
    for name in models:
        print(f"scoring {name}")
        trainer = ContradictionTrainer(model_name=name)
        logits, gold = trainer.predict_logits(test)
        labels = gold
        metrics = compute_classification_metrics((logits, gold))
        per_model.append({"model": name, "metrics": _round(metrics)})
        print("  ", _round(metrics))
        stacked.append(softmax(logits, axis=1))

    mean_probs = np.mean(np.stack(stacked, axis=0), axis=0)
    ensemble = compute_classification_metrics((mean_probs, labels))
    accuracy = float(ensemble["accuracy"])
    payload = {
        "source": "FarsTail test",
        "models": per_model,
        "ensemble": _round(ensemble),
        "target_accuracy": MODEL_CONFIG["target_accuracy"],
        "meets_target": accuracy >= MODEL_CONFIG["target_accuracy"],
        "gap_to_target": round(MODEL_CONFIG["target_accuracy"] - accuracy, 4),
        "method": "mean softmax",
    }
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("ensemble", _round(ensemble))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()

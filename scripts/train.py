import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BEST_MODEL_DIR, OPTUNA_CONFIG
from src.data.loader import load_farstail, load_pairs_csv, split_pairs
from src.models.trainer import ContradictionTrainer, tune_hyperparameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune ParsBERT for pairwise contradiction detection.")
    parser.add_argument("--source", choices=["farstail", "csv"], default="farstail")
    parser.add_argument("--pairs", type=Path, help="Labeled pairs CSV when --source csv")
    parser.add_argument("--output", type=Path, default=BEST_MODEL_DIR)
    parser.add_argument("--tune", action="store_true", help="Light Optuna search before final training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for smoke training")
    return parser.parse_args()


def load_splits(args: argparse.Namespace):
    if args.source == "farstail":
        return load_farstail()
    if not args.pairs:
        raise SystemExit("--pairs is required when --source csv")
    return split_pairs(load_pairs_csv(args.pairs))


def main() -> None:
    args = parse_args()
    train_pairs, val_pairs, test_pairs = load_splits(args)
    if args.max_samples:
        train_pairs = train_pairs[: args.max_samples]
        val_pairs = val_pairs[: max(8, args.max_samples // 5)]
        test_pairs = test_pairs[: max(8, args.max_samples // 5)]
    print(f"splits: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")

    overrides = {}
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    if args.tune:
        print("running light hyperparameter search...")
        best = tune_hyperparameters(train_pairs, val_pairs, n_trials=OPTUNA_CONFIG["n_trials"])
        print(json.dumps(best, ensure_ascii=False, indent=2))
        overrides.update(best["best_params"])

    trainer = ContradictionTrainer()
    val_metrics = trainer.train(train_pairs, val_pairs, output_dir=args.output, **overrides)
    test_metrics = trainer.evaluate(test_pairs)
    print("validation:", {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)})
    print("test:", {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)})
    print(f"saved model to {args.output}")


if __name__ == "__main__":
    main()

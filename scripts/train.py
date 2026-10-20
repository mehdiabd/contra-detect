import argparse
import hashlib
import platform
from importlib.metadata import version
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import SAVED_MODELS_DIR, MODEL_CONFIG, OPTUNA_CONFIG, FARSTAIL_DIR
from src.data.loader import load_farstail, load_pairs_csv, split_pairs
from src.models.trainer import ContradictionTrainer, tune_hyperparameters


def _round_metrics(metrics: dict) -> dict:
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a Persian NLI model for pairwise contradiction detection.")
    parser.add_argument("--source", choices=["farstail", "csv"], default="farstail")
    parser.add_argument("--pairs", type=Path, help="Labeled pairs CSV when --source csv")
    parser.add_argument("--output", type=Path, default=ROOT / "models" / "experiments" / "general" / "model")
    parser.add_argument("--metrics-output", type=Path, help="Defaults to metrics.json beside the experiment model")
    parser.add_argument("--cpu", action="store_true", help="Use CPU for reproducible smoke training")
    parser.add_argument("--base-model", default=MODEL_CONFIG["base_model"])
    parser.add_argument("--tune", action="store_true", help="Light Optuna search before final training")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-samples", type=int, default=None, help="Optional cap per split for smoke training")
    parser.add_argument(
        "--final-train-val",
        action="store_true",
        default=True,
        help="After val selection, retrain on train+val (FarsTail paper protocol).",
    )
    parser.add_argument("--no-final-train-val", action="store_false", dest="final_train_val")
    return parser.parse_args()


def load_splits(args: argparse.Namespace):
    if args.source == "farstail":
        return load_farstail()
    if not args.pairs:
        raise SystemExit("--pairs is required when --source csv")
    return split_pairs(load_pairs_csv(args.pairs))


def main() -> None:
    args = parse_args()
    if args.output.resolve().is_relative_to(SAVED_MODELS_DIR.resolve()):
        raise SystemExit("Use an isolated --output; production model overwrite is prohibited.")
    if args.max_samples is not None and args.max_samples < 1:
        raise SystemExit("--max-samples must be positive")
    train_pairs, val_pairs, test_pairs = load_splits(args)
    if args.max_samples:
        train_pairs = train_pairs[: args.max_samples]
        val_pairs = val_pairs[: max(8, args.max_samples // 5)]
        test_pairs = test_pairs[: max(8, args.max_samples // 5)]
    print(f"splits: train={len(train_pairs)} val={len(val_pairs)} test={len(test_pairs)}")
    print(f"base_model: {args.base_model}")

    overrides = {"use_cpu": args.cpu}
    if args.epochs is not None:
        overrides["num_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    if args.tune:
        print("running light hyperparameter search...")
        best = tune_hyperparameters(
            train_pairs, val_pairs, n_trials=OPTUNA_CONFIG["n_trials"],
            model_name=args.base_model, output_root=args.output.parent / "tuning", use_cpu=args.cpu,
        )
        print(json.dumps(best, ensure_ascii=False, indent=2))
        overrides.update(best["best_params"])

    phase1_dir = Path(args.output)
    trainer = ContradictionTrainer(model_name=args.base_model)
    val_metrics = trainer.train(train_pairs, val_pairs, output_dir=phase1_dir, **overrides)
    best_epoch = trainer.best_epoch()
    print("validation:", {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)})
    print(f"best_epoch: {best_epoch}")

    final_trainer = trainer
    if args.final_train_val and not args.max_samples:
        print(f"retraining on train+val for {best_epoch} epoch(s)...")
        phase2_overrides = dict(overrides)
        phase2_overrides["num_epochs"] = best_epoch
        phase2_overrides["early_stopping"] = False
        final_trainer = ContradictionTrainer(model_name=args.base_model)
        final_trainer.train(
            list(train_pairs) + list(val_pairs),
            None,
            output_dir=args.output,
            **phase2_overrides,
        )

    test_metrics = final_trainer.evaluate(test_pairs)
    val_rounded = {k: round(v, 4) for k, v in val_metrics.items() if isinstance(v, float)}
    test_rounded = {k: round(v, 4) for k, v in test_metrics.items() if isinstance(v, float)}
    print("validation:", val_rounded)
    print("test:", test_rounded)
    print(f"saved model to {args.output}")

    accuracy = float(test_metrics.get("eval_accuracy") or test_metrics.get("accuracy") or 0.0)
    data_paths = ([FARSTAIL_DIR / f"{split}.csv" for split in ("train", "val", "test")]
                  if args.source == "farstail" else [args.pairs])
    payload = {
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "packages": {name: version(name) for name in ("torch", "transformers", "datasets", "numpy", "scipy", "scikit-learn", "accelerate")},
        },
        "data_sha256": {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in data_paths},
        "source": args.source,
        "sample_counts": {"train": len(train_pairs), "val": len(val_pairs), "test": len(test_pairs)},
        "smoke_test": bool(args.max_samples),
        "seed": MODEL_CONFIG["seed"],
        "training_overrides": overrides,
        "base_model": args.base_model,
        "protocol": "train+val" if args.final_train_val and not args.max_samples else "train-only",
        "best_epoch": best_epoch,
        "target_accuracy": MODEL_CONFIG["target_accuracy"],
        "meets_target": None if args.max_samples else accuracy >= MODEL_CONFIG["target_accuracy"],
        "gap_to_target": None if args.max_samples else round(MODEL_CONFIG["target_accuracy"] - accuracy, 4),
        "model_dir": str(args.output),
        "validation": _round_metrics(val_metrics),
        "test": _round_metrics(test_metrics),
        "note": "Smoke metrics are execution checks, not acceptance evidence." if args.max_samples else "Accuracy is reported on labeled pairs, not on unlabeled social-media posts.",
    }
    metrics_path = args.metrics_output or args.output.parent / "metrics.json"
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {metrics_path}")


if __name__ == "__main__":
    main()

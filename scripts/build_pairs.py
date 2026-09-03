import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from config import BEST_MODEL_DIR, LABELED_DATA_DIR, SAMPLE_DATA_PATH
from src.data import elastic
from src.data.loader import ensure_dir, load_posts_csv, load_posts_json, save_pairs_csv
from src.data.schema import Post
from src.features.candidates import CandidateBuilder
from src.models.predictor import ContradictionPredictor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build same-user candidate pairs and optional weak labels.")
    parser.add_argument("--input", type=Path, default=SAMPLE_DATA_PATH, help="JSON or CSV of posts")
    parser.add_argument("--from-es", action="store_true")
    parser.add_argument("--user-ids", nargs="*", default=None)
    parser.add_argument("--output", type=Path, default=LABELED_DATA_DIR / "pairs.csv")
    parser.add_argument("--weak-label", action="store_true", help="Label pairs with the trained model")
    parser.add_argument("--min-confidence", type=float, default=0.8)
    parser.add_argument("--model", type=Path, default=BEST_MODEL_DIR)
    return parser.parse_args()


def load_posts(args) -> list[Post]:
    if args.from_es:
        if not elastic.is_configured():
            raise SystemExit("Elasticsearch is not enabled. Set ES_ENABLED=true in .env")
        return elastic.fetch_posts(args.user_ids or [])
    suffix = args.input.suffix.lower()
    if suffix == ".json":
        posts = load_posts_json(args.input)
    elif suffix == ".csv":
        posts = load_posts_csv(args.input)
    else:
        raise SystemExit("Input must be .json or .csv")
    if args.user_ids:
        wanted = {str(u) for u in args.user_ids}
        posts = [p for p in posts if p.user_id in wanted]
    return posts


def main() -> None:
    args = parse_args()
    posts = load_posts(args)
    pairs = CandidateBuilder().build(posts)
    if args.weak_label:
        if not (Path(args.model) / "config.json").exists():
            raise SystemExit(f"No trained model at {args.model}. Run scripts/train.py first.")
        predictor = ContradictionPredictor(args.model)
        labeled = []
        for pred in predictor.predict_pairs(pairs):
            confidence = max(pred["probs"].values())
            if confidence >= args.min_confidence:
                pair = pred["pair"]
                pair.label = pred["label_id"]
                labeled.append(pair)
        pairs = labeled
    ensure_dir(args.output.parent)
    save_pairs_csv(pairs, args.output)
    print(f"wrote {len(pairs)} pairs to {args.output}")


if __name__ == "__main__":
    main()
